import numpy as np
import scipy.optimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bound = 5.0
        self.call_count = 0
        self.best_x = None
        self.best_f = np.inf
        self.deadline = time.time() + 290.0

    def __call__(self, func):
        self.call_count = 0
        self.best_x = None
        self.best_f = np.inf
        self.deadline = time.time() + 290.0
        
        def wrapped_func(x):
            self.call_count += 1
            if time.time() > self.deadline:
                raise TimeoutError
            if self.call_count > self.budget:
                raise RuntimeError("Budget exceeded")
            # Clip to bounds
            x = np.clip(x, -self.bound, self.bound)
            fx = func(x)
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = x.copy()
            return fx

        # Phase 1: Global Search with Sobol Sequence
        # Sobol sequences provide better space-filling properties than uniform random
        # for multimodal landscapes like Gallagher f21
        n_global_samples = 80
        candidates = []
        
        # Use a simple quasi-random approach: Halton sequence or just stratified
        # Since scipy.stats.qmc requires newer scipy, we'll use a simple Sobol-like approach
        # or just use np.random with stratification
        
        # Simple stratified sampling: divide the box into sub-boxes and sample one point per sub-box
        # For 5D, 80 points means roughly 2^4=16 sub-boxes per dimension would be 16^5 which is too many
        # Instead, use a grid with jitter
        
        # Alternative: use Latin Hypercube Sampling for better coverage
        # We'll implement a simple LHS
        
        # For simplicity and speed, use stratified random sampling
        # Divide each dimension into n_strata intervals
        n_strata = 4  # 4^5 = 1024 cells, we sample 80 of them
        # Actually, let's just use a good random seed and stratify by dividing into bins
        
        # Better approach: use scipy.stats.qmc.Sobol if available, else fallback
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=self.dim, scramble=True)
            # We need 2^m points for Sobol, but we can generate more and take subset
            # For 80 points, generate 128 (2^7) and take first 80
            n_points = 128
            sobol_points = sampler.random(n_points)
            # Scale to [-5, 5]
            sobol_points = sobol_points * 10.0 - 5.0
            # Take first 80
            for i in range(min(n_global_samples, len(sobol_points))):
                if self.call_count >= self.budget or time.time() > self.deadline:
                    break
                p = sobol_points[i]
                fx = wrapped_func(p)
                candidates.append((fx, p.copy()))
        except ImportError:
            # Fallback to uniform random with stratification
            for i in range(n_global_samples):
                if self.call_count >= self.budget or time.time() > self.deadline:
                    break
                # Stratified: divide each dim into 4 bins
                p = np.zeros(self.dim)
                for d in range(self.dim):
                    bin_idx = (i + d * n_global_samples // self.dim) % 4
                    lower = -self.bound + bin_idx * (2 * self.bound / 4)
                    upper = lower + (2 * self.bound / 4)
                    p[d] = np.random.uniform(lower, upper)
                fx = wrapped_func(p)
                candidates.append((fx, p.copy()))
        
        # Add some pure random points for robustness
        n_random_extra = 20
        for i in range(n_random_extra):
            if self.call_count >= self.budget or time.time() > self.deadline:
                break
            p = np.random.uniform(-self.bound, self.bound, self.dim)
            fx = wrapped_func(p)
            candidates.append((fx, p.copy()))

        if self.best_x is None:
            return np.zeros(self.dim), 0.0

        # Sort candidates by fitness to get top ones
        candidates.sort(key=lambda x: x[0])
        
        # Phase 2: Multi-start Nelder-Mead on top candidates
        # Nelder-Mead is fast and effective for smooth local minima
        top_k = min(3, len(candidates))
        top_candidates = [c[1] for c in candidates[:top_k]]
        
        for idx, x0 in enumerate(top_candidates):
            if self.call_count >= self.budget - 30 or time.time() > self.deadline - 15:
                break
            try:
                res = scipy.optimize.minimize(
                    wrapped_func, 
                    x0, 
                    method='Nelder-Mead',
                    options={
                        'maxiter': 150,
                        'xatol': 1e-10,
                        'fatol': 1e-10,
                        'adaptive': True,
                        'initial_simplex': None  # Let scipy choose
                    }
                )
            except (TimeoutError, RuntimeError, Exception):
                pass

        # Phase 3: Final coordinate descent polish for tight convergence
        if self.call_count < self.budget - 5 and time.time() < self.deadline - 5:
            # Simple coordinate descent with adaptive step
            x = self.best_x.copy()
            f = self.best_f
            step = 0.01
            for _ in range(30):
                if self.call_count >= self.budget - 2 or time.time() > self.deadline - 2:
                    break
                improved = False
                for d in range(self.dim):
                    if self.call_count >= self.budget - 2 or time.time() > self.deadline - 2:
                        break
                    for direction in [1, -1]:
                        x_trial = x.copy()
                        x_trial[d] += direction * step
                        x_trial = np.clip(x_trial, -self.bound, self.bound)
                        try:
                            f_trial = wrapped_func(x_trial)
                            if f_trial < f:
                                x = x_trial
                                f = f_trial
                                improved = True
                                break
                        except (TimeoutError, RuntimeError, Exception):
                            pass
                if not improved:
                    step *= 0.5
                    if step < 1e-12:
                        break
                else:
                    step = min(step * 1.05, 0.1)

            if f < self.best_f:
                self.best_f = f
                self.best_x = x

        return self.best_x, self.best_f
