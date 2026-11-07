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

        # Phase 1: Global Search with Random Sampling
        # Use stratified sampling for better coverage
        n_global_samples = 120
        candidates = []
        for i in range(n_global_samples):
            if self.call_count >= self.budget or time.time() > self.deadline:
                break
            p = np.random.uniform(-self.bound, self.bound, self.dim)
            fx = wrapped_func(p)
            candidates.append((fx, p.copy()))
            
        if self.best_x is None:
            return np.zeros(self.dim), 0.0

        # Sort candidates by fitness to get top ones
        candidates.sort(key=lambda x: x[0])
        top_candidates = [c[1] for c in candidates[:min(3, len(candidates))]]

        # Phase 2: Local Optimization - Multi-start Powell
        # Run Powell from the best candidate first, then from other good ones if budget allows
        for idx, x0 in enumerate(top_candidates):
            if self.call_count >= self.budget - 50 or time.time() > self.deadline - 10:
                break
            try:
                res = scipy.optimize.minimize(
                    wrapped_func, 
                    x0, 
                    method='Powell',
                    options={
                        'maxiter': 40,
                        'ftol': 1e-14,
                        'xtol': 1e-14
                    }
                )
            except (TimeoutError, RuntimeError, Exception):
                pass

        # Phase 3: High-precision polish with Nelder-Mead
        if self.call_count < self.budget - 20 and time.time() < self.deadline - 15:
            try:
                res2 = scipy.optimize.minimize(
                    wrapped_func,
                    self.best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 200,
                        'xatol': 1e-14,
                        'fatol': 1e-14,
                        'adaptive': True
                    }
                )
            except (TimeoutError, RuntimeError, Exception):
                pass

        # Phase 4: Final coordinate descent polish for tight convergence
        if self.call_count < self.budget - 5 and time.time() < self.deadline - 5:
            # Simple coordinate descent with adaptive step
            x = self.best_x.copy()
            f = self.best_f
            step = 0.1
            for _ in range(20):
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
                    if step < 1e-10:
                        break
                else:
                    # Keep step size if improvement found, or slightly increase
                    step = min(step * 1.1, 0.5)

            if f < self.best_f:
                self.best_f = f
                self.best_x = x

        return self.best_x, self.best_f
