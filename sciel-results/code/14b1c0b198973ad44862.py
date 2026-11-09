import numpy as np
import time
import scipy.optimize


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_used = 0

        def track_call(x, f):
            nonlocal calls_used, best_x, best_f
            calls_used += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def time_ok():
            return time.time() < deadline and calls_used < budget

        rng = np.random.default_rng(42)

        # Phase 1: Sobol quasi-random sampling for diverse global exploration
        from scipy.stats import qmc
        n_sobol = min(400, budget - 10)
        sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
        points = sampler.random(n_sobol)
        # Map from [0,1]^dim to [-5,5]^dim
        points = points * 10.0 - 5.0

        sobol_results = []
        for x in points:
            if not time_ok():
                break
            fx = func(x)
            track_call(x, fx)
            sobol_results.append((fx, x.copy()))

        if not sobol_results:
            return np.zeros(dim), np.inf

        # Sort by fitness and take top 5
        sobol_results.sort(key=lambda t: t[0])
        top_candidates = [x for _, x in sobol_results[:5]]

        # Phase 2: Multi-start Nelder-Mead with adaptive simplex sizes
        remaining = budget - calls_used
        if remaining > 0 and top_candidates:
            # Compute pairwise distances to scale simplex sizes
            n_starts = min(5, len(top_candidates))
            for i in range(n_starts):
                if not time_ok():
                    break
                
                x0 = top_candidates[i]
                
                # Adaptive simplex size: scale by distance to nearest other candidate
                if i < len(top_candidates) - 1:
                    dists = [np.linalg.norm(x0 - top_candidates[j]) for j in range(len(top_candidates)) if j != i]
                    if dists:
                        scale = max(0.1, min(1.0, np.min(dists) / 10.0))
                    else:
                        scale = 0.5
                else:
                    scale = 0.5
                
                # Create initial simplex
                simplex = [x0.copy()]
                for j in range(dim):
                    perturbation = x0.copy()
                    perturbation[j] += scale * 0.5
                    simplex.append(perturbation)
                
                nm_calls = [0]
                max_nm_calls = min(remaining - 5, 80)
                
                def nm_func(x):
                    nm_calls[0] += 1
                    if nm_calls[0] > max_nm_calls:
                        return best_f + 1.0
                    x_clipped = np.clip(x, -5, 5)
                    fx = func(x_clipped)
                    track_call(x_clipped, fx)
                    return fx
                
                try:
                    result = scipy.optimize.minimize(
                        nm_func,
                        x0,
                        method='Nelder-Mead',
                        options={
                            'maxiter': max_nm_calls,
                            'xatol': 1e-10,
                            'fatol': 1e-10,
                            'adaptive': True
                        }
                    )
                    
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = np.clip(result.x, -5, 5).copy()
                except Exception:
                    pass
                
                remaining = budget - calls_used
                if remaining <= 10:
                    break

        # Phase 3: Final fine-tuning with very small random steps
        remaining = budget - calls_used
        if remaining > 0 and best_x is not None:
            polish_sigma = 0.001
            for _ in range(min(remaining, 20)):
                if not time_ok():
                    break
                z = rng.standard_normal(dim)
                y = best_x + polish_sigma * z
                y = np.clip(y, -5, 5)
                fy = func(y)
                track_call(y, fy)
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                else:
                    polish_sigma *= 0.5

        return best_x, best_f
