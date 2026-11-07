"""BBOB f19 (Griewank-Rosenbrock) optimizer with Powell-based local search.

Strategy:
- Broad structured initialization to locate the Rosenbrock valley
- Powell's method for derivative-free local optimization
- Multi-restart with adaptive step sizes
- Nelder-Mead polish for final convergence
- Coordinate descent for ultra-fine tuning
"""
import numpy as np
from scipy.optimize import minimize
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 275.0
        dim = self.dim
        budget = self.budget
        
        best_x = None
        best_f = np.inf
        call_count = 0
        
        def eval_point(x):
            nonlocal best_x, best_f, call_count
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            f = float(func(x))
            call_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f
        
        rng = np.random.default_rng(12345)
        
        # Broad structured initialization: the hidden shift/rotation means
        # the Rosenbrock valley could be anywhere in [-4,4]^5. We need wide
        # coverage to locate it, then exploit local structure.
        
        init_points = []
        
        # Grid of structured points covering the plausible region
        # Rosenbrock valley in f19: consecutive coords satisfy x[i+1] ≈ x[i]^2
        # With Griewank modulation, the minimum is a compromise
        # Try multiple "valley seeds" at different base levels
        
        base_levels = [0.0, 0.3, 0.5, 0.7, 1.0, -0.5, -1.0]
        for base in base_levels:
            # Generate a Rosenbrock-like trajectory from this base
            x = np.zeros(dim)
            x[0] = base
            for i in range(1, dim):
                x[i] = x[i-1] ** 2
            x = np.clip(x, -5, 5)
            init_points.append(x)
            # Perturbed versions
            for _ in range(2):
                x_pert = x + rng.normal(0, 0.5, dim)
                init_points.append(np.clip(x_pert, -5, 5))
        
        # Uniform grid points for broader coverage
        grid_vals = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
        for v in grid_vals:
            init_points.append(np.ones(dim) * v)
        
        # Random points for global coverage
        for _ in range(15):
            init_points.append(rng.uniform(-4, 4, dim))
        
        # Evaluate initial points
        for p in init_points:
            if call_count >= budget or time.time() >= deadline:
                break
            eval_point(p)
        
        if call_count >= budget or time.time() >= deadline:
            return best_x
        
        # Powell's method local search from multiple starting points
        def powell_search(x0, max_iter=100):
            nonlocal best_x, best_f, call_count
            if call_count >= budget or time.time() >= deadline:
                return
            
            try:
                def objective(x):
                    if call_count >= budget or time.time() >= deadline:
                        return best_f
                    x = np.clip(x, -5, 5)
                    f = float(func(x))
                    call_count += 1
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                    return f
                
                res = minimize(
                    objective,
                    np.clip(x0, -5, 5),
                    method='Powell',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': max_iter, 'ftol': 1e-12, 'xtol': 1e-10}
                )
            except Exception:
                pass
        
        # Run Powell from best point first (most important)
        powell_search(best_x.copy(), max_iter=120)
        
        # Run Powell from other promising starting points
        candidate_starts = [best_x.copy()]
        for base in [0.0, 0.5, 1.0, -0.5]:
            x = np.zeros(dim)
            x[0] = base
            for i in range(1, dim):
                x[i] = x[i-1] ** 2
            candidate_starts.append(np.clip(x, -5, 5))
        
        for start in candidate_starts[1:]:
            if call_count >= budget or time.time() >= deadline:
                break
            if np.linalg.norm(start - best_x) > 0.3:
                powell_search(start, max_iter=60)
        
        # Nelder-Mead for additional local refinement
        def nm_search(x0, max_iter=200):
            nonlocal best_x, best_f, call_count
            if call_count >= budget or time.time() >= deadline:
                return
            
            try:
                def objective(x):
                    if call_count >= budget or time.time() >= deadline:
                        return best_f
                    x = np.clip(x, -5, 5)
                    f = float(func(x))
                    call_count += 1
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                    return f
                
                res = minimize(
                    objective,
                    np.clip(x0, -5, 5),
                    method='Nelder-Mead',
                    options={'maxiter': max_iter, 'xatol': 1e-6, 'fatol': 1e-10, 'adaptive': True}
                )
            except Exception:
                pass
        
        if call_count < budget and time.time() < deadline:
            nm_search(best_x.copy(), max_iter=200)
        
        # Adaptive coordinate descent for final convergence
        if call_count < budget and time.time() < deadline:
            x = best_x.copy()
            f = best_f
            step_sizes = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0001]
            
            for step in step_sizes:
                improved = True
                while improved and call_count < budget and time.time() < deadline:
                    improved = False
                    for c in range(dim):
                        if call_count >= budget or time.time() >= deadline:
                            break
                        for direction in [1, -1]:
                            x_trial = x.copy()
                            x_trial[c] += direction * step
                            x_trial = np.clip(x_trial, -5, 5)
                            f_trial = float(func(x_trial))
                            call_count += 1
                            if f_trial < f - 1e-12:
                                x = x_trial.copy()
                                f = f_trial
                                if f < best_f:
                                    best_f = f
                                    best_x = x.copy()
                                improved = True
                            elif f_trial < best_f:
                                best_f = f_trial
                                best_x = x_trial.copy()
        
        # Final Nelder-Mead polish with very tight tolerances
        if call_count < budget and time.time() < deadline:
            try:
                def polish_func(x):
                    if call_count >= budget or time.time() >= deadline:
                        return best_f
                    x = np.clip(x, -5, 5)
                    f = float(func(x))
                    call_count += 1
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                    return f
                
                res = minimize(
                    polish_func,
                    best_x.copy(),
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-14, 'adaptive': True}
                )
            except Exception:
                pass
        
        # Ultra-fine coordinate descent
        if call_count < budget and time.time() < deadline:
            x = best_x.copy()
            f = best_f
            for step in [0.0005, 0.0002, 0.0001, 0.00005, 0.00001]:
                if call_count >= budget or time.time() >= deadline:
                    break
                improved = True
                while improved and call_count < budget and time.time() < deadline:
                    improved = False
                    for c in range(dim):
                        if call_count >= budget or time.time() >= deadline:
                            break
                        for direction in [1, -1]:
                            x_trial = x.copy()
                            x_trial[c] += direction * step
                            x_trial = np.clip(x_trial, -5, 5)
                            f_trial = float(func(x_trial))
                            call_count += 1
                            if f_trial < f - 1e-14:
                                x = x_trial.copy()
                                f = f_trial
                                if f < best_f:
                                    best_f = f
                                    best_x = x.copy()
                                improved = True
        
        # Last few random perturbations if budget remains
        if call_count < budget and time.time() < deadline:
            for scale in [0.05, 0.02, 0.01, 0.005]:
                if call_count >= budget or time.time() >= deadline:
                    break
                x_try = np.clip(best_x + scale * rng.standard_normal(dim), -5, 5)
                eval_point(x_try)
        
        return best_x
