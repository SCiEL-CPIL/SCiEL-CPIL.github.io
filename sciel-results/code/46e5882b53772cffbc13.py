"""BBOB f19 (Griewank-Rosenbrock) optimizer.

Strategy:
- Structured initialization exploiting the Griewank-Rosenbrock structure.
- Powell's method for robust local optimization (handles correlated dimensions well).
- Nelder-Mead for additional refinement.
- Adaptive coordinate descent for final convergence.
- L-BFGS-B for high-precision polish.
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
        deadline = t0 + 280.0  # Safety margin before 300s limit
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

        rng = np.random.default_rng(42)

        # Structured initialization: exploit the Griewank-Rosenbrock structure
        # Rosenbrock part wants x=1, Griewank part wants x=0
        # The minimum is a compromise, likely near x=0.5-0.8 in each dimension

        init_points = []

        # Core structured points - broader coverage
        for val in [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            init_points.append(np.ones(dim) * val)

        # Add diverse points around promising centers with varying spread
        for center in [0.4, 0.5, 0.6, 0.7, 0.8]:
            for scale in [0.15, 0.3, 0.5]:
                offset = rng.normal(0, scale, dim)
                init_points.append(center * np.ones(dim) + offset)

        # Some fully random points for coverage across the box
        for _ in range(8):
            init_points.append(rng.uniform(-4, 4, dim))

        # A few points at corners/edges of the likely minimum region
        for _ in range(4):
            p = rng.uniform(-1, 2, dim)
            init_points.append(p)

        # Additional structured points to improve coverage
        for val in [0.1, 0.3, 0.55, 0.65, 0.75, 0.85, 0.95]:
            init_points.append(np.ones(dim) * val)
        
        # Points with mixed values (not all same)
        for _ in range(6):
            p = rng.uniform(0.2, 1.0, dim)
            init_points.append(p)

        # Evaluate initial points
        for p in init_points:
            if call_count >= budget or time.time() >= deadline:
                break
            eval_point(p)

        if call_count >= budget or time.time() >= deadline:
            return best_x

        # Powell's method local search from multiple starting points
        def powell_search(x0, max_iter=50):
            nonlocal best_x, best_f, call_count
            if call_count >= budget or time.time() >= deadline:
                return

            try:
                def objective(x):
                    if call_count >= budget or time.time() >= deadline:
                        return best_f
                    f = float(func(x))
                    call_count += 1
                    if f < best_f:
                        best_f = f
                        best_x = np.clip(x, -5, 5).copy()
                    return f

                res = minimize(
                    objective,
                    np.clip(x0, -5, 5),
                    method='Powell',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': max_iter, 'ftol': 1e-10, 'xtol': 1e-8}
                )
            except Exception:
                pass

        # Run Powell from best point first with more iterations
        powell_search(best_x.copy(), max_iter=120)

        # Run Powell from other promising starting points
        candidate_starts = [best_x.copy()]
        for center in [0.3, 0.5, 0.7, 1.0]:
            candidate_starts.append(center * np.ones(dim))

        for start in candidate_starts[1:]:
            if call_count >= budget or time.time() >= deadline:
                break
            if np.linalg.norm(start - best_x) > 0.3:  # Only if far enough
                powell_search(start, max_iter=60)

        # Nelder-Mead for additional local refinement
        def nm_search(x0, max_iter=100):
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
            step_sizes = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001]

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
                            if f_trial < f - 1e-10:
                                x = x_trial.copy()
                                f = f_trial
                                if f < best_f:
                                    best_f = f
                                    best_x = x.copy()
                                improved = True
                            elif f_trial < best_f:
                                best_f = f_trial
                                best_x = np.clip(x_trial, -5, 5).copy()

        # Final high-precision polish with L-BFGS-B
        if call_count < budget and time.time() < deadline:
            try:
                def polish_func(x):
                    if call_count >= budget or time.time() >= deadline:
                        return best_f
                    f = float(func(x))
                    call_count += 1
                    if f < best_f:
                        best_f = f
                        best_x = np.clip(x, -5, 5).copy()
                    return f

                res = minimize(
                    polish_func,
                    best_x.copy(),
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 200}
                )
            except Exception:
                pass

        # Second round of Powell if budget remains - can find better local minima
        if call_count < budget and time.time() < deadline:
            powell_search(best_x.copy(), max_iter=100)

        # Final Nelder-Mead pass for fine-tuning
        if call_count < budget and time.time() < deadline:
            nm_search(best_x.copy(), max_iter=150)

        # Last few random perturbations if budget remains
        if call_count < budget and time.time() < deadline:
            for scale in [0.05, 0.02, 0.01, 0.005]:
                if call_count >= budget or time.time() >= deadline:
                    break
                x_try = np.clip(best_x + scale * rng.standard_normal(dim), -5, 5)
                eval_point(x_try)

        return best_x
