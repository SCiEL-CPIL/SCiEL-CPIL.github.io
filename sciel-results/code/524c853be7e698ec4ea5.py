"""Multi-start Nelder-Mead with adaptive local exploitation for Katsuura f23.
Uses diverse initial points and structured local searches to navigate the
multimodal landscape with hidden shift/rotation/offset."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        calls = 0
        start_time = time.time()
        deadline = start_time + 290  # 40s margin before 330s kill
        best_x = None
        best_f = np.inf
        all_points = []  # track all evaluated points for diversity

        def safe_func(x):
            nonlocal calls
            if time.time() > deadline:
                raise TimeoutError
            if calls >= budget:
                raise StopIteration
            calls += 1
            x = np.asarray(x, dtype=float).flatten()
            if len(x) != dim:
                x = x[:dim]
            x = np.clip(x, -5.0, 5.0)
            val = float(func(x))
            all_points.append(x.copy())
            return val

        # Phase 1: Global exploration with diverse starting points
        # Generate a mix of structured and random starting points
        rng = np.random.default_rng(42)
        
        start_points = []
        # Center
        start_points.append(np.zeros(dim))
        # Axis-aligned points
        for d in range(dim):
            for v in [2.0, -2.0, 4.0, -4.0]:
                x = np.zeros(dim)
                x[d] = v
                start_points.append(x)
        # Diagonal points
        for v in [2.0, -2.0, 3.0, -3.0, 4.0, -4.0]:
            start_points.append(np.full(dim, v))
        # Random points
        for _ in range(15):
            start_points.append(rng.uniform(-4.5, 4.5, size=dim))
        
        # Limit to a reasonable number of starts
        max_starts = min(25, len(start_points))
        start_points = start_points[:max_starts]
        
        # Evaluate each starting point with a small Nelder-Mead
        for x0 in start_points:
            if calls >= budget or time.time() > deadline:
                break
            try:
                res = opt.minimize(
                    safe_func,
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': 8, 'xatol': 1e-3, 'fatol': 1e-3}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except (TimeoutError, StopIteration):
                break
            except Exception:
                pass

        # Phase 2: Local exploitation from the best point with more iterations
        if best_x is not None and calls < budget and time.time() < deadline:
            try:
                remaining_budget = budget - calls
                est_maxiter = max(1, remaining_budget // (dim + 1))
                res = opt.minimize(
                    safe_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': est_maxiter, 'xatol': 1e-8, 'fatol': 1e-8}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except (TimeoutError, StopIteration):
                pass
            except Exception:
                pass

        # Phase 3: If we have budget left, try a few more diverse restarts
        if calls < budget and time.time() < deadline:
            # Generate new random points far from current best
            for _ in range(5):
                if calls >= budget or time.time() > deadline:
                    break
                new_point = rng.uniform(-5.0, 5.0, size=dim)
                try:
                    res = opt.minimize(
                        safe_func,
                        new_point,
                        method='Nelder-Mead',
                        options={'maxiter': 10, 'xatol': 1e-4, 'fatol': 1e-4}
                    )
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
                except (TimeoutError, StopIteration):
                    break
                except Exception:
                    pass

        # Ensure we have a valid result
        if best_x is None:
            best_x = np.zeros(dim)
            try:
                best_f = safe_func(best_x)
            except:
                best_f = 0.0

        return best_x, best_f
# EVOLVE-BLOCK-END
