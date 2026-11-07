import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Optimizer for BBOB f13 (Sharp Ridge) in 5D with budget 500.
    Strategy:
    1. Random search to locate the ridge basin.
    2. L-BFGS-B local refinement (handles anisotropy well).
    3. Multiple restarts from perturbed best points if budget remains.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290.0
        
        best_x = None
        best_f = float('inf')
        calls = 0
        rng = np.random.default_rng(1234)
        
        def safe_call(x):
            nonlocal calls, best_f, best_x
            if calls >= budget:
                return None
            if time.time() > deadline:
                return None
            calls += 1
            try:
                x = np.asarray(x, dtype=np.float64)
                x = np.clip(x, -5.0, 5.0)
                f_val = float(func(x))
                if np.isfinite(f_val) and f_val < best_f:
                    best_f = f_val
                    best_x = x.copy()
                return f_val
            except Exception:
                return None

        # Phase 1: Random search to find a good starting point
        n_random = 20
        for _ in range(n_random):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5.0, 5.0, size=dim)
            f_val = safe_call(x)
            if f_val is None:
                break

        if best_x is None:
            x0 = np.zeros(dim)
            f0 = safe_call(x0)
            if f0 is None:
                return best_x, best_f
            best_x = x0
            best_f = f0

        # Phase 2: L-BFGS-B local refinement
        def wrapped_func(x):
            if calls >= budget or time.time() > deadline:
                return best_f  # Return current best to stop optimization
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            f_val = safe_call(x)
            if f_val is None:
                return best_f
            return f_val

        bounds = [(-5.0, 5.0)] * dim
        
        try:
            result = minimize(
                wrapped_func,
                best_x.copy(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'maxfun': 200, 'gtol': 1e-6}
            )
            if result.success or result.fun < best_f:
                x_opt = np.clip(result.x, -5.0, 5.0)
                f_opt = safe_call(x_opt)
                if f_opt is not None and f_opt < best_f:
                    best_f = f_opt
                    best_x = x_opt.copy()
        except Exception:
            pass

        # Phase 3: Multiple restarts with adaptive perturbation
        if calls < budget - 30 and time.time() < deadline:
            for restart in range(8):
                if calls >= budget - 20 or time.time() > deadline:
                    break

                # Adaptive perturbation: larger early, smaller late
                noise_scale = 2.0 / (restart + 1)
                perturbation = rng.normal(0, noise_scale, size=dim)
                x_restart = np.clip(best_x + perturbation, -5.0, 5.0)

                f_restart = safe_call(x_restart)
                if f_restart is None:
                    break

                if f_restart < best_f:
                    best_f = f_restart
                    best_x = x_restart.copy()

                # L-BFGS-B from this point
                try:
                    result = minimize(
                        wrapped_func,
                        x_restart.copy(),
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 50, 'maxfun': 100, 'gtol': 1e-5}
                    )
                    if result.success or result.fun < best_f:
                        x_opt = np.clip(result.x, -5.0, 5.0)
                        f_opt = safe_call(x_opt)
                        if f_opt is not None and f_opt < best_f:
                            best_f = f_opt
                            best_x = x_opt.copy()
                except Exception:
                    pass

        return best_x, best_f
