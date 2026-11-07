"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


class Optimizer:
    """
    Optimizer for Rosenbrock (f8) on [-5,5]^5.
    Strategy:
      1. Dense random sampling to find good basins.
      2. L-BFGS-B local refinement from the best samples.
      3. Nelder-Mead refinement for robustness.
      4. Final L-BFGS-B polish.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        bounds = [(-5.0, 5.0)] * dim
        deadline = time.time() + 280.0

        call_count = [0]
        best_f = np.inf
        best_x = None

        def counting_func(x):
            nonlocal best_f, best_x
            if call_count[0] >= budget:
                return best_f
            call_count[0] += 1
            val = func(x)
            if val < best_f:
                best_f = val
                best_x = np.array(x, copy=True)
            return val

        # Phase 1: Dense random sampling
        # Use Sobol-like or just dense uniform for initial exploration
        rng = np.random.default_rng(42)
        n_samples = min(50, budget // 2)  # Use at most 50 for sampling
        samples = rng.uniform(-4.5, 4.5, size=(n_samples, dim))
        
        # Add some structured points: center, axis-aligned
        structured = np.array([
            np.zeros(dim),
            np.ones(dim) * 0.5,
            np.ones(dim) * -0.5,
            np.ones(dim) * 2.0,
            np.ones(dim) * -2.0,
        ])
        samples = np.vstack([samples, structured])

        for i in range(len(samples)):
            if call_count[0] >= budget or time.time() > deadline:
                break
            x = np.clip(samples[i], -5.0, 5.0)
            val = counting_func(x)
            if val < best_f:
                best_f = val
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = counting_func(best_x)

        # Phase 2: L-BFGS-B from best sample
        if call_count[0] < budget and time.time() < deadline:
            try:
                res = minimize(
                    counting_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'maxfun': 100, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Phase 3: Nelder-Mead from current best (derivative-free, robust)
        if call_count[0] < budget and time.time() < deadline:
            try:
                res = minimize(
                    counting_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'maxfev': 100, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Phase 4: Final L-BFGS-B polish
        if call_count[0] < budget and time.time() < deadline:
            try:
                res = minimize(
                    counting_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'maxfun': 100, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Phase 5: If still have budget, try a few more random restarts
        if call_count[0] < budget - 20 and time.time() < deadline:
            for _ in range(5):
                if call_count[0] >= budget - 10 or time.time() > deadline:
                    break
                x0 = rng.uniform(-4.0, 4.0, dim)
                try:
                    res = minimize(
                        counting_func,
                        x0,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 50, 'maxfun': 50, 'ftol': 1e-10, 'gtol': 1e-6}
                    )
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
                except Exception:
                    pass

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = counting_func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
