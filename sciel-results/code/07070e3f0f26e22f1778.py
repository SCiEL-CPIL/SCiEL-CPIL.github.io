# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize

class Optimizer:
    """Hybrid optimizer for Bent Cigar: uses Powell to align search directions
    with the anisotropic valley, then L-BFGS-B for fast local convergence.
    Multiple restarts with adaptive budget allocation."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 280.0
        dim = self.dim
        budget = self.budget
        calls = 0
        best_f = float('inf')
        best_x = np.zeros(dim)

        def wrapper(x):
            nonlocal calls, best_f, best_x
            if calls >= budget:
                return float('inf')
            if time.time() > deadline:
                return float('inf')
            calls += 1
            try:
                x_clipped = np.clip(x, -5, 5)
                val = float(func(x_clipped))
                if val < best_f:
                    best_f = val
                    best_x = x_clipped.copy()
                return val
            except Exception:
                return float('inf')

        rng = np.random.default_rng(42)

        # Phase 1: Evaluate origin to establish baseline
        x_origin = np.zeros(dim)
        f_origin = wrapper(x_origin)
        if f_origin < best_f:
            best_f = f_origin
            best_x = x_origin.copy()

        # Phase 2: Powell + L-BFGS-B from origin
        if calls < budget - 20 and time.time() < deadline:
            try:
                # Powell to align with valley structure
                res_powell = scipy.optimize.minimize(
                    wrapper, x_origin, method='Powell',
                    options={'maxiter': 50, 'maxfev': min(40, budget - calls - 10)}
                )
                x_mid = res_powell.x if res_powell is not None else x_origin
            except Exception:
                x_mid = x_origin

            # L-BFGS-B to polish
            if calls < budget - 5 and time.time() < deadline:
                try:
                    res_lbfgs = scipy.optimize.minimize(
                        wrapper, x_mid, method='L-BFGS-B',
                        bounds=[(-5, 5)] * dim,
                        options={'maxiter': 100, 'maxfev': min(60, budget - calls - 5)}
                    )
                    if res_lbfgs.fun < best_f:
                        best_f = res_lbfgs.fun
                        best_x = res_lbfgs.x.copy()
                except Exception:
                    pass

        # Phase 3: Multiple random restarts with Powell + L-BFGS-B
        n_restarts = 6
        for i in range(n_restarts):
            if calls >= budget - 15:
                break
            if time.time() > deadline:
                break

            x_start = rng.uniform(-5, 5, dim)

            # Powell phase: align search directions
            remaining = budget - calls - 10
            if remaining < 15:
                break

            powell_budget = min(30, remaining)
            if powell_budget < 10:
                break

            try:
                res_powell = scipy.optimize.minimize(
                    wrapper, x_start, method='Powell',
                    options={'maxiter': 40, 'maxfev': powell_budget}
                )
                x_mid = res_powell.x if res_powell is not None else x_start
            except Exception:
                x_mid = x_start

            # L-BFGS-B phase: fast local convergence
            remaining = budget - calls - 5
            if remaining < 10:
                break

            lbfgs_budget = min(50, remaining)
            if lbfgs_budget < 8:
                break

            try:
                res_lbfgs = scipy.optimize.minimize(
                    wrapper, x_mid, method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 80, 'maxfev': lbfgs_budget}
                )
                if res_lbfgs.fun < best_f:
                    best_f = res_lbfgs.fun
                    best_x = res_lbfgs.x.copy()
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
