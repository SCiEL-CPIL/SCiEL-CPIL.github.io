"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-phase optimizer: random exploration + SLSQP refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Track best solution found
        best_x = None
        best_f = np.inf
        call_count = 0

        def check_deadline():
            return time.time() > deadline

        # Phase 1: Random exploration - evaluate 20 diverse points
        n_random = 20
        for i in range(n_random):
            if check_deadline() or call_count >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: SLSQP refinement from top 5 diverse seeds
        # We'll try SLSQP from the best point and a few others
        # Collect candidate starting points
        candidates = [best_x.copy()]
        
        # Add a few more diverse starting points near the best
        for _ in range(4):
            if check_deadline() or call_count >= budget:
                break
            # Perturb the best point
            perturbation = rng.normal(0, 1.0, dim)
            x_cand = np.clip(best_x + perturbation, -5, 5)
            fx_cand = func(x_cand)
            call_count += 1
            if fx_cand < best_f:
                best_f = fx_cand
                best_x = x_cand.copy()
            candidates.append(x_cand.copy())

        # Phase 3: Multi-scale SLSQP refinement
        # Start with the best point and do SLSQP with decreasing scales
        if best_x is not None:
            # First SLSQP run with moderate tolerance
            try:
                res = minimize(
                    func,
                    best_x,
                    method='SLSQP',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 100, 'ftol': 1e-12, 'maxfun': 80}
                )
                if res.success or res.fun < best_f:
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
            except Exception:
                pass

            if check_deadline() or call_count >= budget:
                return best_x, best_f

            # Second SLSQP run with tighter tolerance
            try:
                res = minimize(
                    func,
                    best_x,
                    method='SLSQP',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 150, 'ftol': 1e-15, 'maxfun': 100}
                )
                if res.success or res.fun < best_f:
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
            except Exception:
                pass

            if check_deadline() or call_count >= budget:
                return best_x, best_f

            # Third SLSQP run - final polish
            try:
                res = minimize(
                    func,
                    best_x,
                    method='SLSQP',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 200, 'ftol': 1e-16, 'maxfun': 120}
                )
                if res.success or res.fun < best_f:
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
            except Exception:
                pass

        # Phase 4: If we have budget left, do a few more random restarts + SLSQP
        while not check_deadline() and call_count < budget - 50:
            x_new = rng.uniform(-5, 5, dim)
            fx_new = func(x_new)
            call_count += 1
            if fx_new < best_f:
                best_f = fx_new
                best_x = x_new.copy()
                # Quick SLSQP from this new best
                try:
                    res = minimize(
                        func,
                        best_x,
                        method='SLSQP',
                        bounds=[(-5, 5)] * dim,
                        options={'maxiter': 50, 'ftol': 1e-14, 'maxfun': 40}
                    )
                    if res.success or res.fun < best_f:
                        if res.fun < best_f:
                            best_f = res.fun
                            best_x = res.x.copy()
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
