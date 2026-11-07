"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-restart (1+1)-ES with Nelder-Mead polish for BBOB f16 Weierstrass."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # safety margin before 330s kill
        rng = np.random.default_rng(42)

        call_count = 0
        dim = self.dim
        budget = self.budget

        def count_func(x):
            nonlocal call_count
            if call_count >= budget:
                return np.inf
            call_count += 1
            x = np.asarray(x, dtype=np.float64)
            if x.shape != (dim,):
                x = np.resize(x, dim)
            x = np.clip(x, -5.0, 5.0)
            val = float(func(x))
            if not np.isfinite(val):
                return 1e10
            return val

        def nelder_mead(x0, max_iter=50):
            """Local polish using Nelder-Mead, respecting budget and deadline."""
            try:
                res = minimize(
                    count_func,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': max_iter,
                        'xatol': 1e-5,
                        'fatol': 1e-7,
                        'adaptive': True,
                    }
                )
                if np.isfinite(res.fun) and res.fun < 1e10:
                    return np.clip(res.x, -5.0, 5.0), float(res.fun)
            except Exception:
                pass
            return x0, np.inf

        # Phase 1: Multi-restart (1+1)-ES
        # We use a fraction of the budget for global exploration via restarts,
        # then spend remaining on local polish of the best point found.
        
        # Strategy: 
        # - Run (1+1)-ES with restarts. 
        # - Each restart allocates a portion of the remaining budget.
        # - We track the best point found so far.
        
        best_x = None
        best_fx = np.inf

        # Initial random start
        x0 = rng.uniform(-4.0, 4.0, dim)
        fx0 = count_func(x0)
        best_x = x0.copy()
        best_fx = fx0

        # Adaptive step size (sigma)
        sigma = 1.5
        no_improve = 0
        max_no_improve = 25  # Restart if stuck for 25 steps

        # Main loop: (1+1)-ES with restarts
        # We stop when budget is exhausted or time is up.
        # We reserve some calls for final polish if possible.
        # Let's say we use 70% of budget for exploration, 30% for polish.
        # But polish is expensive in calls (Nelder-Mead uses ~10*dim calls per iter).
        # In 5D, one NM iter ~ 10 calls. 50 iters = 500 calls. Too many.
        # So we need to be careful.
        # Let's run ES until we have ~100 calls left, then do a few NM iters.
        
        # Actually, the best strategy for Weierstrass is often:
        # 1. Find a good basin with ES.
        # 2. Polish it with NM.
        
        # Let's implement a loop that restarts ES from random points near the current best
        # or completely random points if stuck, until budget is low.
        
        while call_count < budget - 20 and time.time() < deadline:
            # (1+1)-ES step
            y = np.clip(x0 + sigma * rng.standard_normal(dim), -5.0, 5.0)
            fy = count_func(y)

            if fy < fx0:
                x0, fx0 = y, fy
                if fx0 < best_fx:
                    best_x = x0.copy()
                    best_fx = fx0
                    no_improve = 0
                sigma *= 1.2  # Increase step on success
            else:
                sigma *= 0.85  # Decrease step on failure
                no_improve += 1

            # Restart logic
            if no_improve > max_no_improve:
                # Restart from a point near the best found so far
                # Or if best is not very good, try a random point
                if rng.random() < 0.5:
                    # Local restart
                    x0 = np.clip(best_x + rng.normal(0, sigma * 0.5, dim), -5.0, 5.0)
                else:
                    # Global restart
                    x0 = rng.uniform(-4.0, 4.0, dim)
                
                fx0 = count_func(x0)
                no_improve = 0
                sigma = 1.5  # Reset sigma

        # Phase 2: Local Polish
        # Use remaining budget for Nelder-Mead on the best point found.
        if call_count < budget and time.time() < deadline:
            # Determine how many NM iterations we can afford
            # NM in D dims uses roughly D+1 calls per iteration (sometimes more with adaptive)
            # Let's assume ~2*D calls per iteration to be safe.
            remaining_calls = budget - call_count
            max_iter = max(5, int(remaining_calls / (2 * dim)))
            
            x_polished, fx_polished = nelder_mead(best_x, max_iter=max_iter)
            
            if fx_polished < best_fx:
                best_x = x_polished
                best_fx = fx_polished

        return best_x, best_fx
# EVOLVE-BLOCK-END
