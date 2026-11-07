"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


class Optimizer:
    """Multi-restart L-BFGS-B with Nelder-Mead polish for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin within 300s limit
        
        call_count = 0
        max_calls = self.budget
        
        best_x = None
        best_f = np.inf

        def wrapped_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= max_calls:
                return 1e10
            call_count += 1
            x = np.clip(x, -5, 5)
            val = func(x)
            if not np.isfinite(val):
                val = 1e10
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        rng = np.random.default_rng(42)
        
        # Generate diverse initial points
        # Start with origin, then random points in the box
        initial_points = [np.zeros(self.dim)]
        for _ in range(9):
            initial_points.append(rng.uniform(-4.0, 4.0, self.dim))

        # Phase 1: L-BFGS-B from multiple starts
        # Allocate budget carefully. 
        # We have 500 calls. 
        # L-BFGS-B typically converges in ~50-100 calls for smooth 5D functions.
        # 10 restarts * 50 calls = 500 calls. 
        # So we limit maxfun per restart to ensure we can try multiple restarts.
        
        for i, x0 in enumerate(initial_points):
            if time.time() > deadline - 10 or call_count >= max_calls:
                break
            
            x0 = np.clip(x0, -5, 5)
            
            # Determine how many function evaluations to allow for this restart
            # Early restarts get more budget if we haven't found a good solution yet?
            # Or just fixed budget per restart.
            # Let's give each restart a fixed budget, but stop if we hit the global limit.
            # If we use 50 calls per restart, 10 restarts = 500 calls.
            # This is tight. Let's try 40 calls per restart to leave room for polish.
            maxfun_per_restart = 40
            if i == 0:
                # Give the first restart (origin) a bit more chance
                maxfun_per_restart = 60
            
            try:
                result = minimize(
                    wrapped_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={
                        'maxiter': 100,
                        'maxfun': maxfun_per_restart,
                        'ftol': 1e-8,
                        'gtol': 1e-6
                    }
                )
                
                # Update best if this restart found something better
                # Note: wrapped_func already updates best_x and best_f internally
            except Exception:
                continue

        # Phase 2: Polish with Nelder-Mead if budget remains
        if best_x is not None and call_count < max_calls and time.time() < deadline - 5:
            remaining = max_calls - call_count
            if remaining > 20:
                try:
                    # Nelder-Mead can be used to fine-tune the best point found
                    # It doesn't use gradients, so it can handle the final narrow valley
                    # if L-BFGS-B got stuck on the boundary or precision limits.
                    result_nm = minimize(
                        wrapped_func,
                        best_x,
                        method='Nelder-Mead',
                        options={
                            'maxiter': 50,
                            'xatol': 1e-8,
                            'fatol': 1e-10,
                            'adaptive': True
                        }
                    )
                except Exception:
                    pass

        # Final fallback
        if best_x is None:
            best_x = np.zeros(self.dim)
            if call_count < max_calls:
                best_f = wrapped_func(best_x)
            else:
                best_f = 1e10

        return best_x, best_f
# EVOLVE-BLOCK-END
