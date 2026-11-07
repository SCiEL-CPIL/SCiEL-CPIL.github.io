"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, fmin
import time


class Optimizer:
    """Multi-restart L-BFGS-B + Nelder-Mead polish optimizer for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # safety margin within 300s limit
        
        # Count function calls
        call_count = 0
        max_calls = self.budget
        
        # Best solution found
        best_x = None
        best_f = np.inf
        
        # Wrap func to count calls, enforce budget, and track best
        def wrapped_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= max_calls:
                return 1e10  # Return large value to stop optimization
            call_count += 1
            # Clip to box
            x = np.clip(x, -5, 5)
            val = func(x)
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val
        
        # Phase 1: Multi-restart L-BFGS-B
        # Discus is unimodal but hidden shift/rotation means center may not be optimal.
        # Use multiple diverse starting points to ensure we find the basin.
        
        # Generate diverse starting points: center + random perturbations
        rng = np.random.default_rng(42)
        start_points = [np.zeros(self.dim)]
        
        # Add a few random starts to cover the space
        n_restarts = 5
        for i in range(n_restarts):
            # Spread starts across the box
            x_start = rng.uniform(-4, 4, self.dim)
            start_points.append(x_start)
        
        # Allocate budget: give more to center (likely best), less to others
        # Center gets ~40%, each restart gets ~12%
        center_budget = int(max_calls * 0.40)
        restart_budget = int(max_calls * 0.12)
        
        # Run L-BFGS-B from each start point
        for idx, x0 in enumerate(start_points):
            if time.time() > deadline - 10:
                break
            if call_count >= max_calls - 50:  # Reserve some for Nelder-Mead
                break
                
            # Determine maxfun for this restart
            if idx == 0:
                mf = center_budget
            else:
                mf = restart_budget
            
            # Don't exceed remaining budget
            remaining = max_calls - call_count - 50
            if remaining <= 0:
                break
            mf = min(mf, remaining)
            
            try:
                result = minimize(
                    wrapped_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={
                        'maxiter': 1000, 
                        'maxfun': mf,
                        'ftol': 1e-15,
                        'gtol': 1e-12
                    }
                )
            except Exception:
                pass
        
        # Fallback if no valid result
        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = wrapped_func(best_x)

        # Phase 2: Nelder-Mead polish with remaining budget
        remaining_calls = max_calls - call_count
        if remaining_calls > 10 and time.time() < deadline - 5:
            x_start = best_x.copy()
            
            try:
                result_nm = fmin(
                    wrapped_func,
                    x_start,
                    xtol=1e-12,
                    ftol=1e-15,
                    maxiter=200,
                    disp=False
                )
            except Exception:
                pass
        
        # If no valid result, use a default
        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = wrapped_func(best_x)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
