"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Hybrid optimizer for BBOB f7 (Step Ellipsoidal).
    1. Coarse global search using a lightweight (1+1)-ES with adaptive step size.
    2. Local refinement using L-BFGS-B to exploit the smooth regions near the optimum.
    
    The (1+1)-ES is robust to the 'step' nature of the function and helps find
    the correct orthant/region quickly without the overhead of full CMA-ES,
    especially given the small dimension (5) and budget (500).
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: evaluator kills at 330s. We stop at 310s.
        deadline = start_time + 310.0

        dim = self.dim
        rng = np.random.default_rng(42)

        calls = 0
        best_x = None
        best_f = np.inf

        def _eval(x):
            nonlocal calls, best_x, best_f
            if calls >= self.budget:
                return None
            if time.time() > deadline:
                return None
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # --- Phase 1: Global Search with (1+1)-ES ---
        # Initialize at origin. For BBOB, origin is often a decent start,
        # or at least (1+1)-ES will explore away from it.
        # To be more robust against shifts, we could try a few random starts,
        # but (1+1)-ES with a large initial sigma covers space well.
        
        # Strategy: Run (1+1)-ES for a portion of the budget (e.g., 60-70%).
        # Then switch to local refinement.
        
        # Parameters for (1+1)-ES
        # Sigma adaptation: If step succeeds, increase sigma; if fails, decrease.
        # A common scheme is the 1/5-th rule or similar.
        # Let's use a simple multiplicative update.
        
        current_x = np.zeros(dim)
        sigma = 2.0  # Initial step size. Box is [-5, 5]^5, so 2.0 is reasonable.
        
        # We will run this loop until we have used ~70% of the budget or time runs out.
        # Let's allocate 350 calls to global search, leaving 150 for local.
        global_budget = 350
        global_calls = 0
        
        while global_calls < global_budget and calls < self.budget and time.time() < deadline:
            # Generate candidate
            z = rng.standard_normal(dim)
            candidate_x = current_x + sigma * z
            # Clip to box
            candidate_x = np.clip(candidate_x, -5, 5)
            
            f_candidate = _eval(candidate_x)
            if f_candidate is None:
                break
                
            if f_candidate < best_f:
                # Success
                current_x = candidate_x
                # Increase step size
                sigma *= 1.05
            else:
                # Failure
                # Decrease step size
                sigma *= 0.95
            
            global_calls += 1
            sigma = np.clip(sigma, 1e-5, 10.0)

        # --- Phase 2: Local Refinement ---
        # If we still have budget and time, use L-BFGS-B or Nelder-Mead.
        # L-BFGS-B is generally better for smooth functions, but f7 has steps.
        # However, near the optimum, the function is locally constant or smooth.
        # The "step" implies that once you are in the optimal orthant, the function
        # becomes an ellipsoidal function, which is smooth.
        # So L-BFGS-B should work well if we are close to the true optimum.
        
        if best_x is not None and calls < self.budget and time.time() < deadline - 5:
            remaining = self.budget - calls
            
            # Define a wrapper for scipy that respects the budget and deadline
            def wrapped_func(x):
                nonlocal calls, best_f, best_x
                if calls >= self.budget:
                    return best_f
                if time.time() > deadline:
                    return best_f
                fx = func(x)
                calls += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                return fx

            # Try L-BFGS-B first. It uses gradient estimates.
            # Bounds are [-5, 5]
            bounds = [(-5, 5)] * dim
            
            try:
                # L-BFGS-B
                res = minimize(wrapped_func, best_x.copy(), method='L-BFGS-B',
                               bounds=bounds,
                               options={'maxiter': min(remaining, 100), 'ftol': 1e-10, 'gtol': 1e-8})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

            # If L-BFGS-B didn't help much or if we have even more budget,
            # try Nelder-Mead which is derivative-free and robust to non-smoothness.
            # But L-BFGS-B is usually faster. Let's stick to L-BFGS-B unless it fails.
            # Actually, since f7 is non-smooth, Nelder-Mead might be safer if L-BFGS-B gets stuck.
            # But given the "Step Ellipsoidal" nature, once in the basin, it's smooth.
            
            # Let's add a second round of local search with Nelder-Mead if we have significant budget left.
            if calls < self.budget and time.time() < deadline - 5:
                remaining2 = self.budget - calls
                if remaining2 > 20:
                    try:
                        res_nm = minimize(wrapped_func, best_x.copy(), method='Nelder-Mead',
                                          options={'maxiter': min(remaining2, 50), 'xatol': 1e-8, 'fatol': 1e-10})
                        if res_nm.fun < best_f:
                            best_f = res_nm.fun
                            best_x = res_nm.x.copy()
                    except Exception:
                        pass

        if best_x is None:
            # Fallback if no evaluation was made (should not happen)
            best_x = np.zeros(dim)
            best_f = func(best_x)

        return best_x, best_f

# EVOLVE-BLOCK-END
