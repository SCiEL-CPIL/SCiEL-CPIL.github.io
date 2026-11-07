"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # 330s wall clock limit. Keep a safety margin of 5s.
        deadline = start_time + 325.0
        
        best_x = None
        best_f = np.inf
        call_count = 0
        n = self.dim
        
        def make_call(x):
            nonlocal call_count
            if call_count >= self.budget:
                return None
            if time.time() > deadline:
                return None
            fx = func(x)
            call_count += 1
            return fx

        # 1. Initial Probe
        # Start at origin. It's a safe, central point.
        x0 = np.zeros(n)
        f0 = make_call(x0)
        if f0 is not None and np.isfinite(f0):
            best_f = f0
            best_x = x0.copy()
        else:
            # Fallback if origin fails (unlikely but safe)
            return None, np.inf

        # 2. Simulated Annealing Parameters
        # We want to explore early to get a good AOCC.
        # High initial temperature allows large jumps.
        # Fast cooling rate ensures we refine quickly.
        
        # Budget allocation:
        # Use most of the budget for SA.
        # Reserve a small amount for a final "polish" if needed, but SA should handle it.
        
        # Temperature schedule:
        # T(t) = T0 * alpha^t
        # We want T to drop to near 0 by the end of the budget.
        # Let total_steps = budget - 1 (since we already called once)
        # T_end = 1e-4 (small enough to accept only improvements)
        # T0 = 5.0 (large enough to jump across the [-5,5] box)
        
        T0 = 5.0
        T_end = 1e-5
        # alpha = (T_end / T0)^(1 / total_steps)
        # We don't know exact total steps if we break early, but we can estimate.
        # Let's assume we use ~90% of budget for SA.
        est_steps = int(self.budget * 0.9)
        if est_steps < 10:
            est_steps = 10
            
        alpha = (T_end / T0) ** (1.0 / est_steps)
        
        current_T = T0
        x = best_x.copy()
        f = best_f
        
        # Step size adaptation for SA:
        # Instead of fixed Gaussian sigma, we can adapt it based on acceptance rate.
        # Or simply use a fixed sigma that scales with T?
        # Standard SA uses fixed sigma or sigma proportional to T.
        # Let's use sigma = 1.0 initially, and maybe scale it down slightly as T drops?
        # Actually, for ill-conditioned functions, a fixed sigma might be better if T handles the acceptance.
        # But if the function is very flat, we need larger steps.
        # Let's try sigma = 0.5 * (T / T0 + 0.1) to keep it somewhat active.
        # Or simpler: fixed sigma = 1.0. The "rotation" in the problem description implies 
        # the landscape is rotated, so axis-aligned steps are fine if isotropic.
        
        sigma = 1.0
        
        # Reheat mechanism:
        # If no improvement for X steps, increase T to escape local minima.
        stagnation_limit = 20
        stagnation_count = 0
        max_reheats = 5
        reheats_used = 0
        
        while call_count < self.budget and time.time() < deadline:
            # Generate neighbor
            # Use standard normal perturbation
            delta = np.random.normal(0, sigma, n)
            x_new = x + delta
            x_new = np.clip(x_new, -5, 5)
            
            f_new = make_call(x_new)
            if f_new is None:
                break
                
            if not np.isfinite(f_new):
                # If we hit an invalid point (shouldn't happen with clipping, but safe)
                f_new = f + 1000.0 
                
            # Acceptance criterion
            if f_new < f:
                # Always accept improvement
                x = x_new
                f = f_new
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
                    stagnation_count = 0
            else:
                # Deterioration: accept with probability exp(-delta_f / T)
                delta_f = f_new - f
                if delta_f > 0:
                    # Avoid log(0) or division by zero
                    if current_T < 1e-12:
                        p = 0.0
                    else:
                        p = np.exp(-delta_f / current_T)
                    
                    if np.random.rand() < p:
                        x = x_new
                        f = f_new
                        stagnation_count += 1
                    else:
                        stagnation_count += 1
                else:
                    # This case is covered by f_new < f, but just in case of floating point
                    x = x_new
                    f = f_new
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                    stagnation_count = 0

            # Update Temperature
            current_T *= alpha
            
            # Reheat Logic
            if stagnation_count >= stagnation_limit:
                if reheats_used < max_reheats:
                    # Reheat: jump back to a higher temperature
                    current_T = max(current_T, T0 * 0.5)
                    # Also perturb current position to escape the local basin
                    # This is a "kicked" SA or "iterated" SA hybrid
                    x = x + np.random.normal(0, 1.0, n)
                    x = np.clip(x, -5, 5)
                    f = make_call(x)
                    if f is None:
                        break
                    if not np.isfinite(f):
                        f = best_f + 1000.0
                        x = best_x.copy()
                        f = best_f
                    else:
                        if f < best_f:
                            best_f = f
                            best_x = x.copy()
                    
                    stagnation_count = 0
                    reheats_used += 1
                else:
                    # No more reheats, just keep cooling (or stop if T is tiny)
                    if current_T < 1e-8:
                        break

        # 3. Final Polish (Optional)
        # If we have budget left and T is very low, we might do a few greedy steps.
        # But SA with low T is already effectively greedy.
        # Let's just ensure we return the best found.
        
        if best_x is None:
            # Fallback
            best_x = x0
            best_f = f0

        return best_x, best_f
# EVOLVE-BLOCK-END
