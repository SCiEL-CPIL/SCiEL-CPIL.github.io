"""Hybrid (1+1)-ES optimizer with multi-start for BBOB f11 (Discus)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer using a fast (1+1)-ES with multi-start strategy.
    Designed for high-speed convergence on the BBOB Discus function.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: 330s kill limit, aim to finish by 320s
        deadline = start_time + 320
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = float('inf')
        call_count = 0

        def try_eval(x):
            nonlocal call_count, best_f, best_x
            if call_count >= budget:
                return None
            # Clip to box [-5, 5]
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def time_ok():
            return time.time() < deadline - 5

        def run_1plus1_es(x_start, sigma_start, max_evals):
            """
            Run a (1+1)-ES from x_start for up to max_evals function calls.
            Returns the best x and f found in this run.
            """
            nonlocal best_x, best_f, call_count
            
            x = x_start.copy()
            # Evaluate starting point if not already done? 
            # We assume x_start is a new point or we just start optimizing from it.
            # To be safe and consistent with budget, we evaluate x_start.
            fx = try_eval(x)
            if fx is None:
                return x, fx
            
            sigma = sigma_start
            # Success rate adaptation parameters
            success_count = 0
            window = 10
            evals_in_run = 0
            
            while call_count < budget and evals_in_run < max_evals:
                if not time_ok():
                    break
                
                # Generate offspring
                y = x + sigma * rng.standard_normal(dim)
                fy = try_eval(y)
                if fy is None:
                    break
                evals_in_run += 1
                
                if fy < fx:
                    # Success: move to y, increase sigma
                    x = y
                    fx = fy
                    sigma *= 1.2
                    success_count += 1
                else:
                    # Failure: stay, decrease sigma
                    sigma *= 0.85
                    # success_count unchanged
                
                # Adapt sigma based on success rate every 'window' steps
                if evals_in_run % window == 0:
                    success_rate = success_count / window
                    if success_rate > 0.5:
                        sigma *= 1.1
                    elif success_rate < 0.2:
                        sigma *= 0.9
                    success_count = 0
                
                # Early stopping if sigma becomes too small or no improvement
                if sigma < 1e-8:
                    break

            return x, fx

        # Phase 1: Primary search from center
        # The minimum is in [-4, 4]^5. Center (0,0,0,0,0) is a good bet.
        x0 = np.zeros(dim)
        sigma0 = 3.0 # Large initial step to explore quickly
        # Allocate 60% of budget to the main run
        main_budget = int(budget * 0.6)
        run_1plus1_es(x0, sigma0, main_budget)

        # Phase 2: Multi-start if budget remains
        # If we haven't found a very good solution (heuristic: if best_f is still high),
        # or just to be safe, try a few other starts.
        # For f11, the landscape is unimodal but rotated. 
        # If the first run converged, it's likely in the global basin.
        # If it stagnated, a restart might help.
        
        remaining_budget = budget - call_count
        if remaining_budget > 20 and time_ok():
            # Generate diverse restart points
            # Avoid points too close to the current best
            restart_points = []
            # Random points in the box
            for _ in range(5):
                rp = rng.uniform(-4.5, 4.5, dim)
                # Check distance to best_x
                if best_x is not None and np.linalg.norm(rp - best_x) < 1.0:
                    continue
                restart_points.append(rp)
            
            # Also try the origin again with a smaller sigma if it failed before? 
            # No, we already tried origin.
            
            # Sort restart points by potential (random here, but we can prioritize corners if needed)
            # For Discus, corners are less likely to be the min, but random is fine.
            
            for rp in restart_points:
                if not time_ok():
                    break
                if call_count >= budget:
                    break
                
                # Check if we are close to best
                if best_x is not None:
                    dist = np.linalg.norm(rp - best_x)
                    if dist < 0.5:
                        continue
                
                # Allocate remaining budget divided among restarts, or a fixed chunk
                # Let's give each restart up to 20% of the *original* budget or remaining/2
                chunk_budget = min(100, (budget - call_count) // 2)
                if chunk_budget < 10:
                    break
                    
                # Start with a smaller sigma for restarts to avoid overshooting
                sigma_restart = 1.5
                run_1plus1_es(rp, sigma_restart, chunk_budget)

        # Phase 3: Fine-tuning / Polishing
        # Use the last 5-10% of budget for very precise local search around the best point
        if best_x is not None and call_count < budget and time_ok():
            fine_budget = min(20, budget - call_count)
            x_fine = best_x.copy()
            sigma_fine = 0.05
            
            for _ in range(fine_budget):
                if not time_ok():
                    break
                if call_count >= budget:
                    break
                
                y = x_fine + sigma_fine * rng.standard_normal(dim)
                fy = try_eval(y)
                if fy is None:
                    break
                
                if fy < best_f:
                    x_fine = y
                    # Keep sigma or slightly increase if improving
                    sigma_fine *= 1.1
                else:
                    sigma_fine *= 0.8

        # Ensure we return a valid result
        if best_x is None:
            best_x = np.zeros(dim)
            # This should not happen if budget > 0, but just in case
            try:
                best_f = func(best_x)
            except:
                best_f = 0.0

        return best_x, best_f
# EVOLVE-BLOCK-END
