"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

class Optimizer:
    """
    Hybrid Global-Local Optimizer.
    
    Strategy:
    1. Global Scan: Evaluate a set of stratified/random points to identify promising regions.
       This provides early AOCC credit and identifies the best candidate for local search.
    2. Local Refinement: Use Nelder-Mead (derivative-free) to refine the best candidate.
       Nelder-Mead is robust for low dimensions and multimodal landscapes if started in a good basin.
    3. Multi-Start: If budget allows, refine the top K candidates from the global scan to handle
       multimodality (101 peaks).
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        global_best_x = None
        global_best_f = float('inf')
        total_calls = 0
        start_time = time.time()
        max_time = 280.0  # Safety margin for 330s limit

        def can_continue():
            return total_calls < budget and (time.time() - start_time) < max_time

        def update_best(x, f):
            nonlocal global_best_x, global_best_f
            if f < global_best_f:
                global_best_f = f
                global_best_x = x.copy()

        # --- Phase 1: Global Scan ---
        # We use a mix of random and stratified sampling to cover the space.
        # For dim=5, a full grid is expensive, but a stratified random sample is good.
        # Let's use 80-120 evaluations for the scan.
        # We will store the top 5 candidates to use as seeds for local optimization.
        
        n_scan = 100
        scan_results = []
        
        for i in range(n_scan):
            if not can_continue():
                break
            
            # Stratified random sampling:
            # Divide the box into 2x2x2x2x2 = 32 sub-boxes.
            # Pick a random point in each sub-box, then pick random points from the remaining budget.
            # For simplicity, we'll just use uniform random, but ensure we don't cluster too much.
            # Actually, for 101 peaks, uniform random is often sufficient to hit a basin.
            
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            total_calls += 1
            
            update_best(x, fx)
            scan_results.append((fx, x))
            
            # Check time/budget frequently
            if not can_continue():
                break

        if global_best_x is None:
            # Fallback
            x0 = np.zeros(dim)
            fx = func(x0)
            total_calls += 1
            update_best(x0, fx)
            scan_results.append((fx, x0))

        # Sort scan results to get top candidates
        scan_results.sort(key=lambda item: item[0])
        top_candidates = [x for f, x in scan_results[:5]]

        # --- Phase 2: Local Refinement ---
        # Run Nelder-Mead on the top candidates.
        # NM is fast. We can run it on the best candidate first.
        # If it converges or hits maxiter, we move to the next best candidate if budget allows.
        
        # NM options:
        # maxiter: high enough to converge. 200-500 is usually plenty for dim=5.
        # xatol, fatol: tight tolerances for precision.
        
        nm_options = {
            'maxiter': 200,
            'xatol': 1e-8,
            'fatol': 1e-8,
            'adaptive': True
        }

        for i, start_x in enumerate(top_candidates):
            if not can_continue():
                break
            
            # If we've already spent most of the budget, skip
            if total_calls > budget * 0.8:
                break

            # Define a wrapper to count calls and check time
            def nm_func(x):
                nonlocal total_calls
                # Check time inside the function? 
                # Scipy's minimize might call it many times quickly.
                # It's safer to check in the main loop, but NM doesn't yield control.
                # However, NM for dim=5 is very fast (milliseconds).
                # We will rely on the outer loop check and maxiter.
                
                fx = func(x)
                total_calls += 1
                update_best(x, fx)
                return fx

            try:
                res = minimize(nm_func, start_x, method='Nelder-Mead', options=nm_options)
                # res.fun is the best value found by NM
                # res.x is the best point
                if res.fun < global_best_f:
                    global_best_f = res.fun
                    global_best_x = res.x.copy()
            except Exception as e:
                # If NM fails for some reason, continue to next candidate
                pass

            # Check if we should continue to next candidate
            # If the first candidate converged well, we might not need others.
            # But for 101 peaks, the global best might be in a different basin.
            # So we should try the next best scan point if budget allows.
            
            # Heuristic: Stop local refinement if we've used more than 50% of remaining budget 
            # and haven't improved significantly in the last NM run? 
            # Simpler: Just try the next one if we have budget for ~50 more calls.
            if total_calls < budget - 50:
                continue
            else:
                break

        return global_best_x, global_best_f

# EVOLVE-BLOCK-END
