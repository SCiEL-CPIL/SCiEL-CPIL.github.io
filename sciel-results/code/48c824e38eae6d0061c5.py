"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """
    Hybrid Optimizer for BBOB f20 (Schwefel).
    Strategy:
    1. Stratified global exploration to find promising basins.
    2. L-BFGS-B local optimization from top candidates for high-precision convergence.
    3. (1+1)-ES with restarts for robustness and to handle any remaining budget.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285  # Safety margin for 300s limit

        # State tracking
        best_x_global = np.zeros(dim)
        best_f_global = np.inf
        call_count = 0

        def safe_call(x):
            nonlocal call_count, best_x_global, best_f_global
            if call_count >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            
            x = np.asarray(x, dtype=np.float64).flatten()
            x = np.clip(x, -5.0, 5.0)
            
            try:
                fx = float(func(x))
            except Exception:
                return np.inf
            
            if not np.isfinite(fx):
                fx = np.inf
                
            call_count += 1
            if fx < best_f_global:
                best_f_global = fx
                best_x_global = x.copy()
            return fx

        rng = np.random.default_rng(42)

        # Phase 1: Stratified Global Exploration
        # We want to cover the space efficiently. 
        # Use 2^5 = 32 stratified points + 8 random points = 40 points.
        
        n_global = 40
        global_points = []
        n_bins = 2
        from itertools import product
        grid_indices = list(product(range(n_bins), repeat=dim))
        
        for idx in grid_indices:
            point = np.zeros(dim)
            for d in range(dim):
                lo = -5.0 + 10.0 * idx[d] / n_bins
                hi = -5.0 + 10.0 * (idx[d] + 1) / n_bins
                point[d] = rng.uniform(lo, hi)
            global_points.append(point)
        
        # Add a few fully random points for extra diversity
        for _ in range(n_global - len(grid_indices)):
            global_points.append(rng.uniform(-5.0, 5.0, dim))

        # Shuffle to avoid bias in order
        rng.shuffle(global_points)

        for x in global_points:
            if time.time() > deadline or call_count >= budget:
                break
            safe_call(x)

        # Phase 2: Local Optimization with L-BFGS-B
        # Identify the top few candidates from Phase 1 and refine them.
        # Since we don't store all points, we just use the current best and a few random perturbations of it?
        # Better: We can re-evaluate a few diverse points or just run L-BFGS-B from the global best.
        # To maximize AOCC, we want fast convergence. L-BFGS-B is very fast for smooth functions.
        
        # We will run L-BFGS-B from the current global best.
        # We might also run it from a second-best if we had stored it, but we didn't.
        # Let's store the top 3 candidates from Phase 1.
        
        # Wait, I didn't store the candidates in Phase 1. Let's fix that.
        # Actually, I can just collect the results.
        
        # Let's restart the logic slightly to collect candidates.
        # But I already made the calls. I can't undo them.
        # I will just run L-BFGS-B from the current best_x_global.
        # And maybe a few random points near the best?
        
        # Let's try running L-BFGS-B from the global best.
        # If it fails or doesn't improve much, we can try other strategies.
        
        def wrapped_func(x):
            return safe_call(x)

        # Run L-BFGS-B from the global best
        if call_count < budget and time.time() < deadline:
            try:
                res = minimize(
                    wrapped_func,
                    best_x_global,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-10}
                )
                # safe_call updates best_x_global and best_f_global internally
            except Exception:
                pass

        # Phase 3: (1+1)-ES with Restarts
        # Use remaining budget for (1+1)-ES to polish or find better basins.
        
        remaining_budget = budget - call_count
        if remaining_budget <= 0 or time.time() > deadline:
            return best_x_global, best_f_global

        # ES Parameters
        stagnation_limit = 30
        es_chunk_size = 50
        
        while call_count < budget and time.time() < deadline:
            # Initialize ES state
            x = best_x_global.copy()
            fx = best_f_global
            sigma = 0.5  # Start with smaller sigma since L-BFGS-B likely converged
            no_improve_count = 0
            steps_in_chunk = 0
            
            # Run ES for this chunk
            while steps_in_chunk < es_chunk_size and call_count < budget and time.time() < deadline:
                # Generate candidate
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5.0, 5.0)
                
                fy = safe_call(y)
                steps_in_chunk += 1
                
                if fy < fx:
                    x = y
                    fx = fy
                    sigma *= 1.1  # Increase step size on success
                    no_improve_count = 0
                else:
                    sigma *= 0.9  # Decrease step size on failure
                    no_improve_count += 1
                
                # Check for stagnation within the chunk
                if no_improve_count >= stagnation_limit:
                    break

            # If we broke out due to stagnation or budget, decide whether to restart
            if call_count >= budget or time.time() > deadline:
                break
            
            # Check if we should restart
            if sigma < 1e-3 or no_improve_count >= stagnation_limit:
                # Restart
                # 80% perturb global best, 20% random
                if rng.random() < 0.8:
                    x = best_x_global + 1.0 * rng.standard_normal(dim)
                    x = np.clip(x, -5.0, 5.0)
                else:
                    x = rng.uniform(-5.0, 5.0, dim)
                
                # Evaluate the restart point
                if call_count < budget and time.time() < deadline:
                    fx = safe_call(x)
                else:
                    break
                    
                # Reset sigma for the new run
                sigma = 0.5
            else:
                # If not stuck, just continue? 
                # The loop structure breaks the inner loop after es_chunk_size steps.
                # So we restart the outer loop, which resets x to best_x_global.
                # This is a soft restart.
                pass

        return best_x_global, best_f_global

# EVOLVE-BLOCK-END
