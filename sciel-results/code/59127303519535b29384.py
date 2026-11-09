"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-Start Adaptive Coordinate Descent for Lunacek bi-Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280  # Safety margin
        rng = np.random.default_rng(0)
        
        calls_made = 0
        best_x = None
        best_f = np.inf
        
        def safe_call(x):
            nonlocal calls_made
            if calls_made >= self.budget:
                return np.inf
            fx = func(x)
            calls_made += 1
            return fx

        # Phase 1: Global Exploration
        # Use 40 evaluations for global search
        global_budget = 40
        candidates = []
        
        # Stratified random sampling
        # Split the range [-5, 5] into bins?
        # Simpler: Uniform random
        for _ in range(global_budget):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, self.dim)
            fx = safe_call(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            candidates.append((fx, x.copy()))
            
        # Sort candidates
        candidates.sort(key=lambda item: item[0])
        
        # Phase 2: Local Exploitation
        # Select top 4 candidates
        top_k = min(4, len(candidates))
        remaining_budget = self.budget - calls_made
        
        if remaining_budget > 0 and top_k > 0:
            budget_per_restart = remaining_budget // top_k
            
            for i in range(top_k):
                if time.time() > deadline:
                    break
                if calls_made >= self.budget:
                    break
                
                start_x = candidates[i][1].copy()
                start_f = candidates[i][0]
                
                # Run Adaptive Coordinate Descent
                x = start_x
                fx = start_f
                sigma = 1.0
                stagnation_count = 0
                
                # Max iterations for this restart
                # budget_per_restart is the number of calls we can use.
                # Each AC D iteration uses at most 2*dim calls.
                # So max_iter ~ budget_per_restart / (2*dim)
                max_iter = max(1, budget_per_restart // (2 * self.dim))
                
                for _ in range(max_iter):
                    if time.time() > deadline or calls_made >= self.budget:
                        break
                    
                    improved = False
                    # Try each coordinate
                    for coord in range(self.dim):
                        if time.time() > deadline or calls_made >= self.budget:
                            break
                            
                        # Try positive step
                        y = x.copy()
                        y[coord] = np.clip(y[coord] + sigma, -5.0, 5.0)
                        fy = safe_call(y)
                        
                        if fy < fx:
                            x = y
                            fx = fy
                            improved = True
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()
                        
                        # Try negative step
                        y = x.copy()
                        y[coord] = np.clip(y[coord] - sigma, -5.0, 5.0)
                        fy = safe_call(y)
                        
                        if fy < fx:
                            x = y
                            fx = fy
                            improved = True
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()
                    
                    if improved:
                        sigma = min(sigma * 1.2, 5.0)
                        stagnation_count = 0
                    else:
                        sigma = max(sigma * 0.7, 0.05)
                        stagnation_count += 1
                        # If stuck for a while, try a larger jump?
                        # Or just give up on this restart to save budget for others?
                        # If sigma is very small, we are converged.
                        if sigma < 0.1:
                            break
                            
        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = func(best_x)
            
        return best_x, best_f
# EVOLVE-BLOCK-END
