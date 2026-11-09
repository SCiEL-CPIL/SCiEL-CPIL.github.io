"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Multi-start Nelder-Mead simplex optimizer for BBOB f19.
    Strategy:
    1. Brief exploration phase to find diverse starting points.
    2. Sequential Nelder-Mead runs from the best found points.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270.0
        
        dim = self.dim
        budget = self.budget
        
        global_best_x = None
        global_best_f = float('inf')
        calls_made = 0
        
        def call_func(x):
            nonlocal calls_made, global_best_x, global_best_f
            if calls_made >= budget:
                return global_best_f
            val = func(x)
            calls_made += 1
            if val < global_best_f:
                global_best_f = val
                global_best_x = x.copy()
            return val

        # Phase 1: Exploration (10% of budget)
        n_explore = max(10, int(0.1 * budget))
        
        x_center = np.zeros(dim)
        f_center = call_func(x_center)
        
        for _ in range(n_explore - 1):
            if time.time() > deadline or calls_made >= budget:
                break
            x_rand = np.random.uniform(-5, 5, dim)
            call_func(x_rand)
        
        if global_best_x is None:
            global_best_x = x_center
            global_best_f = f_center

        # Phase 2: Multi-start Nelder-Mead with restarts
        n_restarts = 3
        for restart in range(n_restarts):
            remaining_budget = budget - calls_made
            if remaining_budget <= 0 or time.time() > deadline:
                break
            
            x_start = global_best_x.copy()
            f_start = global_best_f
            
            # Nelder-Mead parameters
            # Initial simplex: centered at x_start with small spread
            # Spread size: 0.1 * box_size? Box size is 10. So 1.0 per dim? 
            # Let's use 0.5 to be safe.
            spread = 0.5
            
            # Create initial simplex
            simplex = []
            simplex.append(x_start.copy())
            simplex_f = [f_start]
            
            for i in range(dim):
                x_i = x_start.copy()
                # Perturb dimension i
                # If x_start[i] + spread > 5, go negative, else positive
                if x_start[i] + spread > 5.0:
                    x_i[i] -= spread
                else:
                    x_i[i] += spread
                x_i = np.clip(x_i, -5, 5)
                f_i = call_func(x_i)
                simplex.append(x_i)
                simplex_f.append(f_i)
                
                if calls_made >= budget or time.time() > deadline:
                    break
            
            # If we ran out of budget during simplex init, return best
            if len(simplex) < dim + 1:
                return global_best_x, global_best_f

            # Nelder-Mead Loop
            alpha = 1.0   # Reflection
            gamma = 2.0   # Expansion
            rho = 0.5     # Contraction
            sigma = 0.5   # Shrinkage
            
            max_iterations = int(remaining_budget)
            
            for it in range(max_iterations):
                if time.time() > deadline or calls_made >= budget:
                    break
                
                # Sort simplex by fitness
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]
                
                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Worst point
                x_worst = simplex[-1]
                f_worst = simplex_f[-1]
                
                # Reflection
                x_reflect = centroid + alpha * (centroid - x_worst)
                x_reflect = np.clip(x_reflect, -5, 5)
                f_reflect = call_func(x_reflect)
                
                if f_reflect < simplex_f[0]:
                    # Expansion
                    x_expand = centroid + gamma * (centroid - x_worst)
                    x_expand = np.clip(x_expand, -5, 5)
                    f_expand = call_func(x_expand)
                    
                    if f_expand < f_reflect:
                        # Replace worst with expanded
                        simplex[-1] = x_expand
                        simplex_f[-1] = f_expand
                    else:
                        # Replace worst with reflected
                        simplex[-1] = x_reflect
                        simplex_f[-1] = f_reflect
                elif f_reflect < simplex_f[-2]:
                    # Replace worst with reflected
                    simplex[-1] = x_reflect
                    simplex_f[-1] = f_reflect
                else:
                    # Contraction
                    x_contract = centroid + rho * (x_worst - centroid)
                    x_contract = np.clip(x_contract, -5, 5)
                    f_contract = call_func(x_contract)
                    
                    if f_contract < f_worst:
                        # Replace worst with contracted
                        simplex[-1] = x_contract
                        simplex_f[-1] = f_contract
                    else:
                        # Shrinkage
                        # Move all points (except best) towards best
                        x_best = simplex[0]
                        for i in range(1, len(simplex)):
                            simplex[i] = x_best + sigma * (simplex[i] - x_best)
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            simplex_f[i] = call_func(simplex[i])
                            
                            if time.time() > deadline or calls_made >= budget:
                                break
                        
                        if time.time() > deadline or calls_made >= budget:
                            break
                            
                        # Check convergence
                        # If simplex is very small, break
                        spread_simplex = np.max(np.abs(simplex - simplex[0]), axis=0)
                        if np.max(spread_simplex) < 1e-6:
                            break

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
