"""Portfolio search optimizer for BBOB f20 (Schwefel) with explicit deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """
    Multi-strategy portfolio optimizer for BBOB f20 (Schwefel) on [-5, 5]^5.
    Phase 1: Round-robin between multiple (1+1)-ES and a CMA-ES for global exploration.
    Phase 2: Focused (1+1)-ES polish around the best point found.
    Explicit deadline guard to stay well within the 330s wall-clock limit.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        total_budget = self.budget
        dim = self.dim

        best_x = None
        best_fx = float('inf')
        calls_used = 0

        def try_func(x):
            nonlocal calls_used, best_fx, best_x
            if calls_used >= total_budget:
                return float('inf')
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_used += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        def time_ok():
            return time.time() < deadline

        def budget_ok():
            return calls_used < total_budget

        # Phase 1: Global Exploration (Random Sampling)
        # Use 50% of budget for diverse starts
        n_starts = 10
        phase1_budget = int(total_budget * 0.5)
        
        starts = []
        rng = np.random.default_rng(12345)
        
        # Generate diverse starting points
        for _ in range(n_starts):
            x = rng.uniform(-5, 5, dim)
            fx = try_func(x)
            starts.append((x, fx))
            
            # If we have enough budget, do some local random search around this start
            # to find a better basin
            local_budget = 10
            for _ in range(local_budget):
                if not budget_ok() or not time_ok():
                    break
                y = np.clip(x + 1.0 * rng.standard_normal(dim), -5, 5)
                fy = try_func(y)
                if fy < fx:
                    x, fx = y, fy
            
            starts.append((x, fx))
            # Remove duplicates or keep only best? Let's keep unique bests
            # Actually, let's just collect candidates and pick top ones for Phase 2

        # Sort starts by fitness
        starts.sort(key=lambda item: item[1])
        
        # Phase 2: Local Optimization with Nelder-Mead on top starts
        # We will allocate remaining budget to these local searches
        # Each Nelder-Mead run will use a fraction of the remaining budget
        
        remaining_budget = total_budget - calls_used
        n_local_runs = min(5, len(starts)) # Optimize top 5 basins
        budget_per_run = int(remaining_budget / n_local_runs) if n_local_runs > 0 else 0
        
        for i in range(n_local_runs):
            if not budget_ok() or not time_ok():
                break
            
            x_start = starts[i][0].copy()
            fx_start = starts[i][1]
            
            # Implement a simple Nelder-Mead step loop
            # Initialize simplex
            simplex = [x_start.copy()]
            f_vals = [fx_start]
            
            # Create other vertices
            for j in range(dim):
                x_j = x_start.copy()
                x_j[j] += 0.5 if x_start[j] < 0 else -0.5
                x_j = np.clip(x_j, -5, 5)
                f_j = try_func(x_j)
                simplex.append(x_j)
                f_vals.append(f_j)
                
                if not budget_ok() or not time_ok():
                    break
            
            if len(simplex) < dim + 1:
                continue

            # Nelder-Mead iterations
            max_iter = 20
            for _ in range(max_iter):
                if not budget_ok() or not time_ok():
                    break
                
                # Sort simplex by function value
                idx = np.argsort(f_vals)
                simplex = [simplex[i] for i in idx]
                f_vals = [f_vals[i] for i in idx]
                
                # Calculate centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_ref = centroid + (centroid - simplex[-1])
                x_ref = np.clip(x_ref, -5, 5)
                f_ref = try_func(x_ref)
                
                if f_ref < f_vals[0]:
                    # Expansion
                    x_exp = centroid + 2 * (centroid - simplex[-1])
                    x_exp = np.clip(x_exp, -5, 5)
                    f_exp = try_func(x_exp)
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        f_vals[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        f_vals[-1] = f_ref
                elif f_ref < f_vals[-2]:
                    # Accept reflection
                    simplex[-1] = x_ref
                    f_vals[-1] = f_ref
                else:
                    # Contraction
                    x_con = centroid + 0.5 * (simplex[-1] - centroid)
                    x_con = np.clip(x_con, -5, 5)
                    f_con = try_func(x_con)
                    
                    if f_con < f_vals[-1]:
                        simplex[-1] = x_con
                        f_vals[-1] = f_con
                    else:
                        # Shrink
                        for k in range(1, len(simplex)):
                            simplex[k] = simplex[0] + 0.5 * (simplex[k] - simplex[0])
                            simplex[k] = np.clip(simplex[k], -5, 5)
                            f_vals[k] = try_func(simplex[k])
                            if not budget_ok() or not time_ok():
                                break

        return best_x, best_fx
# EVOLVE-BLOCK-END
