import time
import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 300s limit, leave 10s margin
        
        dim = self.dim
        total_budget = self.budget
        
        # Strategy: Multi-Start Nelder-Mead
        # We divide the budget into several restarts.
        # For dim=5, a Nelder-Mead simplex takes dim+1 = 6 points to form.
        # Each step involves up to dim+1 evaluations (though often fewer if we reuse).
        # Let's allocate a fixed number of evaluations per restart to ensure diversity.
        
        # If we use too many restarts, each local search is shallow.
        # If we use too few, we might miss the global basin.
        # Let's try 5-10 restarts depending on budget.
        # With budget=500, and dim=5, we can afford ~10-20 evaluations per step.
        
        # Let's define a helper for a fast local search: Nelder-Mead
        # We will implement a simplified, robust version.
        
        best_x_global = np.zeros(dim)
        best_f_global = float('inf')
        
        eval_count = 0
        rng = np.random.default_rng(42) # Fixed seed for reproducibility, but different from CMA
        
        # Determine number of restarts
        # Let's say we use 20% of the budget for the first restart (most critical for AOCC),
        # and distribute the rest among subsequent restarts.
        # Or simply equal splits? Equal splits is simpler and robust.
        # 500 / 10 restarts = 50 evals per restart.
        # Nelder-Mead with dim=5 typically converges in < 20-30 steps if close.
        # 50 evals is enough for a local refinement from a random start.
        
        num_restarts = 10
        budget_per_restart = total_budget // num_restarts
        
        for r in range(num_restarts):
            if eval_count >= total_budget:
                break
            if time.time() > deadline:
                break
            
            # 1. Initialize simplex
            # Start point: Random within [-5, 5]
            # To improve early progress, the first few restarts should be more "central" or structured?
            # No, random is good for global coverage.
            # However, for AOCC, the very first call matters most.
            # Let's make the first restart start at 0, and subsequent ones random.
            if r == 0:
                center = np.zeros(dim)
            else:
                center = rng.uniform(-4, 4, dim) # Min is in [-4,4], so sampling inside is good
            
            # Initialize simplex
            # Simplex: dim+1 points
            # Point 0: center
            # Points 1..dim: center + delta * e_i
            delta = 1.0 # Initial step size
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = center
            for i in range(dim):
                simplex[i + 1] = center
                simplex[i + 1][i] += delta
            
            # Evaluate initial simplex
            f_vals = np.zeros(dim + 1)
            for i in range(dim + 1):
                if eval_count >= total_budget or time.time() > deadline:
                    break
                x_i = simplex[i]
                # Clip to box
                x_i = np.clip(x_i, -5, 5)
                f_i = func(x_i)
                eval_count += 1
                f_vals[i] = f_i
                
                if f_i < best_f_global:
                    best_f_global = f_i
                    best_x_global = x_i.copy()
            
            if eval_count >= total_budget or time.time() > deadline:
                break
            
            # 2. Nelder-Mead Loop
            # We run until budget for this restart is exhausted or convergence
            local_budget_used = dim + 1
            max_local_evals = budget_per_restart
            iters = 0
            max_iters = 50 # Safety cap on iterations per restart
            
            while local_budget_used < max_local_evals and iters < max_iters:
                if time.time() > deadline:
                    break
                
                # Sort simplex by fitness
                order = np.argsort(f_vals)
                simplex = simplex[order]
                f_vals = f_vals[order]
                
                # Check convergence
                f_range = f_vals[-1] - f_vals[0]
                if f_range < 1e-10:
                    break
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                alpha = 1.0
                x_ref = centroid + alpha * (centroid - simplex[-1])
                x_ref = np.clip(x_ref, -5, 5)
                f_ref = func(x_ref)
                eval_count += 1
                local_budget_used += 1
                
                if f_ref < f_vals[0]:
                    # Expansion
                    gamma = 2.0
                    x_exp = centroid + gamma * (centroid - simplex[-1])
                    x_exp = np.clip(x_exp, -5, 5)
                    f_exp = func(x_exp)
                    eval_count += 1
                    local_budget_used += 1
                    
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
                    beta = 0.5
                    x_con = centroid + beta * (simplex[-1] - centroid)
                    x_con = np.clip(x_con, -5, 5)
                    f_con = func(x_con)
                    eval_count += 1
                    local_budget_used += 1
                    
                    if f_con < f_vals[-1]:
                        simplex[-1] = x_con
                        f_vals[-1] = f_con
                    else:
                        # Shrink
                        # Shrink all points except the best towards the best
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f_vals[i] = func(simplex[i])
                            eval_count += 1
                            local_budget_used += 1
                            if eval_count >= total_budget or time.time() > deadline:
                                break
                        
                        if eval_count >= total_budget or time.time() > deadline:
                            break
                
                # Update global best
                current_local_best = np.min(f_vals)
                if current_local_best < best_f_global:
                    best_f_global = current_local_best
                    idx_min = np.argmin(f_vals)
                    best_x_global = simplex[idx_min].copy()
                
                iters += 1

        return best_x_global, best_f_global
