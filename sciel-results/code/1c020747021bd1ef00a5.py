"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Hybrid Optimizer: Initial Random Sampling + Multiple Starts Nelder-Mead.
    
    This approach is chosen because:
    1. Schaffers F7 is multimodal and ill-conditioned.
    2. Nelder-Mead is very efficient in low dimensions (5D).
    3. Multiple starts help escape local minima.
    4. Initial sampling provides a baseline score early on.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0
        rng = np.random.default_rng(42)
        
        # 1. Initial Sampling Phase
        # Use a significant portion of the budget for initial exploration to find good basins.
        # 5D problem, let's use 50 initial points.
        n_initial = min(50, self.budget // 2)
        if n_initial < 10:
            n_initial = self.budget
        
        x_init = rng.uniform(-5, 5, (n_initial, self.dim))
        f_init = np.array([func(x) for x in x_init])
        
        # Track best so far
        best_idx = np.argmin(f_init)
        best_x = x_init[best_idx].copy()
        best_f = f_init[best_idx]
        
        calls_used = n_initial
        
        # Sort initial points by fitness to prioritize good starting points for NM
        sorted_indices = np.argsort(f_init)
        
        # 2. Local Refinement Phase using Nelder-Mead
        # We will run Nelder-Mead from the best initial points, using the remaining budget.
        
        # Parameters for Nelder-Mead
        alpha = 1.0  # Reflection
        gamma = 2.0  # Expansion
        rho = 0.5    # Contraction
        sigma = 0.5  # Shrinkage
        
        # Simplex size
        # If the best point is not at the origin, we might want to scale the simplex size
        # based on the distance to the center or just use a fixed reasonable size.
        # A size of 0.5 is reasonable for [-5,5]^5.
        simplex_size = 0.5
        
        def nelder_mead(func, x0, remaining_budget, alpha, gamma, rho, sigma, simplex_size):
            # Initialize simplex
            n = len(x0)
            simplex = np.zeros((n + 1, n))
            simplex[0] = x0
            for i in range(n):
                simplex[i + 1] = x0.copy()
                simplex[i + 1][i] += simplex_size
            
            # Evaluate simplex
            f_simplex = np.array([func(x) for x in simplex])
            calls = n + 1
            
            # Sort simplex
            sort_idx = np.argsort(f_simplex)
            simplex = simplex[sort_idx]
            f_simplex = f_simplex[sort_idx]
            
            best_f_local = f_simplex[0]
            best_x_local = simplex[0].copy()
            
            max_iter = remaining_budget // 2 # Conservative estimate
            iter_count = 0
            
            while calls < remaining_budget and iter_count < max_iter:
                if time.time() > deadline:
                    break
                    
                # Check for convergence (simplex too small)
                if np.max(f_simplex) - np.min(f_simplex) < 1e-10:
                    break
                
                # Worst point
                worst_idx = n
                x_worst = simplex[worst_idx]
                f_worst = f_simplex[worst_idx]
                
                # Centroid of best n points
                x_centroid = np.mean(simplex[:n], axis=0)
                
                # Reflection
                x_reflect = x_centroid + alpha * (x_centroid - x_worst)
                x_reflect = np.clip(x_reflect, -5, 5)
                f_reflect = func(x_reflect)
                calls += 1
                
                if f_reflect < f_simplex[0]:
                    # Expansion
                    x_expand = x_centroid + gamma * (x_reflect - x_centroid)
                    x_expand = np.clip(x_expand, -5, 5)
                    f_expand = func(x_expand)
                    calls += 1
                    
                    if f_expand < f_reflect:
                        simplex[worst_idx] = x_expand
                        f_simplex[worst_idx] = f_expand
                    else:
                        simplex[worst_idx] = x_reflect
                        f_simplex[worst_idx] = f_reflect
                elif f_reflect < f_simplex[n-1]:
                    # Accept reflection
                    simplex[worst_idx] = x_reflect
                    f_simplex[worst_idx] = f_reflect
                else:
                    # Contraction
                    if f_reflect < f_worst:
                        # Outside contraction
                        x_contract = x_centroid + rho * (x_reflect - x_centroid)
                    else:
                        # Inside contraction
                        x_contract = x_centroid - rho * (x_worst - x_centroid)
                        
                    x_contract = np.clip(x_contract, -5, 5)
                    f_contract = func(x_contract)
                    calls += 1
                    
                    if f_contract < min(f_reflect, f_worst):
                        simplex[worst_idx] = x_contract
                        f_simplex[worst_idx] = f_contract
                    else:
                        # Shrinkage
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            f_simplex[i] = func(simplex[i])
                            calls += 1
                        if calls >= remaining_budget:
                            break
                        
                # Re-sort
                sort_idx = np.argsort(f_simplex)
                simplex = simplex[sort_idx]
                f_simplex = f_simplex[sort_idx]
                
                if f_simplex[0] < best_f_local:
                    best_f_local = f_simplex[0]
                    best_x_local = simplex[0].copy()
                
                iter_count += 1
            
            return best_x_local, best_f_local, calls

        # Run Nelder-Mead on the best initial points
        # We have remaining budget = self.budget - calls_used
        remaining_budget = self.budget - calls_used
        
        # Try the top 5 initial points
        for idx in sorted_indices[:5]:
            if calls_used >= self.budget:
                break
            if time.time() > deadline:
                break
                
            x_start = x_init[idx].copy()
            f_start = f_init[idx]
            
            # If this start is not better than current best, skip? 
            # Actually, NM might find a better local min even if start is worse, 
            # but usually we prioritize the best starts.
            
            # Allocate a portion of the remaining budget to this NM run
            # Let's give each NM run a share of the remaining budget
            # But if one finishes early, we can use the rest for the next.
            # To keep it simple, we just pass the *total* remaining budget to the function,
            # and it will stop when its internal budget is exhausted or it converges.
            # However, we need to track global calls.
            
            # Let's modify NM to return calls used, and we subtract from global.
            # But the current NM implementation uses a local `calls` counter starting from 0.
            # We need to pass the *max allowed* calls.
            
            max_calls_for_nm = remaining_budget
            x_best_nm, f_best_nm, calls_used_nm = nelder_mead(func, x_start, max_calls_for_nm, alpha, gamma, rho, sigma, simplex_size)
            
            calls_used += calls_used_nm
            remaining_budget -= calls_used_nm
            
            if f_best_nm < best_f:
                best_f = f_best_nm
                best_x = x_best_nm.copy()
            
            # If NM used all budget, stop
            if remaining_budget <= 0:
                break

        return best_x, best_f

# EVOLVE-BLOCK-END
