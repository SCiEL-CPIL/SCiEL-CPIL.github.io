"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead simplex optimizer for the Bent Cigar function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 330s limit
        
        dim = self.dim
        budget = self.budget
        
        # Nelder-Mead parameters
        # Initial simplex: center at 0, step size 0.5
        # This is a heuristic choice. For BBOB f12 (Bent Cigar), the optimum is often shifted.
        # We will do a brief random search first to get a better starting point if possible,
        # but Nelder-Mead is local. 
        # Actually, the task says "minimizing func on [-5, 5]^5". 
        # The current CMA-ES starts at 0. 
        # Let's start with a small random search to find a promising basin, then refine with NM.
        
        # Phase 1: Random search to find a good starting point (use 20% of budget)
        n_random = max(10, budget // 5)
        if n_random >= budget:
            n_random = budget - 1
            
        best_x = np.zeros(dim)
        best_f = float('inf')
        
        rng = np.random.default_rng(42) # Fixed seed for reproducibility in local testing
        
        for i in range(n_random):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()
                
        calls_made = n_random
        
        # Phase 2: Nelder-Mead refinement
        # Initialize simplex around best_x
        # Simplex has dim+1 vertices
        # v[0] = best_x
        # v[i] = best_x + step * e_i
        
        step = 0.5
        simplex = np.zeros((dim + 1, dim))
        simplex[0] = best_x
        for i in range(dim):
            simplex[i + 1] = best_x
            simplex[i + 1, i] += step
            
        # Evaluate initial simplex
        f_vals = np.zeros(dim + 1)
        f_vals[0] = best_f
        for i in range(dim):
            if calls_made + 1 > budget or time.time() > deadline:
                break
            f_vals[i + 1] = func(simplex[i + 1])
            calls_made += 1
            if f_vals[i + 1] < best_f:
                best_f = f_vals[i + 1]
                best_x = simplex[i + 1].copy()
                
        # Nelder-Mead loop
        alpha = 1.0   # Reflection
        gamma = 2.0   # Expansion
        rho = 0.5     # Contraction
        sigma = 0.5   # Shrink
        
        while calls_made < budget and time.time() < deadline:
            # Sort simplex by function value
            order = np.argsort(f_vals)
            simplex = simplex[order]
            f_vals = f_vals[order]
            
            # Check for convergence (simplex size)
            centroid = np.mean(simplex[:-1], axis=0)
            spread = np.max(simplex, axis=0) - np.min(simplex, axis=0)
            if np.all(spread < 1e-6) or f_vals[0] - f_vals[-1] < 1e-12:
                break
                
            # Centroid of all but worst point
            centroid = np.mean(simplex[:-1], axis=0)
            worst = simplex[-1]
            
            # 1. Reflection
            x_r = centroid + alpha * (centroid - worst)
            x_r = np.clip(x_r, -5, 5)
            f_r = func(x_r)
            calls_made += 1
            
            if f_r < f_vals[0]:
                # 2. Expansion
                x_e = centroid + gamma * (x_r - centroid)
                x_e = np.clip(x_e, -5, 5)
                f_e = func(x_e)
                calls_made += 1
                
                if f_e < f_r:
                    simplex[-1] = x_e
                    f_vals[-1] = f_e
                else:
                    simplex[-1] = x_r
                    f_vals[-1] = f_r
            elif f_r < f_vals[-2]:
                # Accept reflection
                simplex[-1] = x_r
                f_vals[-1] = f_r
            else:
                # 3. Contraction
                # Outer contraction (if reflection is better than worst but not good enough)
                if f_r < f_vals[-1]:
                    x_c = centroid + rho * (x_r - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = func(x_c)
                    calls_made += 1
                    
                    if f_c <= f_r:
                        simplex[-1] = x_c
                        f_vals[-1] = f_c
                    else:
                        # 4. Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            f_vals[i] = func(simplex[i])
                            calls_made += 1
                else:
                    # Inner contraction
                    x_c = centroid - rho * (centroid - worst)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = func(x_c)
                    calls_made += 1
                    
                    if f_c < f_vals[-1]:
                        simplex[-1] = x_c
                        f_vals[-1] = f_c
                    else:
                        # 4. Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            f_vals[i] = func(simplex[i])
                            calls_made += 1
                            
            # Update best
            min_idx = np.argmin(f_vals)
            if f_vals[min_idx] < best_f:
                best_f = f_vals[min_idx]
                best_x = simplex[min_idx].copy()

        return best_x, best_f

# EVOLVE-BLOCK-END
