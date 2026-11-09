"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-start Nelder-Mead Simplex optimizer for Rastrigin-like functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        calls_made = 0
        best_f = np.inf
        best_x = None

        # Nelder-Mead parameters
        # Standard: 2*alpha+1=3 (reflection), gamma=2 (expansion), rho=0.5 (contraction), sigma=0.5 (shrink)
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        # Initial simplex size
        initial_delta = 2.0 
        
        # Number of multi-starts
        # We have 500 budget. NM takes about dim*(dim+1)/2 ~ 15-30 calls to converge or stall.
        # Let's allow up to 20 starts.
        max_starts = 20
        
        # We need to reserve some calls for the best start to refine if possible, 
        # but NM is greedy. We'll just run as many starts as budget allows.
        
        for start_idx in range(max_starts):
            # Check if we have enough budget for a new simplex initialization (dim+1 points)
            if calls_made + (dim + 1) > budget:
                break
            
            # Initialize new simplex
            # Start point: random in [-4, 4] to stay within bounds easily, or [-5,5]
            # Rastrigin global min is at 0. Random start in [-5,5] is fine.
            center = rng.uniform(-4.0, 4.0, dim)
            
            simplex = []
            simplex_f = []
            
            # First point
            x0 = center.copy()
            x0 = np.clip(x0, -5.0, 5.0)
            f0 = func(x0)
            calls_made += 1
            simplex.append(x0)
            simplex_f.append(f0)
            
            # Other points: perturb each dimension
            for i in range(dim):
                xi = center.copy()
                # Perturb dimension i
                if i == 0:
                    xi[0] += initial_delta
                else:
                    xi[i] += initial_delta
                xi = np.clip(xi, -5.0, 5.0)
                fi = func(xi)
                calls_made += 1
                simplex.append(xi)
                simplex_f.append(fi)
                
            # Sort simplex by fitness
            indices = np.argsort(simplex_f)
            simplex = [simplex[i] for i in indices]
            simplex_f = [simplex_f[i] for i in indices]
            
            # Update global best
            if simplex_f[0] < best_f:
                best_f = simplex_f[0]
                best_x = simplex[0].copy()
                
            # Nelder-Mead loop
            # Max iterations per start to avoid getting stuck in slow convergence
            # or infinite loops due to numerical issues.
            max_iters = 100 
            
            for _ in range(max_iters):
                if calls_made >= budget:
                    break
                    
                # Calculate centroid of all points except the worst
                # simplex[0] is best, simplex[-1] is worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                worst_point = simplex[-1]
                worst_f = simplex_f[-1]
                
                x_r = centroid + alpha * (centroid - worst_point)
                x_r = np.clip(x_r, -5.0, 5.0)
                f_r = func(x_r)
                calls_made += 1
                
                if f_r < simplex_f[0]:
                    # Expansion
                    x_e = centroid + gamma * (centroid - worst_point)
                    x_e = np.clip(x_e, -5.0, 5.0)
                    f_e = func(x_e)
                    calls_made += 1
                    
                    if f_e < f_r:
                        new_point = x_e
                        new_f = f_e
                    else:
                        new_point = x_r
                        new_f = f_r
                elif f_r < simplex_f[-2]:
                    # Reflection is good enough
                    new_point = x_r
                    new_f = f_r
                else:
                    # Contraction
                    if f_r < worst_f:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (centroid - worst_point)
                    
                    x_c = np.clip(x_c, -5.0, 5.0)
                    f_c = func(x_c)
                    calls_made += 1
                    
                    if f_c < f_r:
                        new_point = x_c
                        new_f = f_c
                    else:
                        # Shrink
                        # Move all points (except best) towards the best point
                        best_point = simplex[0]
                        best_f_local = simplex_f[0]
                        
                        new_simplex = [best_point]
                        new_simplex_f = [best_f_local]
                        
                        for i in range(1, len(simplex)):
                            x_s = best_point + sigma * (simplex[i] - best_point)
                            x_s = np.clip(x_s, -5.0, 5.0)
                            f_s = func(x_s)
                            calls_made += 1
                            new_simplex.append(x_s)
                            new_simplex_f.append(f_s)
                            
                            if calls_made >= budget:
                                break
                            
                        if calls_made >= budget:
                            break
                        
                        # Sort new simplex
                        indices = np.argsort(new_simplex_f)
                        simplex = [new_simplex[i] for i in indices]
                        simplex_f = [new_simplex_f[i] for i in indices]
                        
                        if simplex_f[0] < best_f:
                            best_f = simplex_f[0]
                            best_x = simplex[0].copy()
                        
                        # Check for convergence or stop if no improvement
                        if np.std(simplex_f) < 1e-8:
                            break
                        continue
                    
                # Replace worst point
                simplex[-1] = new_point
                simplex_f[-1] = new_f
                
                # Sort
                indices = np.argsort(simplex_f)
                simplex = [simplex[i] for i in indices]
                simplex_f = [simplex_f[i] for i in indices]
                
                # Update global best
                if simplex_f[0] < best_f:
                    best_f = simplex_f[0]
                    best_x = simplex[0].copy()
                    
                # Convergence check
                if np.std(simplex_f) < 1e-8:
                    break

            # If we still have budget, we could try to refine the best point from this start
            # but NM usually converges. We just move to the next start.
            
        return best_x, best_f
# EVOLVE-BLOCK-END
