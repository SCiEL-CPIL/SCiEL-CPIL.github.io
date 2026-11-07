"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Nelder-Mead simplex method for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Deadline guard
        start_time = time.time()
        max_time = 300.0
        
        # Nelder-Mead parameters
        alpha = 1.0   # reflection
        gamma = 2.0   # expansion
        rho = 0.5     # contraction
        sigma = 0.5   # shrink
        
        # Initialize simplex
        # Start at origin (common heuristic for shifted functions if shift isn't huge)
        # Or random start? Origin is often safer for symmetric-ish functions, 
        # but Lunacek has two minima. However, without knowing the shift, 
        # a deterministic start is reproducible. 
        # Let's try a small random offset to avoid symmetry traps?
        # No, let's stick to a robust deterministic start for now, 
        # but maybe slightly randomized to break symmetry?
        # The prompt says "hidden seeded shift", so the function is not symmetric around 0.
        # Starting at 0 might be far from the minimum.
        # A random start in [-5, 5] might be better?
        # Let's use a fixed seed for reproducibility in this implementation, 
        # but the evaluator creates a new instance each time.
        # Let's start at 0. It's the center of the box.
        
        x0 = np.zeros(dim)
        f0 = func(x0)
        calls = 1
        
        # Create simplex
        # Simplex has dim+1 points
        # Initial step size
        delta = 0.5
        
        simplex = [x0.copy()]
        f_simplex = [f0]
        
        # Generate other points
        for i in range(dim):
            x_i = x0.copy()
            # Choose a direction. Standard is axis-aligned.
            # To be more robust, maybe use a random direction?
            # Axis-aligned is standard NM.
            if x0[i] + delta > 5.0:
                x_i[i] = x0[i] - delta
            else:
                x_i[i] = x0[i] + delta
            
            # Ensure within bounds
            x_i = np.clip(x_i, -5.0, 5.0)
            
            f_i = func(x_i)
            calls += 1
            
            simplex.append(x_i)
            f_simplex.append(f_i)
            
            if calls >= budget:
                break
            if time.time() - start_time > max_time:
                break

        # Main loop
        while calls < budget:
            if time.time() - start_time > max_time:
                break
            
            # Sort simplex by fitness
            order = np.argsort(f_simplex)
            simplex = [simplex[i] for i in order]
            f_simplex = [f_simplex[i] for i in order]
            
            # Check convergence
            # If the spread of the simplex is very small, or improvement is tiny
            # We can break early if stuck, but NM usually handles it.
            # Let's check if the best point hasn't improved much.
            
            # Calculate centroid of all points except the worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_r = centroid + alpha * (centroid - simplex[-1])
            x_r = np.clip(x_r, -5.0, 5.0)
            f_r = func(x_r)
            calls += 1
            
            if f_simplex[0] <= f_r < f_simplex[-2]:
                # Accept reflection
                simplex[-1] = x_r
                f_simplex[-1] = f_r
            elif f_r < f_simplex[0]:
                # Expansion
                x_e = centroid + gamma * (x_r - centroid)
                x_e = np.clip(x_e, -5.0, 5.0)
                f_e = func(x_e)
                calls += 1
                
                if f_e < f_r:
                    simplex[-1] = x_e
                    f_simplex[-1] = f_e
                else:
                    simplex[-1] = x_r
                    f_simplex[-1] = f_r
            else:
                # Contraction
                # Outside contraction if f_r < f_simplex[-1], otherwise inside
                if f_r < f_simplex[-1]:
                    # Outside contraction
                    x_c = centroid + rho * (x_r - centroid)
                else:
                    # Inside contraction
                    x_c = centroid - rho * (centroid - simplex[-1])
                
                x_c = np.clip(x_c, -5.0, 5.0)
                f_c = func(x_c)
                calls += 1
                
                if f_c < f_simplex[-1]:
                    simplex[-1] = x_c
                    f_simplex[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                        f_simplex[i] = func(simplex[i])
                        calls += 1
                        
                        if calls >= budget:
                            break
                        if time.time() - start_time > max_time:
                            break
                    
                    if calls >= budget or time.time() - start_time > max_time:
                        break

            # Check if we are stuck
            # If the worst point is not much worse than the best, we might be converged
            # But NM can stagnate.
            # A simple restart strategy could be added, but let's keep it simple first.
            
            # If the simplex is very small, we might break
            # Calculate diameter
            diam = np.max(np.linalg.norm(simplex[0] - x, axis=0) for x in simplex)
            if diam < 1e-6:
                break

        # Find the best point
        best_idx = np.argmin(f_simplex)
        best_x = simplex[best_idx]
        best_f = f_simplex[best_idx]
        
        return best_x, best_f

# EVOLVE-BLOCK-END
