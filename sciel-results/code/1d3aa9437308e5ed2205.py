"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead Simplex Optimizer for BBOB-style f9 (Rosenbrock rotated).
    
    Strategy:
    - Uses the Nelder-Mead method, which is highly effective for low-dimensional
      continuous optimization problems without gradient information.
    - Adapts the simplex shape to navigate the narrow valley of the Rosenbrock function.
    - Includes robustness against the box constraints by clipping.
    - Manages budget and time strictly.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 320  # Safety margin
        
        dim = self.dim
        budget = self.budget
        
        # Track best solution
        best_x = None
        best_f = float('inf')
        
        calls_made = 0
        rng = np.random.default_rng(42)
        
        # Nelder-Mead parameters
        alpha = 1.0    # Reflection
        gamma = 2.0    # Expansion
        rho = 0.5      # Contraction
        sigma = 0.5    # Shrinkage
        
        # Initialize simplex
        # Start at 0, which is often a good heuristic start for BBOB
        # But we can also try a random start if 0 is bad, or just stick to 0 for simplicity 
        # given the low budget. Let's start with 0 and a small perturbation simplex.
        x0 = np.zeros(dim)
        f0 = func(x0)
        calls_made += 1
        best_x = x0.copy()
        best_f = f0
        
        # Create initial simplex
        simplex = [x0]
        simplex_f = [f0]
        
        # Standard initial simplex: perturb each dimension
        for i in range(dim):
            xi = x0.copy()
            # Use a small step, e.g., 0.05 or 0.1
            step = 0.05
            if xi[i] + step > 5:
                step = -step
            xi[i] += step
            xi = np.clip(xi, -5, 5)
            fi = func(xi)
            calls_made += 1
            simplex.append(xi)
            simplex_f.append(fi)
            if fi < best_f:
                best_f = fi
                best_x = xi.copy()
        
        # Sort simplex by function value
        indices = np.argsort(simplex_f)
        simplex = [simplex[i] for i in indices]
        simplex_f = [simplex_f[i] for i in indices]
        
        # Main Loop
        while calls_made < budget:
            if time.time() > deadline:
                break
            
            # Check for convergence (simplex size)
            centroid = np.mean(simplex, axis=0)
            centroid_f = func(centroid) if calls_made < budget else None # Optional, but costs a call
            # Let's not evaluate centroid every time to save budget, just check spread
            spread = np.max(simplex, axis=0) - np.min(simplex, axis=0)
            if np.max(spread) < 1e-6:
                break
                
            # 1. Reflection
            x_worst = simplex[-1]
            f_worst = simplex_f[-1]
            centroid_wo = np.mean(simplex[:-1], axis=0)
            x_r = centroid_wo + alpha * (centroid_wo - x_worst)
            x_r = np.clip(x_r, -5, 5)
            f_r = func(x_r)
            calls_made += 1
            if f_r < best_f:
                best_f = f_r
                best_x = x_r.copy()
            
            if f_r < simplex_f[0]:
                # 2. Expansion
                x_e = centroid_wo + gamma * (centroid_wo - x_worst)
                x_e = np.clip(x_e, -5, 5)
                f_e = func(x_e)
                calls_made += 1
                if f_e < best_f:
                    best_f = f_e
                    best_x = x_e.copy()
                
                if f_e < f_r:
                    simplex[-1] = x_e
                    simplex_f[-1] = f_e
                else:
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
            elif f_r < simplex_f[-2]:
                # Simple Reflection
                simplex[-1] = x_r
                simplex_f[-1] = f_r
            else:
                # 3. Contraction
                if f_r < f_worst:
                    # Outside contraction
                    x_c = centroid_wo + rho * (x_r - centroid_wo)
                    f_c = f_r
                else:
                    # Inside contraction
                    x_c = centroid_wo - rho * (x_r - centroid_wo)
                    f_c = f_worst
                    
                x_c = np.clip(x_c, -5, 5)
                f_c = func(x_c)
                calls_made += 1
                if f_c < best_f:
                    best_f = f_c
                    best_x = x_c.copy()
                
                if f_c < f_worst:
                    simplex[-1] = x_c
                    simplex_f[-1] = f_c
                else:
                    # 4. Shrinkage
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        f_i = func(simplex[i])
                        calls_made += 1
                        if f_i < best_f:
                            best_f = f_i
                            best_x = simplex[i].copy()
                        simplex_f[i] = f_i
                    
            # Re-sort simplex
            indices = np.argsort(simplex_f)
            simplex = [simplex[i] for i in indices]
            simplex_f = [simplex_f[i] for i in indices]
            
            # If we are stuck (no improvement in best_f for a while), we could restart
            # But with Nelder-Mead, it usually escapes local minima in low dims if initialized well.
            # Given the tight budget, we stick to the main loop.
            
            # Convergence check based on function values
            if abs(simplex_f[0] - simplex_f[-1]) < 1e-10:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
