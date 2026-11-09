"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Nelder-Mead with adaptive step size and coarse-to-fine initialization for BBOB Sphere."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # Safety margin before 300s limit

        # Fixed seed for reproducibility
        rng = np.random.default_rng(42)

        # 1. Coarse-to-fine Initialization
        # The minimum is in [-4, 4]^5.
        # Start with a coarse grid to find a promising region, then refine.
        # Budget allocation: ~100 calls for initialization, rest for optimization.
        
        # Coarse grid: 3 points per dimension -> 3^5 = 243 points.
        # This might be too many if budget is tight, but 500 is decent.
        # Let's use 2 points per dimension first? No, 3 is better for coverage.
        # However, 243 calls is almost half the budget. 
        # Let's try a smarter initialization: Random sampling + local refinement.
        # Actually, for Sphere, the global minimum is unique. 
        # A coarse grid of 3^5=243 points covers [-5,5] with step 5.
        # Min value in grid will be close to true min if grid is fine enough.
        # Distance from grid point to true min is at most sqrt(5)*(2.5/sqrt(12)) ~ 1.8 in worst case?
        # Step size in grid: 10/2 = 5. Max distance to any point is sqrt(5)*(5/2) ~ 5.5.
        # That's too coarse.
        
        # Alternative: Use a few random starts and pick the best, then start Nelder-Mead.
        # Or: Use a hierarchical approach.
        
        # Let's stick to the directive: "Model the search as a sequential decision process where the step size is adaptively scaled by the inverse of the local gradient magnitude".
        # This suggests a local search method, not global.
        # So, good initialization is key.
        
        # Strategy:
        # 1. Evaluate center (0,0,0,0,0).
        # 2. Evaluate 10 random points in [-4,4]^5.
        # 3. Pick the best one as starting point for Nelder-Mead.
        # 4. Run Nelder-Mead with adaptive step size based on gradient estimate.
        
        x_center = np.zeros(dim)
        fx_center = func(x_center)
        calls = 1
        
        best_x = x_center.copy()
        best_f = fx_center
        
        # Random initialization
        for _ in range(10):
            x_rand = rng.uniform(-4, 4, dim)
            fx_rand = func(x_rand)
            calls += 1
            if fx_rand < best_f:
                best_f = fx_rand
                best_x = x_rand.copy()
        
        # Now start Nelder-Mead from best_x
        # Nelder-Mead requires dim+1 points.
        # We have dim+1 = 6 points.
        # We already used 11 calls. We have 500 - 11 = 489 calls left.
        
        # Initialize simplex
        # Start with best_x
        simplex = [best_x.copy()]
        f_values = [best_f]
        
        # Generate other vertices of the simplex
        # Use a small initial step size for the simplex to ensure local convergence
        # Since we don't know the scale, we start with a moderate step.
        # The adaptive step size will adjust this.
        
        # Estimate gradient magnitude to set initial step size
        # Finite difference gradient estimate at best_x
        eps = 1e-4
        grad = np.zeros(dim)
        for i in range(dim):
            x_plus = best_x.copy()
            x_plus[i] += eps
            fx_plus = func(x_plus)
            calls += 1
            
            x_minus = best_x.copy()
            x_minus[i] -= eps
            fx_minus = func(x_minus)
            calls += 1
            
            grad[i] = (fx_plus - fx_minus) / (2 * eps)
        
        grad_mag = np.linalg.norm(grad)
        
        # If gradient is zero, we might be at the optimum or in a flat region.
        # For Sphere, gradient is 2*(x - x_opt).
        # If grad_mag is small, we are close.
        
        # Set initial simplex step size based on gradient magnitude
        # If gradient is large, we can take larger steps? No, Nelder-Mead step size is usually fixed or adaptive based on function values.
        # The directive says: "step size is adaptively scaled by the inverse of the local gradient magnitude".
        # This implies: if gradient is small (flat), take larger steps to find the slope? Or smaller steps to be precise?
        # "large exploratory jumps in flat regions and tiny precision steps near the optimum"
        # So: step_size = base_size / (1 + grad_mag) ? Or base_size * (1 / (1 + grad_mag))?
        # If grad_mag is large (steep), step_size should be small (precision).
        # If grad_mag is small (flat), step_size should be large (exploration).
        # This matches: step_size ~ 1 / grad_mag.
        
        # However, Nelder-Mead doesn't explicitly use gradient. It uses function values.
        # We can modify the reflection/expansion/contraction factors or the initial simplex size.
        # Let's implement a custom local search that uses gradient magnitude to scale steps.
        # This is more aligned with the directive than standard Nelder-Mead.
        
        # Revised Strategy:
        # 1. Coarse random search to find a good starting point.
        # 2. Local search with adaptive step size based on finite-difference gradient magnitude.
        #    - Estimate gradient at current point.
        #    - Compute step direction = -gradient (steepest descent).
        #    - Compute step size = alpha / (1 + ||grad||).
        #    - Take step, evaluate.
        #    - If improvement, accept and repeat.
        #    - If no improvement, reduce step size (or random restart locally).
        
        # This is essentially a gradient descent with adaptive step size.
        # For Sphere, this is very effective.
        
        x = best_x.copy()
        f = best_f
        
        # Parameters for adaptive gradient descent
        alpha = 10.0  # Base step size
        min_step = 1e-8
        max_iter = budget - calls
        
        for i in range(max_iter):
            if time.time() > deadline:
                break
            
            # Estimate gradient
            eps = 1e-5
            grad = np.zeros(dim)
            for j in range(dim):
                x_p = x.copy()
                x_p[j] += eps
                f_p = func(x_p)
                calls += 1
                
                x_m = x.copy()
                x_m[j] -= eps
                f_m = func(x_m)
                calls += 1
                
                grad[j] = (f_p - f_m) / (2 * eps)
                
                # Early exit if budget exceeded
                if calls >= budget:
                    break
            
            if calls >= budget:
                break
            
            grad_mag = np.linalg.norm(grad)
            
            # Adaptive step size
            # If gradient is large, step is small. If gradient is small, step is large.
            # step_size = alpha / (1 + grad_mag)
            step_size = alpha / (1.0 + grad_mag)
            
            # Ensure step size is within bounds
            step_size = np.clip(step_size, min_step, 10.0)
            
            # Proposed step
            x_new = x - step_size * grad
            x_new = np.clip(x_new, -5, 5)
            
            f_new = func(x_new)
            calls += 1
            
            if f_new < f:
                x = x_new
                f = f_new
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
            else:
                # No improvement. Try a random perturbation to escape local flatness?
                # Or reduce alpha?
                # Since it's Sphere, we shouldn't get stuck in local minima.
                # If no improvement, it might be due to numerical noise or we are very close.
                # Let's try a random step in a random direction with small magnitude.
                dir = rng.standard_normal(dim)
                dir /= np.linalg.norm(dir)
                x_try = x + 0.1 * dir
                x_try = np.clip(x_try, -5, 5)
                f_try = func(x_try)
                calls += 1
                if calls >= budget:
                    break
                if f_try < f:
                    x = x_try
                    f = f_try
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                else:
                    # Stagnation. Reduce alpha slightly.
                    alpha *= 0.9
                    if alpha < min_step:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
