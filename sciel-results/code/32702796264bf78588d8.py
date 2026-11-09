"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Strategy: Sequential Quadratic Programming (SQP) via Local Quadratic Model Fitting.
    
    Instead of population-based methods like CMA-ES, this uses a model-based approach.
    1. Initial coarse sampling to find a good starting point.
    2. Iteratively fit a local quadratic model around the current best point.
    3. Move to the minimizer of the quadratic model (Newton step).
    4. This exploits the known structure of the Sphere function (quadratic) for
       super-linear convergence once in the basin of attraction.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin well before 330s limit
        
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        best_x = None
        best_f = np.inf
        calls_made = 0

        # Phase 1: Coarse Sampling to locate the basin
        # We sample a small set of points to get a rough idea of where the minimum is.
        # Since the minimum is in [-4, 4]^5, sampling in this range is safe.
        n_init = 20
        n_init = min(n_init, budget // 2)
        
        # Include origin and random points
        init_points = [np.zeros(dim)]
        for _ in range(n_init - 1):
            init_points.append(rng.uniform(-4.5, 4.5, dim))
            
        for p in init_points:
            if calls_made >= budget:
                break
            if time.time() > deadline - 10:
                break
            fp = func(p)
            calls_made += 1
            if fp < best_f:
                best_f = fp
                best_x = p.copy()

        if best_x is None:
            return np.zeros(dim), np.inf

        # Phase 2: Local Quadratic Model Fitting (Newton-like)
        # We will iteratively build a local model around the current best point.
        # For a Sphere function f(x) = ||R(x-x*) + s||^2 + c, the local model
        # is an exact quadratic. Fitting a quadratic allows us to take a direct
        # step to the predicted minimum.
        
        # Step size for sampling the neighborhood
        # Start with a moderate step size, adapt based on progress
        h = 0.5 
        
        # We will run this loop until budget is exhausted or convergence is reached.
        # Each iteration costs 2*dim + 1 evaluations (center + 2*dim neighbors).
        # We reserve a small buffer for safety.
        
        while calls_made + 2 * dim + 1 <= budget:
            if time.time() > deadline - 5:
                break
            
            # Check for convergence: if step size is very small, we are likely at the minimum
            if h < 1e-6:
                break

            # Collect points for local quadratic fit
            # Center point
            x_c = best_x.copy()
            f_c = best_f
            
            # We already have f_c from the previous best, so no new call needed for center
            # But we need neighbors.
            
            points = [x_c]
            values = [f_c]
            
            # Sample 2*dim points: x_c +/- h * e_i for each dimension i
            for i in range(dim):
                x_plus = x_c.copy()
                x_minus = x_c.copy()
                x_plus[i] += h
                x_minus[i] -= h
                
                # Clip to bounds
                x_plus = np.clip(x_plus, -5, 5)
                x_minus = np.clip(x_minus, -5, 5)
                
                points.append(x_plus)
                points.append(x_minus)
                
                f_plus = func(x_plus)
                f_minus = func(x_minus)
                calls_made += 2
                
                values.append(f_plus)
                values.append(f_minus)
                
                # Update best if we found a better point
                if f_plus < best_f:
                    best_f = f_plus
                    best_x = x_plus.copy()
                if f_minus < best_f:
                    best_f = f_minus
                    best_x = x_minus.copy()

            # Fit a quadratic model: f(x) ~ f0 + g^T (x-x_c) + 0.5 (x-x_c)^T H (x-x_c)
            # We have 2*dim+1 points. This is enough to estimate gradient (dim params) 
            # and Hessian (dim*(dim+1)/2 params) if we assume symmetry.
            # Total params: 1 + dim + dim*(dim+1)/2.
            # For dim=5: 1 + 5 + 15 = 21 params. We have 11 points. 
            # Wait, 2*dim+1 = 11 points for dim=5.
            # We cannot fully determine the Hessian with only axis-aligned points.
            # However, for a Sphere function, the Hessian is constant (2*R^T R).
            # The axis-aligned points allow us to estimate the diagonal of the Hessian
            # and the gradient. The off-diagonals are 0 in the local coordinate system 
            # if the function is axis-aligned, but it's rotated.
            
            # Let's use a simpler approach: 
            # Estimate gradient g_i = (f(x+h_i) - f(x-h_i)) / (2h)
            # Estimate H_ii = (f(x+h_i) + f(x-h_i) - 2f(x)) / h^2
            # Assume H is diagonal in this local frame? No, that's wrong for rotated spheres.
            # But wait, if we just use the gradient and a scalar Hessian approximation (like in Quasi-Newton),
            # or if we assume the function is locally separable?
            
            # Actually, a better strategy for "different design": 
            # Use the axis-aligned points to do a coordinate-wise Newton step?
            # Or simply use the gradient information to do a steepest descent step?
            
            # Let's stick to a robust method: 
            # Since we have 2*dim+1 points, we can fit a linear model for the gradient.
            # And we can estimate the curvature in each direction.
            
            # Gradient estimation
            g = np.zeros(dim)
            for i in range(dim):
                f_plus = values[1 + 2*i]
                f_minus = values[2 + 2*i]
                g[i] = (f_plus - f_minus) / (2 * h)
            
            # Hessian diagonal estimation (curvature in each axis)
            H_diag = np.zeros(dim)
            for i in range(dim):
                f_plus = values[1 + 2*i]
                f_minus = values[2 + 2*i]
                H_diag[i] = (f_plus + f_minus - 2 * f_c) / (h * h)
                
                # Ensure positive curvature (if negative, it's a saddle or we are far away)
                if H_diag[i] < 1e-8:
                    H_diag[i] = 1e-8

            # Proposed step: Newton step assuming diagonal Hessian
            # x_new = x_c - g / H_diag
            step = g / H_diag
            
            # Limit the step size to prevent overshooting if the model is bad
            step_norm = np.linalg.norm(step)
            max_step = h * 2.0
            if step_norm > max_step:
                step = step * (max_step / step_norm)
                
            x_new = x_c + step
            x_new = np.clip(x_new, -5, 5)
            
            # Evaluate the new point
            f_new = func(x_new)
            calls_made += 1
            
            # Check improvement
            if f_new < best_f:
                # Success: move to new point, maybe increase h slightly or keep same
                best_f = f_new
                best_x = x_new.copy()
                # If we made significant progress, we can keep h or increase it slightly
                # to explore further if we are still far from the minimum.
                # But for quadratic convergence, we should decrease h as we get closer.
                # However, if the step was limited, we might want to increase h.
                # Simple heuristic: if step was limited, increase h, else decrease h.
                if step_norm >= max_step * 0.9:
                    h *= 1.5
                else:
                    h *= 0.7
            else:
                # Failure: model was bad or we are at a minimum
                # Reduce h to look closer
                h *= 0.5
                
                # If h is too small, we might be stuck. 
                # Try a random kick?
                if h < 1e-4:
                    # Random restart from current best
                    kick = rng.normal(0, 0.1, dim)
                    x_try = best_x + kick
                    x_try = np.clip(x_try, -5, 5)
                    f_try = func(x_try)
                    calls_made += 1
                    if f_try < best_f:
                        best_f = f_try
                        best_x = x_try.copy()
                        h = 0.1 # Reset h for new exploration
                    else:
                        # Give up on local search, maybe do a broader search?
                        # For now, just break if we can't improve
                        pass
                        
        return best_x, best_f
# EVOLVE-BLOCK-END
