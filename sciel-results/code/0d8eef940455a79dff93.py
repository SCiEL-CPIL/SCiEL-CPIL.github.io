"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Deterministic Halton exploration followed by SQP local search."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325.0  # Safety margin before 330s kill

        dim = self.dim
        total_budget = self.budget
        calls_made = 0

        global_best_x = np.zeros(dim)
        global_best_f = np.inf

        # Helper to generate Halton sequence points in [0, 1]^dim
        def halton(index, dim, bases):
            point = []
            for d in range(dim):
                f = 1.0
                i = index
                result = 0.0
                while i > 0:
                    f /= bases[d]
                    result += f * (i % bases[d])
                    i //= bases[d]
                point.append(result)
            return point

        # First few primes for bases
        bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

        # Phase 1: Global Exploration with Halton Sequence
        # We want to cover the space uniformly. 
        # Let's allocate 40% of the budget to exploration.
        explore_budget = int(total_budget * 0.4)
        
        # Start with a few points to get a sense of the scale
        # We map Halton points from [0,1]^d to [-5,5]^d
        
        for i in range(1, explore_budget + 1):
            if calls_made >= total_budget:
                break
            if time.time() > deadline - 1.0:
                break
            
            h_point = halton(i, dim, bases[:dim])
            x = np.array(h_point) * 10.0 - 5.0 # Map to [-5, 5]
            
            f_x = func(x)
            calls_made += 1
            
            if f_x < global_best_f:
                global_best_f = f_x
                global_best_x = x.copy()
            
            # Early termination if we get very close to 0? 
            # f_opt is in [-100, 100], so 0 is not necessarily optimal.
            # But if we find a very low value, we might switch to exploitation sooner.
            # However, Halton is deterministic and uniform, so sticking to the plan is safer.
            
        # Phase 2: Local Refinement with Sequential Quadratic Approximation (SQP)
        # We use the best point found so far as the center.
        # We will perform a local search by fitting a quadratic model to recent points.
        
        # To fit a quadratic model, we need at least dim*(dim+1)/2 + 1 points.
        # For dim=5, that's 15 + 1 = 16 points.
        # We have a history of points. Let's use the last 20-30 points for fitting.
        
        # However, the Halton points are spread out. The local structure near the best point
        # might not be well represented by the global Halton points.
        # So, we should generate new points locally around global_best_x.
        
        # Strategy:
        # 1. Start at global_best_x.
        # 2. Generate a local grid or random points (deterministic) around it.
        # 3. Fit a quadratic model to the best N points in the local neighborhood.
        # 4. Find the minimum of the quadratic model.
        # 5. Evaluate that point.
        # 6. Repeat.
        
        # Let's use a deterministic local exploration first to populate the local neighborhood.
        # We'll use a small grid around the best point.
        # Step size for local grid: start with 0.5, then 0.1, etc.
        
        local_center = global_best_x.copy()
        local_best_f = global_best_f
        local_best_x = global_best_x.copy()
        
        # We'll use the remaining budget for this phase.
        # We need to be careful with the number of points for the quadratic fit.
        # Let's collect a set of points near the current best.
        
        # To ensure we have enough points for a quadratic fit (16 points for dim=5),
        # we can evaluate a small set of points around the current best.
        
        # Let's define a local step size schedule.
        local_steps = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
        
        # For each step size, we will evaluate points in a cross pattern (2*dim points)
        # plus the center.
        
        # But we also want to fit a quadratic. 
        # A simpler robust approach: 
        # 1. Use the last K evaluations from the Halton phase that are close to the best.
        # 2. If not enough, generate new local points.
        
        # Let's just generate a local set of points.
        # We'll use a deterministic set of directions.
        # For each step size s:
        #   Evaluate x + s*e_i and x - s*e_i for i in 0..dim-1.
        #   If any is better, update center.
        #   Fit quadratic to the last 16 points (including center and neighbors).
        #   Compute gradient and Hessian from the fit.
        #   Take a step in the negative gradient direction or solve Hessian system.
        
        # This is getting complex. Let's simplify the SQP to a "Quadratic Model Step".
        
        # We will maintain a list of recent points and values.
        recent_points = []
        recent_values = []
        
        # Add the global best to recent points
        recent_points.append(global_best_x)
        recent_values.append(global_best_f)
        
        # We need 16 points. Let's generate 15 more around the best.
        # We'll use a small step size 0.1 for the initial local set.
        s_init = 0.5
        
        for d in range(dim):
            if calls_made >= total_budget: break
            if time.time() > deadline - 1.0: break
            
            for sign in [1, -1]:
                if calls_made >= total_budget: break
                if time.time() > deadline - 1.0: break
                
                x_new = local_best_x.copy()
                x_new[d] += sign * s_init
                x_new = np.clip(x_new, -5.0, 5.0)
                
                # Avoid duplicates
                if np.allclose(x_new, local_best_x):
                    continue
                    
                f_new = func(x_new)
                calls_made += 1
                
                recent_points.append(x_new)
                recent_values.append(f_new)
                
                if f_new < local_best_f:
                    local_best_f = f_new
                    local_best_x = x_new.copy()
                    global_best_f = f_new
                    global_best_x = x_new.copy()
                    
                if len(recent_points) >= 16:
                    break
            if len(recent_points) >= 16:
                break
        
        # Now we have 16 points. Fit a quadratic model.
        # f(x) = c + g^T (x - x0) + 0.5 (x - x0)^T H (x - x0)
        # Let x0 = local_best_x (or mean of points). Let's use local_best_x as center.
        
        def fit_quadratic(points, values, center):
            # points: (N, D), values: (N,), center: (D,)
            N, D = points.shape
            # We want to solve for c, g, H.
            # Number of parameters: 1 + D + D*(D+1)/2
            # For D=5, 1 + 5 + 15 = 21 parameters.
            # We have 16 points. Underdetermined.
            # We need more points or regularization.
            # Let's use the last 21 points if available, or generate more.
            
            # Actually, let's just do a simple gradient/Hessian approximation using finite differences
            # on the current best point, which is more robust with limited budget.
            # The "SQP" in the prompt might be interpreted as using a quadratic model.
            # Given the budget constraints and the need for robustness, a well-tuned
            # Nelder-Mead or Coordinate Descent with adaptive steps is often better.
            # However, the directive asks for SQP.
            
            # Let's try to fit with the available points using least squares.
            # We need at least 21 points for a full quadratic in 5D.
            # We only have 16.
            # Let's add more points to the recent set.
            
            return None # Placeholder
            
        # Since fitting a full quadratic is hard with limited points, let's use a 
        # "Quadratic Step" based on local finite differences which is effectively 
        # a simplified SQP.
        
        # We will estimate the gradient and Hessian at local_best_x using the 
        # points we just generated.
        
        # Gradient estimation:
        # g_i = (f(x + h e_i) - f(x - h e_i)) / (2h)
        # Hessian estimation:
        # H_ii = (f(x + h e_i) - 2f(x) + f(x - h e_i)) / h^2
        # H_ij = (f(x + h e_i + h e_j) - f(x + h e_i - h e_j) - f(x - h e_i + h e_j) + f(x - h e_i - h e_j)) / (4h^2)
        
        # This requires 2*dim for gradient, dim for H_ii, and 2*dim*(dim-1) for H_ij.
        # Total: 2*5 + 5 + 2*5*4 = 10 + 5 + 40 = 55 points.
        # We have budget for this.
        
        h = 0.01
        x_center = local_best_x
        f_center = local_best_f
        
        # Check if we have budget for 55 calls.
        # If not, fall back to coordinate descent.
        
        if total_budget - calls_made > 60:
            # Estimate Gradient
            grad = np.zeros(dim)
            hess = np.zeros((dim, dim))
            
            # Store f(x)
            f_x = f_center
            
            # Evaluate neighbors for gradient and H_ii
            f_plus = np.zeros(dim)
            f_minus = np.zeros(dim)
            
            for i in range(dim):
                if calls_made >= total_budget: break
                if time.time() > deadline - 1.0: break
                
                x_p = x_center.copy()
                x_p[i] += h
                x_p = np.clip(x_p, -5.0, 5.0)
                f_p = func(x_p)
                calls_made += 1
                f_plus[i] = f_p
                
                x_m = x_center.copy()
                x_m[i] -= h
                x_m = np.clip(x_m, -5.0, 5.0)
                f_m = func(x_m)
                calls_made += 1
                f_minus[i] = f_m
                
                grad[i] = (f_p - f_m) / (2 * h)
                hess[i, i] = (f_p - 2 * f_x + f_m) / (h * h)
                
            # Estimate H_ij for i != j
            for i in range(dim):
                for j in range(i + 1, dim):
                    if calls_made >= total_budget: break
                    if time.time() > deadline - 1.0: break
                    
                    x_pp = x_center.copy()
                    x_pp[i] += h
                    x_pp[j] += h
                    x_pp = np.clip(x_pp, -5.0, 5.0)
                    f_pp = func(x_pp)
                    calls_made += 1
                    
                    x_pm = x_center.copy()
                    x_pm[i] += h
                    x_pm[j] -= h
                    x_pm = np.clip(x_pm, -5.0, 5.0)
                    f_pm = func(x_pm)
                    calls_made += 1
                    
                    x_mp = x_center.copy()
                    x_mp[i] -= h
                    x_mp[j] += h
                    x_mp = np.clip(x_mp, -5.0, 5.0)
                    f_mp = func(x_mp)
                    calls_made += 1
                    
                    x_mm = x_center.copy()
                    x_mm[i] -= h
                    x_mm[j] -= h
                    x_mm = np.clip(x_mm, -5.0, 5.0)
                    f_mm = func(x_mm)
                    calls_made += 1
                    
                    hess_ij = (f_pp - f_pm - f_mp + f_mm) / (4 * h * h)
                    hess[i, j] = hess_ij
                    hess[j, i] = hess_ij
            
            # Solve H * delta = -grad
            # Regularize Hessian to ensure positive definiteness
            hess_reg = hess + 1e-6 * np.eye(dim)
            
            try:
                delta = np.linalg.solve(hess_reg, -grad)
            except np.linalg.LinAlgError:
                delta = -grad / (np.linalg.norm(grad) + 1e-6)
                
            # Limit step size
            max_step = 0.5
            norm_delta = np.linalg.norm(delta)
            if norm_delta > max_step:
                delta = delta / norm_delta * max_step
                
            x_new = x_center + delta
            x_new = np.clip(x_new, -5.0, 5.0)
            
            f_new = func(x_new)
            calls_made += 1
            
            if f_new < global_best_f:
                global_best_f = f_new
                global_best_x = x_new.copy()
                local_best_f = f_new
                local_best_x = x_new.copy()
            
            # Repeat this local refinement a few times
            for _ in range(5):
                if calls_made >= total_budget: break
                if time.time() > deadline - 1.0: break
                
                # Re-estimate gradient at new center
                h = 0.005
                x_center = local_best_x
                f_center = local_best_f
                grad = np.zeros(dim)
                
                for i in range(dim):
                    if calls_made >= total_budget: break
                    if time.time() > deadline - 1.0: break
                    
                    x_p = x_center.copy()
                    x_p[i] += h
                    x_p = np.clip(x_p, -5.0, 5.0)
                    f_p = func(x_p)
                    calls_made += 1
                    
                    x_m = x_center.copy()
                    x_m[i] -= h
                    x_m = np.clip(x_m, -5.0, 5.0)
                    f_m = func(x_m)
                    calls_made += 1
                    
                    grad[i] = (f_p - f_m) / (2 * h)
                    
                if np.linalg.norm(grad) < 1e-8:
                    break
                    
                # Simple gradient step
                step_size = 0.1
                x_new = x_center - step_size * grad / (np.linalg.norm(grad) + 1e-6)
                x_new = np.clip(x_new, -5.0, 5.0)
                
                f_new = func(x_new)
                calls_made += 1
                
                if f_new < local_best_f:
                    local_best_f = f_new
                    local_best_x = x_new.copy()
                    global_best_f = f_new
                    global_best_x = x_new.copy()
                else:
                    # Halve step size
                    step_size /= 2
                    x_new = x_center - step_size * grad / (np.linalg.norm(grad) + 1e-6)
                    x_new = np.clip(x_new, -5.0, 5.0)
                    f_new = func(x_new)
                    calls_made += 1
                    if f_new < local_best_f:
                        local_best_f = f_new
                        local_best_x = x_new.copy()
                        global_best_f = f_new
                        global_best_x = x_new.copy()

        # Final fallback: Coordinate descent with tiny steps if budget remains
        if calls_made < total_budget:
            final_steps = [0.001, 0.0005, 0.0001]
            for s in final_steps:
                for d in range(dim):
                    if calls_made >= total_budget: break
                    if time.time() > deadline - 1.0: break
                    
                    for sign in [1, -1]:
                        if calls_made >= total_budget: break
                        if time.time() > deadline - 1.0: break
                        
                        x_new = global_best_x.copy()
                        x_new[d] += sign * s
                        x_new = np.clip(x_new, -5.0, 5.0)
                        
                        if np.allclose(x_new, global_best_x):
                            continue
                        
                        f_new = func(x_new)
                        calls_made += 1
                        
                        if f_new < global_best_f:
                            global_best_f = f_new
                            global_best_x = x_new.copy()

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
