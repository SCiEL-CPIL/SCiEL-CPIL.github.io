import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        calls = 0
        best_x = None
        best_f = np.inf
        dim = self.dim

        def time_ok():
            return time.time() < deadline

        def wrapped_func(x):
            nonlocal calls, best_x, best_f
            if not time_ok():
                raise TimeoutError("Time limit exceeded")
            calls += 1
            if calls > self.budget:
                raise Exception("Budget exceeded")
            
            val = float(func(x))
            
            if val < best_f:
                best_f = val
                best_x = np.array(x, dtype=float).copy()
                
            return val

        # DFTR Strategy:
        # 1. Global Phase: Coarse grid/random sampling to find a decent basin.
        # 2. Local Phase: Derivative-Free Trust Region (DFTR) using quadratic surrogates.
        
        # Phase 1: Global Exploration
        # Use ~20% of budget for global search.
        global_budget = int(0.2 * self.budget)
        
        rng_global = np.random.default_rng(42)
        
        # Start with a few random points and the center
        candidates = [np.zeros(dim)]
        for _ in range(global_budget - 1):
            candidates.append(rng_global.uniform(-4, 4, size=dim))
            
        # Evaluate candidates
        # We need to be careful with the budget here.
        # If we run out of time, we stop.
        
        # Sort by function value
        evaluated_candidates = []
        for x in candidates:
            if not time_ok() or calls >= global_budget:
                break
            fx = wrapped_func(x)
            evaluated_candidates.append((fx, x))
            
        if not evaluated_candidates:
            # Fallback if something went wrong
            x0 = np.zeros(dim)
            try:
                fx0 = wrapped_func(x0)
                best_x = x0
                best_f = fx0
            except:
                pass
            return best_x, best_f

        evaluated_candidates.sort(key=lambda t: t[0])
        best_x = evaluated_candidates[0][1].copy()
        best_f = evaluated_candidates[0][0]

        # Phase 2: DFTR Local Refinement
        # Use remaining budget.
        
        # DFTR Parameters
        # We need 2*dim points to approximate gradient and Hessian (via finite differences of the surrogate).
        # Standard DFTR uses a set of points in a trust region.
        # Let's implement a simplified DFTR:
        # 1. Maintain a trust region radius r.
        # 2. Generate 2*dim points around current best x.
        # 3. Fit a quadratic model to these points + center.
        # 4. Minimize the quadratic model within the trust region.
        # 5. Evaluate the new point. If it improves, accept and possibly increase r. If not, decrease r.
        
        # To be efficient, we can reuse points if the trust region doesn't change much, 
        # but for simplicity and robustness within the budget, we'll regenerate points 
        # when the center moves or radius changes significantly.
        
        # However, regenerating 2*dim points every step is expensive.
        # A better approach: Use the "derivative-free" approximation of the gradient and Hessian
        # from the most recent 2*dim evaluations.
        
        # Let's stick to a robust implementation:
        # We maintain a set of 2*dim + 1 points.
        # Initially, we use the points from the global phase if they are close, 
        # or generate new ones.
        
        # Let's generate a new set of points around the best_x found in global phase.
        # We have `calls` used so far.
        # Remaining budget: self.budget - calls
        
        if calls >= self.budget:
            return best_x, best_f

        # Initialize DFTR state
        x_center = best_x.copy()
        f_center = best_f
        r = 1.0 # Initial trust region radius
        
        # Generate initial set of points for the first quadratic model
        # We need 2*dim points.
        # Directions: e1, -e1, e2, -e2, ...
        
        # Check if we have enough budget for 2*dim + 1 evaluations
        # We already have f_center. We need 2*dim more.
        
        # If budget is tight, we might not be able to do a full DFTR.
        # But 500 budget is plenty for 5D. 2*5=10 points per model.
        
        dftr_rng = np.random.default_rng(1)
        
        # Main DFTR loop
        while calls < self.budget and time_ok():
            # Check if we have enough budget for a full model update (2*dim points)
            if self.budget - calls < 2 * dim + 1:
                # Not enough for a full model, do simple (1+1) ES for the rest
                # Or just break and let the remaining budget be used for small steps?
                # Let's do a simple hill climbing for the remainder.
                while calls < self.budget and time_ok():
                    step = 0.1 * r
                    y = x_center + step * dftr_rng.standard_normal(dim)
                    y = np.clip(y, -5, 5)
                    fy = wrapped_func(y)
                    if fy < f_center:
                        x_center = y
                        f_center = fy
                    else:
                        # Reduce step size if stuck
                        r *= 0.9
                        if r < 1e-6:
                            break
                break
            
            # Generate 2*dim points
            points = []
            values = []
            
            # Directions
            for i in range(dim):
                dir1 = np.zeros(dim)
                dir2 = np.zeros(dim)
                dir1[i] = 1
                dir2[i] = -1
                
                p1 = x_center + r * dir1
                p2 = x_center + r * dir2
                
                p1 = np.clip(p1, -5, 5)
                p2 = np.clip(p2, -5, 5)
                
                points.append(p1)
                points.append(p2)
            
            # Evaluate points
            for p in points:
                if not time_ok() or calls >= self.budget:
                    break
                fp = wrapped_func(p)
                values.append(fp)
            
            # If we couldn't evaluate all points, break
            if len(values) < 2 * dim:
                break
            
            # Fit quadratic model
            # We have 2*dim + 1 points (center + 2*dim neighbors)
            # f(x) ≈ f0 + g^T (x - x_center) + 0.5 (x - x_center)^T H (x - x_center)
            
            # Build design matrix
            # Variables: g (dim), H (dim*dim/2 unique elements), f0 (1)
            # Total variables: dim + dim*(dim+1)/2 + 1
            # For dim=5: 5 + 15 + 1 = 21 variables.
            # Points: 11.
            # We have fewer equations than variables!
            # So we cannot fit a full quadratic model with only 2*dim + 1 points.
            
            # Correction:
            # A standard quadratic model requires at least (dim+1)(dim+2)/2 points.
            # For dim=5, that's 21 points.
            # We only have 11 points (center + 10 neighbors).
            # We can fit a linear model (gradient) with 2*dim points.
            # Or a partial quadratic model.
            
            # Let's fit a linear model first (Gradient estimation)
            # f(x + r*e_i) - f(x) ≈ r * g_i
            # So g_i ≈ (f(x + r*e_i) - f(x)) / r
            
            g = np.zeros(dim)
            for i in range(dim):
                # p1 is x + r*e_i, p2 is x - r*e_i
                f_p1 = values[2*i]
                f_p2 = values[2*i+1]
                # Central difference for gradient
                g[i] = (f_p1 - f_p2) / (2 * r)
            
            # Now we have an estimate of the gradient.
            # We can use this to propose a step.
            # Step direction: -g
            # Step size: alpha * ||g||? Or just a fixed fraction of r?
            
            # Let's try a step in the negative gradient direction.
            # We need to choose a step length.
            # We can minimize the linear model: f(x) + g^T (x - x_center)
            # Within the trust region ||x - x_center|| <= r.
            # The minimum of the linear model is at infinity in the -g direction.
            # So we take a step of size r in the direction -g/||g||.
            
            norm_g = np.linalg.norm(g)
            if norm_g < 1e-10:
                # Gradient is zero, try a random perturbation
                direction = dftr_rng.standard_normal(dim)
                direction /= np.linalg.norm(direction)
            else:
                direction = -g / norm_g
            
            # Propose new point
            x_new = x_center + r * direction
            x_new = np.clip(x_new, -5, 5)
            
            # Evaluate new point
            f_new = wrapped_func(x_new)
            
            # Accept/Reject
            if f_new < f_center:
                # Accept
                x_center = x_new
                f_center = f_new
                # Increase trust region slightly
                r *= 1.1
                # Cap r
                r = min(r, 5.0)
            else:
                # Reject
                # Decrease trust region
                r *= 0.5
                if r < 1e-6:
                    break

        return best_x, best_f
