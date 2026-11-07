import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290
        dim = self.dim
        budget = self.budget
        
        calls_made = 0
        best_x = np.zeros(dim)
        best_f = float('inf')
        
        def can_call():
            nonlocal calls_made
            if calls_made >= budget:
                return False
            if time.time() > deadline:
                return False
            return True

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            if not can_call():
                return float('inf')
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Strategy:
        # 1. Initial coarse search to find a reasonably good basin.
        # 2. Refine using local quadratic model (Newton step) derived from finite differences.
        # 3. If Newton step fails or doesn't improve, fall back to local random perturbations or restart.
        # Since f11 (Discus) is a shifted/rotated quadratic, a local quadratic fit should be very accurate
        # once we are close to the minimum.
        
        # Phase 1: Global Search
        # Use random sampling to find a good starting point.
        # For dim=5, 100 random points is cheap and effective for global localization.
        n_global = 100
        for _ in range(n_global):
            if not can_call():
                break
            x_rand = np.random.uniform(-5, 5, dim)
            evaluate(x_rand)
            
        # If we ran out of budget, return best
        if not can_call():
            return best_x

        # Phase 2: Local Refinement via Quadratic Model
        # We will iterate:
        # 1. Fit a quadratic model at current best_x.
        # 2. Solve for the minimum of the model (Newton step).
        # 3. Evaluate the new point.
        # 4. If improved, update best. If not, try a smaller step or perturb best_x and refit.
        
        # Finite difference step size
        # Too small: numerical noise. Too large: model inaccuracy.
        # For f11, the curvature is relatively mild (diagonal 1 to 100).
        # h = 1e-2 is a safe bet for double precision.
        h = 1e-2
        
        max_refine_iters = 300 # We have ~400 calls left. Each refine iter costs 2*dim+1 = 11 calls.
        # 300 iters * 11 calls = 3300 calls, which is way more than budget.
        # So we are budget-limited, not iteration-limited.
        # Let's just run until budget is exhausted or we converge.
        
        current_x = best_x.copy()
        current_f = best_f
        
        # We'll keep a history of recent improvements to detect stagnation
        last_improvement_calls = calls_made
        
        while can_call():
            # Check time
            if time.time() > deadline - 5:
                break
                
            # Build finite difference matrix
            # f(x) + sum_i grad_i x_i + 0.5 * x^T H x
            # We need f(x), grad, H.
            # grad_i = (f(x+h e_i) - f(x-h e_i)) / (2h)
            # H_ij = (f(x+h e_i) - f(x) + f(x-h e_i) - f(x) ... wait, standard central diff for Hessian is tricky with 2nd order.
            # Better: Use forward differences for Hessian approximation? Or just use the standard 2nd order central difference for diagonal and cross terms?
            # Actually, for a quadratic, we can just fit the coefficients using 2*dim+1 points?
            # No, a quadratic in dim dims has dim*(dim+1)/2 + dim + 1 parameters.
            # For dim=5: 15 + 5 + 1 = 21 parameters.
            # We have 500 calls. 21 points per fit is very efficient!
            # Let's fit a full quadratic model using 21 points around current_x.
            # Points:
            # 1. center
            # 2. center + h*e_i for i in 1..dim
            # 3. center - h*e_i for i in 1..dim
            # 4. center + h*(e_i + e_j) for i<j ? No, that's not standard.
            # Standard finite difference for Hessian:
            # H_ij = (f(x+h(e_i+e_j)) - f(x+h e_i) - f(x+h e_j) + f(x)) / h^2  (for i!=j)
            # H_ii = (f(x+h e_i) - 2f(x) + f(x-h e_i)) / h^2
            # This requires:
            # 1 center
            # dim forward (x+h e_i)
            # dim backward (x-h e_i)
            # dim*(dim-1)/2 cross (x+h(e_i+e_j))
            # Total: 1 + dim + dim + dim*(dim-1)/2 = 1 + 2*5 + 10 = 21 points.
            
            # Check if we have enough budget for a full fit (21 calls)
            if budget - calls_made < 25:
                # Not enough for a full fit, just do random local search
                for _ in range(10):
                    if not can_call(): break
                    x_try = current_x + np.random.normal(0, h*0.5, dim)
                    evaluate(x_try)
                continue

            # Generate points for quadratic fit
            x_center = current_x
            points = [x_center.copy()]
            f_values = [current_f] # We already have this, but let's re-evaluate to be safe? No, use cached.
            # Wait, current_f might be stale if current_x was updated by a previous step but we didn't re-evaluate?
            # In this loop, current_x is always the best_x found so far, which was evaluated.
            # However, if we moved current_x via Newton step, we evaluated it.
            # So f_values[0] is valid.
            
            # Forward and Backward
            for i in range(dim):
                x_f = x_center.copy()
                x_b = x_center.copy()
                x_f[i] += h
                x_b[i] -= h
                points.append(x_f)
                points.append(x_b)
                
            # Cross terms
            for i in range(dim):
                for j in range(i+1, dim):
                    x_c = x_center.copy()
                    x_c[i] += h
                    x_c[j] += h
                    points.append(x_c)
                    
            # Evaluate all new points
            for p in points[1:]:
                if not can_call():
                    # Budget ran out mid-fit. Abort fit.
                    break
                f_val = evaluate(p)
                f_values.append(f_val)
                
            if not can_call():
                break
                
            # If we didn't get all points, skip this iteration
            if len(f_values) < len(points):
                continue
                
            # Fit quadratic model
            # f(x) = c + g^T (x - x_c) + 0.5 (x - x_c)^T H (x - x_c)
            # Let d = x - x_c
            # We have values at d=0, d=h e_i, d=-h e_i, d=h(e_i+e_j)
            
            # Gradient estimation
            grad = np.zeros(dim)
            for i in range(dim):
                f_f = f_values[1 + 2*i]
                f_b = f_values[1 + 2*i + 1]
                grad[i] = (f_f - f_b) / (2*h)
                
            # Hessian estimation
            H = np.zeros((dim, dim))
            # Diagonal
            for i in range(dim):
                f_f = f_values[1 + 2*i]
                f_b = f_values[1 + 2*i + 1]
                f_c = f_values[0]
                H[i, i] = (f_f - 2*f_c + f_b) / (h*h)
                
            # Off-diagonal
            # Indexing in f_values:
            # 0: center
            # 1..2*dim: forward/backward
            # 2*dim + 1 ... : cross terms
            cross_idx = 1 + 2*dim
            for i in range(dim):
                for j in range(i+1, dim):
                    f_cross = f_values[cross_idx]
                    f_fi = f_values[1 + 2*i]
                    f_fj = f_values[1 + 2*j]
                    f_c = f_values[0]
                    # H_ij = (f(x+h(e_i+e_j)) - f(x+h e_i) - f(x+h e_j) + f(x)) / h^2
                    # Note: f(x+h e_i) is f_fi. f(x+h e_j) is f_fj.
                    H[i, j] = (f_cross - f_fi - f_fj + f_c) / (h*h)
                    H[j, i] = H[i, j]
                    cross_idx += 1
            
            # Solve for minimum of the quadratic model
            # grad + H * d = 0 => d = -H^{-1} grad
            # Regularize H to ensure positive definiteness / invertibility
            # If H is not positive definite, the model might be a saddle or max.
            # Add small ridge for stability.
            H_reg = H + 1e-6 * np.eye(dim)
            
            try:
                # Check if H is positive definite
                eigvals = np.linalg.eigvalsh(H_reg)
                if np.min(eigvals) < 1e-8:
                    # Not convex. Maybe use gradient descent step instead?
                    # Or just perturb and retry.
                    # For now, take a small gradient step
                    step = -0.01 * grad
                else:
                    d = np.linalg.solve(H_reg, -grad)
                    # Limit step size to avoid going too far
                    # If norm(d) is huge, clip it
                    norm_d = np.linalg.norm(d)
                    if norm_d > 2.0:
                        d = d / norm_d * 2.0
                    else:
                        d = d
            except np.linalg.LinAlgError:
                d = -0.01 * grad
            
            x_new = x_center + d
            # Clip to box
            x_new = np.clip(x_new, -5, 5)
            
            # Evaluate new point
            f_new = evaluate(x_new)
            
            if f_new < best_f:
                best_f = f_new
                best_x = x_new.copy()
                current_x = x_new.copy()
                current_f = f_new
                last_improvement_calls = calls_made
                
                # If we got very close, reduce h?
                # For f11, h=1e-2 is fine.
            else:
                # No improvement. 
                # Maybe the model is bad or we are at a boundary.
                # Try a smaller step in the Newton direction?
                # Or just perturb current_x slightly to escape local flatness?
                # Since f11 is convex (positive definite Hessian), if Newton step doesn't improve,
                # it might be due to numerical error or we are already at the minimum.
                # Let's try a halving line search.
                improved = False
                for alpha in [0.5, 0.25, 0.1]:
                    if not can_call():
                        break
                    x_try = x_center + alpha * d
                    x_try = np.clip(x_try, -5, 5)
                    f_try = evaluate(x_try)
                    if f_try < best_f:
                        best_f = f_try
                        best_x = x_try.copy()
                        current_x = x_try.copy()
                        current_f = f_try
                        last_improvement_calls = calls_made
                        improved = True
                        break
                if not improved:
                    # Stagnation. Perturb current_x to find a new basin or refine h.
                    # Since f11 is unimodal (convex), if we don't improve, we might be stuck.
                    # Let's do a few random local steps.
                    for _ in range(5):
                        if not can_call():
                            break
                        x_try = current_x + np.random.normal(0, 0.1, dim)
                        x_try = np.clip(x_try, -5, 5)
                        f_try = evaluate(x_try)
                        if f_try < best_f:
                            best_f = f_try
                            best_x = x_try.copy()
                            current_x = x_try.copy()
                            current_f = f_try
                            last_improvement_calls = calls_made
                            break

        return best_x
