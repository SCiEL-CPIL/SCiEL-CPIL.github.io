<<<<<<< SEARCH
    def objective_and_constraints(vars):
        # vars: [x1, y1, r1, x2, y2, r2, ...]
        x = vars[0::3]
        y = vars[1::3]
        r = vars[2::3]
        
        # Objective: Maximize sum of radii -> Minimize -sum(r)
        obj = -np.sum(r)
        
        # Constraints:
        # 1. Boundary: x >= r, x <= 1-r, y >= r, y <= 1-r
        # 2. Non-overlap: dist(i,j) >= ri + rj
        
        constraints = []
        
        # Boundary constraints (inequality: c(x) >= 0)
        for i in range(N):
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i] - v[3*i+2]})       # x_i - r_i >= 0
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - v[3*i] - v[3*i+2]})   # 1 - x_i - r_i >= 0
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i+1] - v[3*i+2]})     # y_i - r_i >= 0
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - v[3*i+1] - v[3*i+2]}) # 1 - y_i - r_i >= 0
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i+2] - 1e-6})         # r_i > 0
        
        # Non-overlap constraints
        for i in range(N):
            for j in range(i+1, N):
                def overlap_constraint(v, i=i, j=j):
                    xi, yi, ri = v[3*i], v[3*i+1], v[3*i+2]
                    xj, yj, rj = v[3*j], v[3*j+1], v[3*j+2]
                    dist_sq = (xi - xj)**2 + (yi - yj)**2
                    sum_r = ri + rj
                    return dist_sq - sum_r**2
                constraints.append({'type': 'ineq', 'fun': overlap_constraint})
        
        return obj, constraints

    def run_optimizer(init_config):
        x0 = init_config.flatten()
        
        # Bounds for variables
        # x, y in [0, 1], r in [0, 0.5]
        bounds = []
        for i in range(N):
            bounds.append((0, 1)) # x
            bounds.append((0, 1)) # y
            bounds.append((1e-6, 0.5)) # r
            
        try:
            res = minimize(
                lambda v: objective_and_constraints(v)[0],
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=lambda v: [c['fun'](v) for c in objective_and_constraints(v)[1]],
                options={'maxiter': 1000, 'ftol': 1e-8, 'disp': False}
            )
            if res.success:
                val = -res.fun
                # Verify validity
                x = res.x[0::3]
                y = res.x[1::3]
                r = res.x[2::3]
                
                # Check boundaries
                if np.any(x < r - 1e-5) or np.any(x > 1 - r + 1e-5) or \
                   np.any(y < r - 1e-5) or np.any(y > 1 - r + 1e-5) or \
                   np.any(r < 1e-6):
                    return None, -np.inf
                
                # Check overlaps
                for i in range(N):
                    for j in range(i+1, N):
                        dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                        if dist < r[i] + r[j] - 1e-5:
                            return None, -np.inf
                
                return np.column_stack((x, y, r)), val
            else:
                return None, -np.inf
        except Exception as e:
            return None, -np.inf
=======
    def compute_grad_penalty(v):
        """
        Computes gradient of the penalty function and the objective value.
        v: flattened array [x1, y1, r1, ...]
        Returns: obj_val, grad
        """
        x = v[0::3].copy()
        y = v[1::3].copy()
        r = v[2::3].copy()
        
        # Objective: maximize sum(r) -> minimize -sum(r)
        obj = -np.sum(r)
        grad = np.zeros_like(v)
        
        # Gradient of objective: -1 for each r component
        grad[2::3] = -1.0
        
        # Penalty for boundary violations
        # We want x >= r, x <= 1-r, y >= r, y <= 1-r
        # Soft penalty: max(0, violation)^2
        # Let's use a quadratic penalty for smoothness
        
        lam = 100.0 # Penalty weight
        
        # Boundary penalties
        # x - r < 0
        viol_x_min = np.maximum(0, r - x)
        if np.any(viol_x_min > 0):
            # d/dx (r-x)^2 = -2(r-x), d/dr (r-x)^2 = 2(r-x)
            grad[0::3] += -2 * lam * viol_x_min
            grad[2::3] += 2 * lam * viol_x_min
            obj += lam * np.sum(viol_x_min**2)
            
        # 1 - x - r < 0  => x + r > 1
        viol_x_max = np.maximum(0, x + r - 1)
        if np.any(viol_x_max > 0):
            # d/dx (x+r-1)^2 = 2(x+r-1), d/dr (x+r-1)^2 = 2(x+r-1)
            grad[0::3] += 2 * lam * viol_x_max
            grad[2::3] += 2 * lam * viol_x_max
            obj += lam * np.sum(viol_x_max**2)
            
        # y - r < 0
        viol_y_min = np.maximum(0, r - y)
        if np.any(viol_y_min > 0):
            grad[1::3] += -2 * lam * viol_y_min
            grad[2::3] += 2 * lam * viol_y_min
            obj += lam * np.sum(viol_y_min**2)
            
        # 1 - y - r < 0
        viol_y_max = np.maximum(0, y + r - 1)
        if np.any(viol_y_max > 0):
            grad[1::3] += 2 * lam * viol_y_max
            grad[2::3] += 2 * lam * viol_y_max
            obj += lam * np.sum(viol_y_max**2)

        # Overlap penalties
        # dist(i,j) < r_i + r_j
        # Penalty: max(0, (r_i+r_j) - dist)^2
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                sum_r = r[i] + r[j]
                
                if dist < sum_r:
                    # violation = sum_r - dist
                    viol = sum_r - dist
                    # Gradient of (sum_r - dist)^2
                    # d/dx_i = 2*viol * d/dx_i (sum_r - dist) = 2*viol * (-dx/dist)
                    # d/dr_i = 2*viol * 1
                    if dist > 1e-8:
                        factor = 2 * lam * viol / dist
                        grad[3*i] += -factor * dx
                        grad[3*i+1] += -factor * dy
                        grad[3*i+2] += 2 * lam * viol
                        
                        grad[3*j] += factor * dx
                        grad[3*j+1] += factor * dy
                        grad[3*j+2] += 2 * lam * viol
                    else:
                        # If dist is 0, random push apart
                        grad[3*i] += 2 * lam * viol * 0.1
                        grad[3*j] -= 2 * lam * viol * 0.1
                        grad[3*i+2] += 2 * lam * viol
                        grad[3*j+2] += 2 * lam * viol
                    
                    obj += lam * viol**2
        
        return obj, grad

    def run_optimizer(init_config):
        x0 = init_config.flatten().copy()
        step_size = 0.01
        max_iters = 5000
        last_obj = np.inf
        
        for it in range(max_iters):
            if time.time() - start_time > DEADLINE:
                break
            
            obj, grad = compute_grad_penalty(x0)
            
            # Simple backtracking line search
            alpha = step_size
            new_obj = obj
            for _ in range(20):
                x_new = x0 - alpha * grad
                # Clip to valid bounds roughly to keep it sane
                x_new[0::3] = np.clip(x_new[0::3], 0.0, 1.0)
                x_new[1::3] = np.clip(x_new[1::3], 0.0, 1.0)
                x_new[2::3] = np.clip(x_new[2::3], 1e-6, 0.5)
                
                new_obj, _ = compute_grad_penalty(x_new)
                if new_obj < obj:
                    x0 = x_new
                    break
                alpha *= 0.5
            
            if new_obj > last_obj:
                step_size *= 0.5
            last_obj = new_obj
            
            # Convergence check
            if np.linalg.norm(grad) < 1e-6:
                break

        # Final verification
        x = x0[0::3]
        y = x0[1::3]
        r = x0[2::3]
        
        # Check boundaries
        if np.any(x < r - 1e-5) or np.any(x > 1 - r + 1e-5) or \
           np.any(y < r - 1e-5) or np.any(y > 1 - r + 1e-5) or \
           np.any(r < 1e-6):
            # Try to fix small violations by shrinking radii
            r = np.minimum(r, x)
            r = np.minimum(r, 1 - x)
            r = np.minimum(r, y)
            r = np.minimum(r, 1 - y)
            r = np.maximum(r, 1e-6)
            
            # Re-check
            if np.any(x < r - 1e-5) or np.any(x > 1 - r + 1e-5) or \
               np.any(y < r - 1e-5) or np.any(y > 1 - r + 1e-5) or \
               np.any(r < 1e-6):
                return None, -np.inf
        
        # Check overlaps
        for i in range(N):
            for j in range(i+1, N):
                dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                if dist < r[i] + r[j] - 1e-5:
                    # Shrink radii slightly to make it valid? 
                    # Or reject. Rejecting is safer for "maximize sum" if we can't fix easily.
                    # But for a valid result, we need disjoint interiors.
                    # Let's try a simple local repair: shrink both
                    overlap = r[i] + r[j] - dist
                    if overlap > 0:
                        shrink_i = overlap / 2
                        shrink_j = overlap / 2
                        r[i] -= shrink_i
                        r[j] -= shrink_j
                        # Ensure r doesn't go below 0 or boundary limits
                        r[i] = max(1e-6, min(r[i], x[i], 1-x[i], y[i], 1-y[i]))
                        r[j] = max(1e-6, min(r[j], x[j], 1-x[j], y[j], 1-y[j]))
                    
                    # Re-check after repair
                    dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                    if dist < r[i] + r[j] - 1e-5:
                        return None, -np.inf
        
        val = np.sum(r)
        return np.column_stack((x, y, r)), val
>>>>>>> REPLACE

I also need to fix the fallback bug in the main function.

