import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        calls_made = 0
        best_x = np.zeros(dim)
        best_f = float('inf')

        def evaluate(pt):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return best_f
            if time.time() > deadline:
                return best_f
            pt = np.asarray(pt, dtype=np.float64)
            if pt.shape != (dim,):
                pt = pt.reshape(dim)
            val = func(pt)
            calls_made += 1
            if val < best_f:
                best_f = val
                best_x = pt.copy()
            return val

        # Phase 1: Sphere-Specific Model Construction
        # Assume f(x) = a * ||x||^2 + q^T x + c
        # Points: 0, +/- d * e_i
        d = 2.0
        points = [np.zeros(dim)]
        for i in range(dim):
            p1 = np.zeros(dim); p1[i] = d
            p2 = np.zeros(dim); p2[i] = -d
            points.append(p1)
            points.append(p2)
            
        values = np.array([evaluate(p) for p in points])
        
        c_val = values[0]
        a_vals = []
        q_vec = np.zeros(dim)
        
        for i in range(dim):
            val_pos = values[1 + 2*i]
            val_neg = values[2 + 2*i]
            
            # f(d*e_i) = a*d^2 + q_i*d + c
            # f(-d*e_i) = a*d^2 - q_i*d + c
            # Diff: 2*q_i*d = val_pos - val_neg
            # Sum: 2*a*d^2 + 2*c = val_pos + val_neg
            
            q_i = (val_pos - val_neg) / (2 * d)
            a_i = (val_pos + val_neg - 2*c_val) / (2 * d * d)
            
            a_vals.append(a_i)
            q_vec[i] = q_i
            
        a = np.mean(a_vals)
        
        # Robustness: If a is very small or negative (shouldn't happen for Sphere minimization),
        # fall back to a heuristic.
        if abs(a) < 1e-10:
            # If a is near 0, the function is nearly linear or constant.
            # The minimum is likely at the boundary or determined by q.
            # For now, treat as small positive to avoid division by zero.
            a = 1e-10 if a >= 0 else -1e-10
            
        # Minimum at x* = -q / (2*a)
        x_init = -q_vec / (2 * a)
        x_init = np.clip(x_init, -5, 5)
        
        # Evaluate the estimated optimum
        f_init = evaluate(x_init)
        
        # Phase 2: Local Polish
        # Use a (1+1)-ES with adaptive step size.
        # Since we are likely at the optimum, start with a small sigma.
        x = best_x.copy()
        fx = best_f
        
        # Estimate scale from 'a'. If a is large, curvature is high, step should be small.
        sigma_scale = 1.0 / np.sqrt(abs(a)) if abs(a) > 1e-12 else 1.0
        
        # Start with a relatively small sigma to avoid overshooting the exact minimum
        # found by the analytic fit.
        sigma = 0.1 * sigma_scale
        
        rng = np.random.default_rng(12345)
        
        # Polish loop
        no_improvement_count = 0
        max_no_improvement = 20
        
        while calls_made < budget and time.time() < deadline:
            z = rng.standard_normal(dim)
            step = sigma * z
            y = x + step
            y = np.clip(y, -5, 5)
            
            fy = evaluate(y)
            
            if fy < fx:
                x = y
                fx = fy
                # Update global best
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                no_improvement_count = 0
                # Increase sigma slightly on success
                sigma *= 1.1
            else:
                no_improvement_count += 1
                # Decrease sigma on failure
                sigma *= 0.9
                
            # Convergence checks
            if sigma < 1e-12 * sigma_scale:
                break
                
            if no_improvement_count > max_no_improvement:
                # If we haven't improved in many steps, we are likely at the minimum
                # within numerical precision.
                break

        return best_x, best_f
