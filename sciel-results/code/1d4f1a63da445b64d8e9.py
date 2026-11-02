<<<<<<< SEARCH
    def polish_gradient(state, time_limit, max_iter=5000):
        """
        Gradient-based polish using smooth penalty formulation.
        """
        from scipy.optimize import minimize
        
        def objective(s):
            t = s[0]
            w, h = get_rect_dims(t)
            coords = np.zeros((N, 3))
            for i in range(N):
                coords[i, 0] = s[1 + 3*i]
                coords[i, 1] = s[1 + 3*i+1]
                coords[i, 2] = s[1 + 3*i+2]
            
            radii = coords[:, 2]
            x = coords[:, 0]
            y = coords[:, 1]
            
            # Negative sum of radii (we minimize)
            obj = -np.sum(radii)
            
            # Smooth boundary penalties using softplus-like approximation
            # Use quadratic penalty with smoothing
            eps = 1e-6
            viol_l = np.maximum(0, radii - x)
            viol_r = np.maximum(0, x + radii - w)
            viol_b = np.maximum(0, radii - y)
            viol_t = np.maximum(0, y + radii - h)
            
            obj += 1000.0 * (np.sum(viol_l**2) + np.sum(viol_r**2) + np.sum(viol_b**2) + np.sum(viol_t**2))
            
            # Overlap penalty using smooth approximation
            idx_i, idx_j = np.triu_indices(N, k=1)
            dx = x[idx_i] - x[idx_j]
            dy = y[idx_i] - y[idx_j]
            dist_sq = dx*dx + dy*dy
            min_dist = radii[idx_i] + radii[idx_j]
            
            # Smooth penalty: use max(0, min_dist^2 - dist_sq) with smoothing
            overlap_amount = min_dist**2 - dist_sq
            # Use a smooth approximation of max(0, x) = 0.5*(x + sqrt(x^2 + eps^2))
            smooth_overlap = 0.5 * (overlap_amount + np.sqrt(overlap_amount**2 + 1e-8))
            obj += 5000.0 * np.sum(smooth_overlap**2)
            
            return obj
        
        def gradient(s):
            t = s[0]
            w, h = get_rect_dims(t)
            coords = np.zeros((N, 3))
            for i in range(N):
                coords[i, 0] = s[1 + 3*i]
                coords[i, 1] = s[1 + 3*i+1]
                coords[i, 2] = s[1 + 3*i+2]
            
            radii = coords[:, 2]
            x = coords[:, 0]
            y = coords[:, 1]
            
            grad = np.zeros_like(s)
            
            # d(-sum_r)/dr_i = -1
            for i in range(N):
                grad[1 + 3*i + 2] += -1.0
            
            # Boundary gradients
            viol_l = np.maximum(0, radii - x)
            for i in range(N):
                if viol_l[i] > 0:
                    grad[1 + 3*i] += -2.0 * 1000.0 * viol_l[i]
                    grad[1 + 3*i + 2] += 2.0 * 1000.0 * viol_l[i]
            
            viol_r = np.maximum(0, x + radii - w)
            for i in range(N):
                if viol_r[i] > 0:
                    grad[1 + 3*i] += 2.0 * 1000.0 * viol_r[i]
                    grad[1 + 3*i + 2] += 2.0 * 1000.0 * viol_r[i]
            
            viol_b = np.maximum(0, radii - y)
            for i in range(N):
                if viol_b[i] > 0:
                    grad[1 + 3*i + 1] += -2.0 * 1000.0 * viol_b[i]
                    grad[1 + 3*i + 2] += 2.0 * 1000.0 * viol_b[i]
            
            viol_t = np.maximum(0, y + radii - h)
            for i in range(N):
                if viol_t[i] > 0:
                    grad[1 + 3*i + 1] += 2.0 * 1000.0 * viol_t[i]
                    grad[1 + 3*i + 2] += 2.0 * 1000.0 * viol_t[i]
            
            # Overlap gradients
            idx_i, idx_j = np.triu_indices(N, k=1)
            dx = x[idx_i] - x[idx_j]
            dy = y[idx_i] - y[idx_j]
            dist_sq = dx*dx + dy*dy
            min_dist = radii[idx_i] + radii[idx_j]
            overlap_amount = min_dist**2 - dist_sq
            
            # Gradient of smooth_overlap = 0.5 * (1 + overlap_amount / sqrt(overlap_amount^2 + 1e-8))
            smooth_grad = 0.5 * (1.0 + overlap_amount / np.sqrt(overlap_amount**2 + 1e-8))
            
            # d/dx_i of smooth_overlap: -2*dx * smooth_grad
            # d/dx_j of smooth_overlap: +2*dx * smooth_grad
            # d/dr_i of smooth_overlap: 2*min_dist * smooth_grad
            # d/dr_j of smooth_overlap: 2*min_dist * smooth_grad
            
            # Penalty is 5000 * sum(smooth_overlap^2)
            # d/dx_i: 5000 * 2 * smooth_overlap * (-2*dx) * smooth_grad
            # d/dr_i: 5000 * 2 * smooth_overlap * (2*min_dist) * smooth_grad
            
            for k in range(len(idx_i)):
                i = idx_i[k]
                j = idx_j[k]
                so = smooth_overlap[k]
                sg = smooth_grad[k]
                
                # x_i
                grad[1 + 3*i] += 5000.0 * 2.0 * so * (-2.0 * dx[k]) * sg
                # x_j
                grad[1 + 3*j] += 5000.0 * 2.0 * so * (2.0 * dx[k]) * sg
                # y_i
                grad[1 + 3*i + 1] += 5000.0 * 2.0 * so * (-2.0 * dy[k]) * sg
                # y_j
                grad[1 + 3*j + 1] += 5000.0 * 2.0 * so * (2.0 * dy[k]) * sg
                # r_i
                grad[1 + 3*i + 2] += 5000.0 * 2.0 * so * (2.0 * min_dist[k]) * sg
                # r_j
                grad[1 + 3*j + 2] += 5000.0 * 2.0 * so * (2.0 * min_dist[k]) * sg
            
            return grad
        
        # Use scipy minimize with L-BFGS-B
        bounds = [(0.1, 0.9)]
        for i in range(N):
            bounds.append((0, 2.0))  # x
            bounds.append((0, 2.0))  # y
            bounds.append((1e-4, 0.5))  # r
        
        try:
            result = minimize(
                objective,
                state,
                jac=gradient,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if result.success or result.fun < objective(state):
                return result.x
        except Exception:
            pass
        
        return state
=======
    def force_relax(state, n_iters=200, dt=0.01):
        """
        Force-directed relaxation: push overlapping circles apart, pull toward center.
        This helps discover contact graphs and escape local minima.
        """
        state = state.copy()
        for it in range(n_iters):
            t = state[0]
            w, h = get_rect_dims(t)
            
            forces_x = np.zeros(N)
            forces_y = np.zeros(N)
            forces_r = np.zeros(N)
            
            # Repulsive forces between overlapping or close circles
            for i in range(N):
                for j in range(i+1, N):
                    xi = state[1 + 3*i]
                    yi = state[1 + 3*i + 1]
                    ri = state[1 + 3*i + 2]
                    xj = state[1 + 3*j]
                    yj = state[1 + 3*j + 1]
                    rj = state[1 + 3*j + 2]
                    
                    dx = xi - xj
                    dy = yi - yj
                    dist_sq = dx*dx + dy*dy
                    min_dist = ri + rj
                    
                    if dist_sq < min_dist * min_dist:
                        dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 1e-6
                        overlap = min_dist - dist
                        # Force magnitude proportional to overlap
                        f = 10.0 * overlap
                        # Normalize direction
                        fx = f * dx / dist
                        fy = f * dy / dist
                        
                        forces_x[i] += fx
                        forces_y[i] += fy
                        forces_x[j] -= fx
                        forces_y[j] -= fy
                        
                        # Also push radii apart slightly
                        forces_r[i] -= 0.1 * overlap
                        forces_r[j] -= 0.1 * overlap
                    elif dist_sq < (min_dist * 1.2)**2:
                        # Mild attraction to maintain contact (helps packing density)
                        dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 1e-6
                        gap = dist - min_dist
                        if gap < 0.1:
                            f = -1.0 * gap  # attractive
                            fx = f * dx / dist
                            fy = f * dy / dist
                            forces_x[i] += fx
                            forces_y[i] += fy
                            forces_x[j] -= fx
                            forces_y[j] -= fy
            
            # Boundary forces: push circles inward if they're too close to walls
            for i in range(N):
                xi = state[1 + 3*i]
                yi = state[1 + 3*i + 1]
                ri = state[1 + 3*i + 2]
                
                # Left wall
                if xi - ri < 0.05:
                    forces_x[i] += 5.0 * (0.05 - (xi - ri))
                # Right wall
                if xi + ri > w - 0.05:
                    forces_x[i] -= 5.0 * ((xi + ri) - (w - 0.05))
                # Bottom wall
                if yi - ri < 0.05:
                    forces_y[i] += 5.0 * (0.05 - (yi - ri))
                # Top wall
                if yi + ri > h - 0.05:
                    forces_y[i] -= 5.0 * ((yi + ri) - (h - 0.05))
                
                # Gentle centering force to prevent drift
                forces_x[i] += 0.1 * (w/2.0 - xi)
                forces_y[i] += 0.1 * (h/2.0 - yi)
            
            # Apply forces with adaptive step size
            for i in range(N):
                state[1 + 3*i] += dt * forces_x[i]
                state[1 + 3*i + 1] += dt * forces_y[i]
                state[1 + 3*i + 2] += dt * forces_r[i]
                
                # Clamp to valid range
                state[1 + 3*i] = np.clip(state[1 + 3*i], 0, w)
                state[1 + 3*i + 1] = np.clip(state[1 + 3*i + 1], 0, h)
                state[1 + 3*i + 2] = np.clip(state[1 + 3*i + 2], 1e-4, 0.5 * min(w, h))
            
            # Decay step size
            dt *= 0.995
        
        return state

    def polish_gradient(state, time_limit, max_iter=5000):
        """
        Gradient-based polish using smooth penalty formulation with log-sum-exp smoothing.
        """
        from scipy.optimize import minimize
        
        def objective_and_grad(s):
            t = s[0]
            w, h = get_rect_dims(t)
            coords = np.zeros((N, 3))
            for i in range(N):
                coords[i, 0] = s[1 + 3*i]
                coords[i, 1] = s[1 + 3*i+1]
                coords[i, 2] = s[1 + 3*i+2]
            
            radii = coords[:, 2]
            x = coords[:, 0]
            y = coords[:, 1]
            
            # Negative sum of radii (we minimize)
            obj = -np.sum(radii)
            grad = np.zeros_like(s)
            
            # d(-sum_r)/dr_i = -1
            for i in range(N):
                grad[1 + 3*i + 2] += -1.0
            
            # Smooth boundary penalties using log-sum-exp for smoothness
            beta_b = 50.0  # sharpness parameter
            
            viol_l = radii - x
            viol_r = x + radii - w
            viol_b = radii - y
            viol_t = y + radii - h
            
            # Log-sum-exp approximation of max(0, v)^2
            def smooth_quad_max(v, beta):
                """Smooth approximation of max(0, v)^2 using log-sum-exp."""
                # max(0,v)^2 ≈ (1/beta) * log(1 + exp(2*beta*v)) for v > 0, ~0 for v < 0
                # But we want it to be 0 for v <= 0, so use:
                # f(v) = (1/beta) * log(1 + exp(2*beta*v)) - (1/beta)*log(2) approximately
                # Simpler: use softplus squared
                softplus = np.log1p(np.exp(2.0 * beta * v)) / beta
                # This is smooth and ~0 for v<<0, ~v for v>>0
                return softplus**2
            
            def smooth_quad_max_grad(v, beta):
                """Gradient of smooth_quad_max w.r.t. v."""
                softplus = np.log1p(np.exp(2.0 * beta * v)) / beta
                # d/dv of softplus = sigmoid(2*beta*v)
                sig = 1.0 / (1.0 + np.exp(-2.0 * beta * v))
                return 2.0 * softplus * sig
            
            penalty_b = 1000.0
            obj += penalty_b * (np.sum(smooth_quad_max(viol_l, beta_b)) + 
                                np.sum(smooth_quad_max(viol_r, beta_b)) +
                                np.sum(smooth_quad_max(viol_b, beta_b)) +
                                np.sum(smooth_quad_max(viol_t, beta_b)))
            
            gl = smooth_quad_max_grad(viol_l, beta_b)
            gr = smooth_quad_max_grad(viol_r, beta_b)
            gb = smooth_quad_max_grad(viol_b, beta_b)
            gt = smooth_quad_max_grad(viol_t, beta_b)
            
            for i in range(N):
                grad[1 + 3*i] += penalty_b * (-gl[i] + gr[i])
                grad[1 + 3*i + 1] += penalty_b * (-gb[i] + gt[i])
                grad[1 + 3*i + 2] += penalty_b * (gl[i] + gr[i] + gb[i] + gt[i])
            
            # Overlap penalty using smooth log-sum-exp
            idx_i, idx_j = np.triu_indices(N, k=1)
            dx = x[idx_i] - x[idx_j]
            dy = y[idx_i] - y[idx_j]
            dist_sq = dx*dx + dy*dy
            min_dist = radii[idx_i] + radii[idx_j]
            
            beta_o = 30.0  # sharpness for overlap
            # overlap_amount = min_dist^2 - dist_sq
            # We want smooth max(0, overlap_amount)^2
            overlap_amount = min_dist**2 - dist_sq
            
            # Use softplus-based smoothing
            softplus_o = np.log1p(np.exp(2.0 * beta_o * overlap_amount)) / beta_o
            penalty_o = 5000.0
            obj += penalty_o * np.sum(softplus_o**2)
            
            # Gradient of softplus_o w.r.t. overlap_amount
            sig_o = 1.0 / (1.0 + np.exp(-2.0 * beta_o * overlap_amount))
            d_softplus_d_overlap = sig_o
            
            # d(overlap_amount)/dx_i = -2*dx, d/dx_j = 2*dx
            # d(overlap_amount)/dr_i = 2*min_dist, d/dr_j = 2*min_dist
            
            for k in range(len(idx_i)):
                i = idx_i[k]
                j = idx_j[k]
                so = softplus_o[k]
                ds = d_softplus_d_overlap[k]
                
                coeff = 2.0 * penalty_o * so * ds
                
                grad[1 + 3*i] += coeff * (-2.0 * dx[k])
                grad[1 + 3*j] += coeff * (2.0 * dx[k])
                grad[1 + 3*i + 1] += coeff * (-2.0 * dy[k])
                grad[1 + 3*j + 1] += coeff * (2.0 * dy[k])
                grad[1 + 3*i + 2] += coeff * (2.0 * min_dist[k])
                grad[1 + 3*j + 2] += coeff * (2.0 * min_dist[k])
            
            return obj, grad
        
        def objective(s):
            obj, _ = objective_and_grad(s)
            return obj
        
        def gradient(s):
            _, grad = objective_and_grad(s)
            return grad
        
        # Use scipy minimize with L-BFGS-B
        bounds = [(0.1, 0.9)]
        for i in range(N):
            bounds.append((0, 2.0))  # x
            bounds.append((0, 2.0))  # y
            bounds.append((1e-4, 0.5))  # r
        
        try:
            result = minimize(
                objective,
                state,
                jac=gradient,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
            )
            if result.success or result.fun < objective(state):
                return result.x
        except Exception:
            pass
        
        return state
>>>>>>> REPLACE

