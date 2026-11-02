import numpy as np
from scipy.optimize import minimize, linprog
from scipy.spatial import Delaunay
import time

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    def check_time():
        return time.time() > deadline

    def pack_hexagonal(rows):
        """
        Generate a hexagonal packing configuration.
        Returns: w, h, r, xs, ys
        """
        if sum(rows) != N:
            return None

        pts = []
        for k, n_circles in enumerate(rows):
            y_k = k * np.sqrt(3)
            offset = 1.0 if k % 2 == 1 else 0.0
            for i in range(n_circles):
                x_k = i * 2.0 + offset
                pts.append((x_k, y_k))
        
        pts = np.array(pts)
        if len(pts) != N:
            return None

        center = pts.mean(axis=0)
        pts -= center
        
        xs = pts[:, 0]
        ys = pts[:, 1]
        w_centers = xs.max() - xs.min()
        h_centers = ys.max() - ys.min()
        
        w_total = w_centers + 2.0
        h_total = h_centers + 2.0
        
        scale = 2.0 / (w_total + h_total)
        r = scale
        w = scale * w_total
        h = scale * h_total
        
        xs_final = pts[:, 0] * scale + w/2.0
        ys_final = pts[:, 1] * scale + h/2.0
        
        return w, h, r, xs_final, ys_final

    def pack_grid(rows, cols):
        """
        Generate a grid packing configuration.
        """
        pts = []
        count = 0
        for i in range(cols):
            for j in range(rows):
                if count >= N:
                    break
                pts.append((i * 2.0, j * 2.0))
                count += 1
            if count >= N:
                break
        
        pts = np.array(pts[:N])
        
        center = pts.mean(axis=0)
        pts -= center
        
        xs = pts[:, 0]
        ys = pts[:, 1]
        w_centers = xs.max() - xs.min()
        h_centers = ys.max() - ys.min()
        
        w_total = w_centers + 2.0
        h_total = h_centers + 2.0
        
        scale = 2.0 / (w_total + h_total)
        r = scale
        w = scale * w_total
        h = scale * h_total
        
        xs_final = pts[:, 0] * scale + w/2.0
        ys_final = pts[:, 1] * scale + h/2.0
        
        return w, h, r, xs_final, ys_final

    def get_delaunay_edges(points, w, h):
        """
        Get Delaunay edges from points plus boundary corners.
        Returns set of (i, j) pairs for circle-circle edges.
        """
        # Add 4 corners of the rectangle to help with boundary Delaunay
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]])
        all_pts = np.vstack([points, corners])
        
        try:
            tri = Delaunay(all_pts)
        except Exception:
            # Fallback: use all pairs
            return [(i, j) for i in range(N) for j in range(i+1, N)]
        
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i+1, 3):
                    a, b = simplex[i], simplex[j]
                    if a < N and b < N:
                        edges.add((min(a,b), max(a,b)))
        
        # Also add edges to boundary corners to capture boundary proximity
        # But we only care about circle-circle edges for the LP
        return list(edges)

    def solve_lp_for_radii(points, w, h):
        """
        Solve LP to maximize sum of radii given fixed centers.
        Variables: r_0, ..., r_{N-1}
        Maximize: sum(r_i)
        Subject to:
          r_i + r_j <= dist(i,j) for all pairs (i,j)
          r_i <= min(x_i, w-x_i, y_i, h-y_i) for each i
          r_i >= 0
        """
        # Get Delaunay edges for efficiency
        edges = get_delaunay_edges(points, w, h)
        
        # Also need to consider all pairs that are close (within some threshold)
        # to avoid missing tight constraints
        # For now, use Delaunay edges + a few extra close pairs
        
        # Build constraint matrix
        # Each row: coefficients for r_0, ..., r_{N-1}
        # r_i + r_j <= dist(i,j)
        
        constraints = []
        rhs = []
        
        # Circle-circle constraints from Delaunay edges
        for (i, j) in edges:
            row = np.zeros(N)
            row[i] = 1.0
            row[j] = 1.0
            dist = np.sqrt((points[i,0]-points[j,0])**2 + (points[i,1]-points[j,1])**2)
            constraints.append(row)
            rhs.append(dist)
        
        # Also add constraints for all pairs within a certain distance
        # to be safe (Delaunay may miss some tight constraints)
        for i in range(N):
            for j in range(i+1, N):
                if (i,j) in edges or (j,i) in edges:
                    continue
                dist = np.sqrt((points[i,0]-points[j,0])**2 + (points[i,1]-points[j,1])**2)
                # Only add if this constraint could be tight
                # Heuristic: if dist < 2 * avg_radius_estimate
                # For now, skip non-Delaunay edges for speed
                # But add all pairs for correctness in small N
                # Actually, for N=21, all pairs is only 210, which is fine
                row = np.zeros(N)
                row[i] = 1.0
                row[j] = 1.0
                constraints.append(row)
                rhs.append(dist)
        
        # Boundary constraints: r_i <= min(x_i, w-x_i, y_i, h-y_i)
        for i in range(N):
            row = np.zeros(N)
            row[i] = 1.0
            boundary_dist = min(points[i,0], w-points[i,0], points[i,1], h-points[i,1])
            constraints.append(row)
            rhs.append(boundary_dist)
        
        if not constraints:
            return np.zeros(N), None
        
        A_ub = np.array(constraints)
        b_ub = np.array(rhs)
        
        # Maximize sum(r) => minimize -sum(r)
        c = -np.ones(N)
        
        bounds = [(0, None)] * N
        
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                radii = np.clip(res.x, 0, None)
                duals = res.ineqlin.marginals if hasattr(res, 'ineqlin') else None
                # Get shadow prices from the constraint matrix
                # For HiGHS, duals are in res.ineqlin.marginals
                if duals is not None:
                    return radii, duals
                return radii, None
            else:
                return np.zeros(N), None
        except Exception:
            return np.zeros(N), None

    def refine_with_lp(points, w, h, max_iters=50, lr_base=0.01):
        """
        Rigid-body assembly: optimize centers via gradient descent
        using LP dual values as forces, while solving LP for radii.
        """
        points = points.copy()
        best_sum_r = 0
        best_points = points.copy()
        
        for it in range(max_iters):
            if check_time():
                break
            
            # Solve LP for radii given current centers
            radii, duals = solve_lp_for_radii(points, w, h)
            sum_r = np.sum(radii)
            
            if sum_r > best_sum_r:
                best_sum_r = sum_r
                best_points = points.copy()
            
            # duals will be used below if available
            
            # Compute forces from dual values
            # The dual of constraint r_i + r_j <= dist(i,j) gives force magnitude
            # Force on i is in direction away from j, magnitude = dual_value
            # Force on j is in direction away from i, magnitude = dual_value
            
            # Reconstruct which constraints correspond to which pairs
            # This requires tracking the constraint indices
            # For simplicity, use a different approach: 
            # Compute gradient of sum_r w.r.t. positions directly
            
            # Alternative: use the LP dual to compute forces
            # Let's redo this more carefully
            
            # Get the edges again
            edges = get_delaunay_edges(points, w, h)
            edge_set = set(edges)
            
            # Build list of all constraints in the same order as LP
            all_constraints = []
            for (i, j) in edges:
                all_constraints.append(('circle', i, j))
            for i in range(N):
                for j in range(i+1, N):
                    if (i,j) not in edge_set and (j,i) not in edge_set:
                        all_constraints.append(('circle', i, j))
            for i in range(N):
                all_constraints.append(('boundary', i, -1))
            
            # Now compute forces
            forces = np.zeros((N, 2))
            
            if duals is not None and len(duals) == len(all_constraints):
                for idx, (ctype, i, j) in enumerate(all_constraints):
                    dual_val = abs(duals[idx])
                    if dual_val < 1e-10:
                        continue
                    
                    if ctype == 'circle':
                        dx = points[i,0] - points[j,0]
                        dy = points[i,1] - points[j,1]
                        dist = np.sqrt(dx*dx + dy*dy)
                        if dist < 1e-10:
                            continue
                        fx = dual_val * dx / dist
                        fy = dual_val * dy / dist
                        forces[i, 0] += fx
                        forces[i, 1] += fy
                        forces[j, 0] -= fx
                        forces[j, 1] -= fy
                    elif ctype == 'boundary':
                        # Boundary constraint: r_i <= min(x_i, w-x_i, y_i, h-y_i)
                        # The binding side determines the force direction
                        x, y = points[i,0], points[i,1]
                        left_dist = x
                        right_dist = w - x
                        bottom_dist = y
                        top_dist = h - y
                        
                        min_dist = min(left_dist, right_dist, bottom_dist, top_dist)
                        
                        if abs(min_dist - left_dist) < 1e-10:
                            forces[i, 0] += dual_val
                        elif abs(min_dist - right_dist) < 1e-10:
                            forces[i, 0] -= dual_val
                        elif abs(min_dist - bottom_dist) < 1e-10:
                            forces[i, 1] += dual_val
                        elif abs(min_dist - top_dist) < 1e-10:
                            forces[i, 1] -= dual_val
            
            # Update positions with adaptive learning rate
            lr = lr_base / (1 + it * 0.1)
            points += lr * forces
            
            # Project back to valid region
            # Ensure points stay within [r_i, w-r_i] x [r_i, h-r_i]
            # Use current radii estimate
            for i in range(N):
                r_est = max(radii[i], 0.01) if len(radii) > i else 0.01
                points[i, 0] = np.clip(points[i, 0], r_est, w - r_est)
                points[i, 1] = np.clip(points[i, 1], r_est, h - r_est)
        
        return best_points, best_sum_r

    def refine_slsqp(w, h, r, xs, ys, is_polish=False):
        """
        Refine the packing using SLSQP.
        Variables: w, r, xs, ys. h = 2 - w.
        """
        if check_time():
            return w, h, r, xs, ys

        v0 = np.zeros(2 + 2*N)
        v0[0] = w
        v0[1] = r
        v0[2:2+N] = xs
        v0[2+N:2+2*N] = ys

        def objective(v):
            w_val = v[0]
            r_val = v[1]
            h_val = 2.0 - w_val
            if w_val <= 0.1 or h_val <= 0.1 or r_val <= 0:
                return 1e6
            return -r_val

        def constraints(v):
            w_val = v[0]
            r_val = v[1]
            h_val = 2.0 - w_val
            xs = v[2:2+N]
            ys = v[2+N:2+2*N]
            
            cons = []
            
            cons.append({'type': 'ineq', 'fun': lambda v: v[0] - 0.2})
            cons.append({'type': 'ineq', 'fun': lambda v: 2.0 - v[0] - 0.2})
            
            for i in range(N):
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[2+i] - v[1]})
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[0] - v[2+i] - v[1]})
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[2+N+i] - v[1]})
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: (2.0 - v[0]) - v[2+N+i] - v[1]})
                
            for i in range(N):
                for j in range(i+1, N):
                    cons.append({
                        'type': 'ineq', 
                        'fun': lambda v, i=i, j=j: ((v[2+i]-v[2+j])**2 + (v[2+N+i]-v[2+N+j])**2) - 4*(v[1]**2)
                    })
            return cons

        bounds = [(0.2, 1.8), (0.01, 1.0)] + [(0.0, 2.0)] * (2*N)

        try:
            if is_polish:
                options = {'maxiter': 200, 'ftol': 1e-12, 'disp': False}
            else:
                options = {'maxiter': 150, 'ftol': 1e-9, 'disp': False}

            res = minimize(objective, v0, method='SLSQP', bounds=bounds, constraints=constraints, 
                           options=options)
            
            if res.success or res.fun < -r:
                w_res = res.x[0]
                r_res = res.x[1]
                h_res = 2.0 - w_res
                xs_res = res.x[2:2+N]
                ys_res = res.x[2+N:2+2*N]
                
                valid = True
                if w_res <= 0 or h_res <= 0 or r_res <= 0:
                    valid = False
                else:
                    for i in range(N):
                        if xs_res[i] < r_res - 1e-7 or xs_res[i] > w_res - r_res + 1e-7:
                            valid = False
                            break
                        if ys_res[i] < r_res - 1e-7 or ys_res[i] > h_res - r_res + 1e-7:
                            valid = False
                            break
                    if valid:
                        for i in range(N):
                            for j in range(i+1, N):
                                dist_sq = (xs_res[i]-xs_res[j])**2 + (ys_res[i]-ys_res[j])**2
                                if dist_sq < 4*(r_res**2) - 1e-7:
                                    valid = False
                                    break
                            if not valid:
                                break
                
                if valid:
                    return w_res, h_res, r_res, xs_res, ys_res
        except Exception:
            pass
            
        return w, h, r, xs, ys

    def evaluate_packing(points, w, h):
        """
        Given points and rectangle, solve LP for radii and return sum of radii.
        """
        radii, _ = solve_lp_for_radii(points, w, h)
        return np.sum(radii), radii

    best_sum_r = 0
    best_config = None

    # List of initial configurations
    configs = []
    
    # Hexagonal variations
    configs.append(pack_hexagonal([7, 7, 7]))
    configs.append(pack_hexagonal([4, 5, 4, 4, 4]))
    configs.append(pack_hexagonal([3, 4, 4, 4, 4, 2]))
    configs.append(pack_hexagonal([5, 5, 5, 3, 3]))
    configs.append(pack_hexagonal([6, 6, 5, 4]))
    
    # Grid variations
    configs.append(pack_grid(5, 5))
    configs.append(pack_grid(3, 7))
    configs.append(pack_grid(4, 6))
    configs.append(pack_grid(7, 3))
    configs.append(pack_grid(6, 4))
    
    # Random restarts
    for _ in range(6):
        if check_time():
            break
        w = 1.0
        h = 1.0
        r = 0.08
        xs = np.random.uniform(r, w-r, N)
        ys = np.random.uniform(r, h-r, N)
        configs.append((w, h, r, xs, ys))

    valid_configs = [c for c in configs if c is not None]

    # First pass: LP-based refinement for each config
    for cfg in valid_configs:
        if check_time():
            break
        w, h, r, xs, ys = cfg
        points = np.column_stack([xs, ys])
        
        # Try LP-based center optimization
        best_pts, sum_r = refine_with_lp(points, w, h, max_iters=30, lr_base=0.005)
        
        if sum_r > best_sum_r:
            best_sum_r = sum_r
            best_config = (w, h, best_pts, sum_r)
        
        # Also try SLSQP on the equal-radius problem for comparison
        w_new, h_new, r_new, xs_new, ys_new = refine_slsqp(w, h, r, xs, ys, is_polish=False)
        sum_r_slqp = N * r_new
        if sum_r_slqp > best_sum_r:
            best_sum_r = sum_r_slqp
            best_config = (w_new, h_new, np.column_stack([xs_new, ys_new]), sum_r_slqp, 'slqp', r_new)

    # Second pass: Polish the best result
    if best_config is not None and not check_time():
        if len(best_config) == 5 and best_config[4] == 'slqp':
            w, h, points, _, _, r_val = best_config
            # Polish with SLSQP
            w_pol, h_pol, r_pol, xs_pol, ys_pol = refine_slsqp(w, h, r_val, points[:,0], points[:,1], is_polish=True)
            sum_r_pol = N * r_pol
            if sum_r_pol > best_sum_r:
                best_sum_r = sum_r_pol
                best_config = (w_pol, h_pol, np.column_stack([xs_pol, ys_pol]), sum_r_pol, 'slqp', r_pol)
        else:
            w, h, points, sum_r = best_config
            # Polish with more LP iterations
            best_pts, sum_r_pol = refine_with_lp(points, w, h, max_iters=50, lr_base=0.002)
            if sum_r_pol > best_sum_r:
                best_sum_r = sum_r_pol
                best_config = (w, h, best_pts, sum_r_pol)

    # Final: convert to equal-radius output for the interface
    if best_config is None:
        # Fallback
        if valid_configs:
            w, h, r, xs, ys = valid_configs[0]
            return np.column_stack([xs, ys, np.full(N, r)])
        else:
            side = int(np.ceil(np.sqrt(N)))
            cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
            return np.array(cells[:N])

    # Extract final configuration
    if len(best_config) == 5 and best_config[4] == 'slqp':
        w, h, points, _, _, r_val = best_config
        # Use equal radius
        r_final = r_val * (1 - 1e-6)
        return np.column_stack([points[:,0], points[:,1], np.full(N, r_final)])
    else:
        w, h, points, sum_r = best_config
        # For variable radii, we need to output equal radius
        # Use the minimum radius from LP as the safe equal radius
        radii, _ = solve_lp_for_radii(points, w, h)
        r_min = np.min(radii)
        r_final = r_min * (1 - 1e-6)
        return np.column_stack([points[:,0], points[:,1], np.full(N, r_final)])
