"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii."""
import numpy as np
import time
import random
from scipy.optimize import minimize, linprog
from scipy.spatial import Delaunay

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    def softplus(x):
        return np.where(x > 20, x, np.log1p(np.exp(x)))

    def sigmoid(x):
        return np.where(x > 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

    def solve_lp_radii(xs, ys, w, h):
        """
        Solve LP for radii given fixed centers.
        Maximize sum(r) subject to:
        r_i + r_j <= dist(i,j) for Delaunay edges
        r_i <= min(x_i, w-x_i, y_i, h-y_i)
        r_i >= 0
        """
        # Compute Delaunay triangulation
        points = np.column_stack((xs, ys))
        try:
            tri = Delaunay(points)
        except Exception:
            # Fallback if Delaunay fails (e.g., points collinear)
            # Just use boundary constraints
            A_ub = np.zeros((0, N))
            b_ub = np.zeros(0)
            # Boundary constraints: r_i <= limit_i
            limits = np.minimum(np.minimum(xs, w - xs), np.minimum(ys, h - ys))
            A_bound = np.eye(N)
            b_bound = limits
            c = -np.ones(N)
            res = linprog(c, A_ub=np.vstack([A_ub, A_bound]), b_ub=np.concatenate([b_ub, b_bound]), bounds=(0, None), method='highs')
            if res.success:
                return res.x
            else:
                return np.zeros(N)

        # Get edges from Delaunay
        edges = set()
        for simplex in tri.simplices:
            edges.add(tuple(sorted((simplex[0], simplex[1]))))
            edges.add(tuple(sorted((simplex[1], simplex[2]))))
            edges.add(tuple(sorted((simplex[0], simplex[2]))))
        
        # Build constraint matrix for pairwise distances
        # r_i + r_j <= d_ij
        A_list = []
        b_list = []
        
        for i, j in edges:
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx*dx + dy*dy)
            row = np.zeros(N)
            row[i] = 1
            row[j] = 1
            A_list.append(row)
            b_list.append(dist)
        
        # Boundary constraints: r_i <= limit_i
        limits = np.minimum(np.minimum(xs, w - xs), np.minimum(ys, h - ys))
        # Ensure limits are non-negative, else infeasible
        if np.any(limits < 0):
             return np.zeros(N)
             
        A_bound = np.eye(N)
        
        A_ub = np.vstack([A_list, A_bound]) if A_list else A_bound
        b_ub = np.concatenate([b_list, limits]) if A_list else limits
        
        c = -np.ones(N)  # Maximize sum(r) -> Minimize -sum(r)
        
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=(0, None), method='highs')
            if res.success:
                return res.x
            else:
                # If LP fails, return heuristic radii
                return np.maximum(0, limits * 0.9)
        except Exception:
            return np.maximum(0, limits * 0.9)

    def make_initial_centers(w, h, n_circles, seed):
        """Create a hexagonal-like packing initialization for centers."""
        rng = np.random.RandomState(seed)
        
        best_xs = None
        best_ys = None
        best_score = -1
        
        # Try different grid configurations
        for rows in range(1, n_circles + 1):
            cols = (n_circles + rows - 1) // rows
            
            xs = np.zeros(n_circles)
            ys = np.zeros(n_circles)
            
            for i in range(n_circles):
                row = i // cols
                col = i % cols
                
                x = (col + 0.5) * (w / cols)
                y = (row + 0.5) * (h / rows)
                
                if row % 2 == 1:
                    x += 0.5 * (w / cols)
                    
                x = np.clip(x, 0.05 * w, 0.95 * w)
                y = np.clip(y, 0.05 * h, 0.95 * h)
                
                xs[i] = x
                ys[i] = y
            
            # Quick validity check for centers (just bounds)
            valid = True
            for i in range(n_circles):
                if xs[i] < 0 or xs[i] > w or ys[i] < 0 or ys[i] > h:
                    valid = False
                    break
            
            if valid:
                # Solve LP to get radii and sum
                rs = solve_lp_radii(xs, ys, w, h)
                score = np.sum(rs)
                if score > best_score:
                    best_score = score
                    best_xs = xs.copy()
                    best_ys = ys.copy()
        
        if best_xs is None:
            side = int(np.ceil(np.sqrt(n_circles)))
            best_xs = np.zeros(n_circles)
            best_ys = np.zeros(n_circles)
            for i in range(n_circles):
                row = i // side
                col = i % side
                best_xs[i] = (col + 0.5) / side * w
                best_ys[i] = (row + 0.5) / side * h
        
        # Add small random perturbation
        if seed > 0:
            best_xs += rng.randn(n_circles) * 0.01 * w
            best_ys += rng.randn(n_circles) * 0.01 * h
            best_xs = np.clip(best_xs, 0.01, w - 0.01)
            best_ys = np.clip(best_ys, 0.01, h - 0.01)
            
        return best_xs, best_ys

    def objective_and_grad(params, w, h, penalty_weight):
        """
        Objective: Minimize -sum(r) + penalty * (boundary_violation + overlap_violation)
        params: [x1, y1, r1, x2, y2, r2, ...]
        """
        circles = params
        obj_val = 0.0
        grad = np.zeros_like(params)
        
        sum_r = 0.0
        for i in range(N):
            xi = circles[3*i]
            yi = circles[3*i+1]
            ri = circles[3*i+2]
            if ri < 1e-5:
                ri = 1e-5
            sum_r += ri
            obj_val -= ri
            grad[3*i+2] -= 1.0

            # Boundary penalties
            v_left = ri - xi
            if v_left > -20:
                sp_grad = sigmoid(v_left)
                obj_val += penalty_weight * softplus(v_left)
                grad[3*i] -= penalty_weight * sp_grad
                grad[3*i+2] += penalty_weight * sp_grad
            
            v_right = xi + ri - w
            if v_right > -20:
                sp_grad = sigmoid(v_right)
                obj_val += penalty_weight * softplus(v_right)
                grad[3*i] += penalty_weight * sp_grad
                grad[3*i+2] += penalty_weight * sp_grad
                
            v_bot = ri - yi
            if v_bot > -20:
                sp_grad = sigmoid(v_bot)
                obj_val += penalty_weight * softplus(v_bot)
                grad[3*i+1] -= penalty_weight * sp_grad
                grad[3*i+2] += penalty_weight * sp_grad

            v_top = yi + ri - h
            if v_top > -20:
                sp_grad = sigmoid(v_top)
                obj_val += penalty_weight * softplus(v_top)
                grad[3*i+1] += penalty_weight * sp_grad
                grad[3*i+2] += penalty_weight * sp_grad

        # Overlap penalties
        for i in range(N):
            xi = circles[3*i]
            yi = circles[3*i+1]
            ri = circles[3*i+2]
            for j in range(i+1, N):
                xj = circles[3*j]
                yj = circles[3*j+1]
                rj = circles[3*j+2]
                
                dx = xi - xj
                dy = yi - yj
                dist_sq = dx*dx + dy*dy
                min_dist = ri + rj
                
                if dist_sq < min_dist * min_dist:
                    dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 1e-12
                    v = min_dist - dist
                    if v > -20:
                        sp_grad = sigmoid(v)
                        obj_val += penalty_weight * softplus(v)
                        
                        coeff = penalty_weight * sp_grad / dist
                        grad[3*i] -= coeff * dx
                        grad[3*i+1] -= coeff * dy
                        grad[3*i+2] += penalty_weight * sp_grad
                        
                        grad[3*j] += coeff * dx
                        grad[3*j+1] += coeff * dy
                        grad[3*j+2] += penalty_weight * sp_grad

        return obj_val, grad

    def is_valid(params, w, h):
        """Check if configuration is valid."""
        circles = params
        for i in range(N):
            xi = circles[3*i]
            yi = circles[3*i+1]
            ri = circles[3*i+2]
            if ri < 1e-6:
                return False
            if xi - ri < -1e-5 or xi + ri > w + 1e-5:
                return False
            if yi - ri < -1e-5 or yi + ri > h + 1e-5:
                return False
        
        for i in range(N):
            xi = circles[3*i]
            yi = circles[3*i+1]
            ri = circles[3*i+2]
            for j in range(i+1, N):
                xj = circles[3*j]
                yj = circles[3*j+1]
                rj = circles[3*j+2]
                dist_sq = (xi - xj)**2 + (yi - yj)**2
                if dist_sq < (ri + rj - 1e-5)**2:
                    return False
        return True

    def get_value(params, w, h):
        """Return sum of radii if valid, else -inf."""
        if not is_valid(params, w, h):
            return -np.inf
        return np.sum(params[2::3])

    best_value = -np.inf
    best_params = None
    best_w = 1.0

    random.seed(42)
    np.random.seed(42)

    # Scan aspect ratios
    w_list = np.linspace(0.7, 1.3, 13)
    
    for w in w_list:
        if time.time() > deadline - 10:
            break
        h = 2.0 - w
        
        # Initializations
        inits = []
        # 1. Hexagonal
        xs0, ys0 = make_initial_centers(w, h, N, 0)
        # 2. Randomized grids
        for seed in [1, 2, 3]:
             xs_s, ys_s = make_initial_centers(w, h, N, seed)
             inits.append((xs_s, ys_s))
        inits.insert(0, (xs0, ys0))
        
        for xs_init, ys_init in inits:
            if time.time() > deadline - 5:
                break
                
            # Solve LP for initial radii
            rs_init = solve_lp_radii(xs_init, ys_init, w, h)
            
            # Construct initial params [x, y, r]
            current_params = np.zeros(3*N)
            current_params[0::3] = xs_init
            current_params[1::3] = ys_init
            current_params[2::3] = rs_init
            
            # Penalty Annealing
            penalties = [50, 200, 1000]
            
            for p in penalties:
                if time.time() > deadline - 2:
                    break
                    
                def obj_func(x):
                    return objective_and_grad(x, w, h, p)
                
                bounds = [(0, w)] * N + [(0, h)] * N + [(0, min(w, h)/2)] * N
                
                try:
                    result = minimize(
                        obj_func,
                        current_params,
                        method='L-BFGS-B',
                        jac=True,
                        bounds=bounds,
                        options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-10}
                    )
                    current_params = result.x
                except Exception:
                    pass
            
            # Check validity and update best
            val = get_value(current_params, w, h)
            if val > best_value:
                best_value = val
                best_params = current_params.copy()
                best_w = w

    # If we have a valid best, try to refine with more iterations
    if best_params is not None and time.time() < deadline - 10:
        for _ in range(3):
            if time.time() > deadline - 2:
                break
            # Try slightly different w around best_w
            w_new = best_w + np.random.uniform(-0.05, 0.05)
            w_new = np.clip(w_new, 0.5, 1.5)
            h_new = 2.0 - w_new
            
            # Rescale centers to new aspect ratio
            xs = best_params[0::3]
            ys = best_params[1::3]
            rs = best_params[2::3]
            
            # Simple rescale
            xs = xs * (w_new / best_w)
            ys = ys * (h_new / (2.0 - best_w))
            
            # Re-solve LP for new aspect ratio
            rs_new = solve_lp_radii(xs, ys, w_new, h_new)
            
            current_params = np.zeros(3*N)
            current_params[0::3] = xs
            current_params[1::3] = ys
            current_params[2::3] = rs_new
            
            penalties = [100, 500, 1000]
            for p in penalties:
                if time.time() > deadline - 2:
                    break
                def obj_func(x):
                    return objective_and_grad(x, w_new, h_new, p)
                
                bounds = [(0, w_new)] * N + [(0, h_new)] * N + [(0, min(w_new, h_new)/2)] * N
                
                try:
                    result = minimize(
                        obj_func,
                        current_params,
                        method='L-BFGS-B',
                        jac=True,
                        bounds=bounds,
                        options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
                    )
                    current_params = result.x
                except Exception:
                    pass
            
            val = get_value(current_params, w_new, h_new)
            if val > best_value:
                best_value = val
                best_params = current_params.copy()
                best_w = w_new

    # Fallback
    if best_params is None or best_value < 0:
        w = 1.0
        h = 1.0
        side = 5
        result = np.zeros((N, 3))
        for i in range(N):
            row = i // side
            col = i % side
            result[i, 0] = (col + 0.5) / side
            result[i, 1] = (row + 0.5) / side
            result[i, 2] = 0.48 / side
        return result

    # Extract final result
    w_final = best_w
    h_final = 2.0 - w_final
    circles = best_params
    
    result = np.zeros((N, 3))
    for i in range(N):
        result[i, 0] = circles[3*i]
        result[i, 1] = circles[3*i+1]
        result[i, 2] = circles[3*i+2]
    
    # Final verification
    xs = result[:, 0]
    ys = result[:, 1]
    rs = result[:, 2]
    min_x = np.min(xs - rs)
    max_x = np.max(xs + rs)
    min_y = np.min(ys - rs)
    max_y = np.max(ys + rs)
    bbox_w = max_x - min_x
    bbox_h = max_y - min_y
    bbox_perim = 2 * (bbox_w + bbox_h)
    
    if bbox_perim > 4.0 + 1e-5:
        scale = 4.0 / bbox_perim
        cx = (min_x + max_x) / 2
        cy = (min_y + max_y) / 2
        for i in range(N):
            result[i, 0] = cx + (result[i, 0] - cx) * scale
            result[i, 1] = cy + (result[i, 1] - cy) * scale
            result[i, 2] = result[i, 2] * scale
    
    return result
