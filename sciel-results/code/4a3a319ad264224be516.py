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

    W_BOUNDARY = 100.0
    W_OVERLAP = 50.0

    def make_initial(w, h, n_circles):
        """Create a hexagonal-like packing initialization."""
        if n_circles == 0:
            return np.zeros(0)
        
        best_score = -1
        best_arr = None
        
        for rows in range(1, n_circles + 1):
            cols = (n_circles + rows - 1) // rows
            xs = np.zeros(n_circles)
            ys = np.zeros(n_circles)
            rs = np.zeros(n_circles)
            
            for i in range(n_circles):
                row = i // cols
                col = i % cols
                x = (col + 0.5) * (w / cols)
                y = (row + 0.5) * (h / rows)
                if row % 2 == 1:
                    x += 0.5 * (w / cols)
                x = np.clip(x, 0.1 * w, 0.9 * w)
                y = np.clip(y, 0.1 * h, 0.9 * h)
                cell_w = w / cols
                cell_h = h / rows
                r = 0.45 * min(cell_w, cell_h)
                xs[i] = x
                ys[i] = y
                rs[i] = r
            
            valid = True
            for i in range(n_circles):
                if xs[i] < rs[i] or xs[i] > w - rs[i]:
                    valid = False
                    break
                if ys[i] < rs[i] or ys[i] > h - rs[i]:
                    valid = False
                    break
            if valid:
                for i in range(n_circles):
                    for j in range(i + 1, n_circles):
                        dist = np.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)
                        if dist < rs[i] + rs[j]:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                score = np.sum(rs)
                if score > best_score:
                    best_score = score
                    best_arr = np.concatenate([xs, ys, rs])
        
        if best_arr is None:
            side = int(np.ceil(np.sqrt(n_circles)))
            xs = np.zeros(n_circles)
            ys = np.zeros(n_circles)
            rs = np.zeros(n_circles)
            for i in range(n_circles):
                row = i // side
                col = i % side
                xs[i] = (col + 0.5) / side * w
                ys[i] = (row + 0.5) / side * h
                rs[i] = 0.45 / side * min(w, h)
            best_arr = np.concatenate([xs, ys, rs])
        
        return best_arr

    def objective_and_grad(params):
        """Penalty-based objective with analytic gradient."""
        w = params[0]
        w_clipped = np.clip(w, 0.3, 1.7)
        h = 2.0 - w_clipped
        
        circles = params[1:]
        obj_val = 0.0
        grad = np.zeros_like(params)
        
        for i in range(N):
            xi = circles[3*i]
            yi = circles[3*i+1]
            ri = circles[3*i+2]
            if ri < 1e-5:
                ri = 1e-5
            obj_val -= ri
            grad[3*i+2] -= 1.0

            v_left = ri - xi
            if v_left > -20:
                sp_grad = sigmoid(v_left)
                obj_val += W_BOUNDARY * softplus(v_left)
                grad[3*i] -= W_BOUNDARY * sp_grad
                grad[3*i+2] += W_BOUNDARY * sp_grad
            
            v_right = xi + ri - w_clipped
            if v_right > -20:
                sp_grad = sigmoid(v_right)
                obj_val += W_BOUNDARY * softplus(v_right)
                grad[3*i] += W_BOUNDARY * sp_grad
                grad[3*i+2] += W_BOUNDARY * sp_grad
                
            v_bot = ri - yi
            if v_bot > -20:
                sp_grad = sigmoid(v_bot)
                obj_val += W_BOUNDARY * softplus(v_bot)
                grad[3*i+1] -= W_BOUNDARY * sp_grad
                grad[3*i+2] += W_BOUNDARY * sp_grad

            v_top = yi + ri - h
            if v_top > -20:
                sp_grad = sigmoid(v_top)
                obj_val += W_BOUNDARY * softplus(v_top)
                grad[3*i+1] += W_BOUNDARY * sp_grad
                grad[3*i+2] += W_BOUNDARY * sp_grad

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
                        obj_val += W_OVERLAP * softplus(v)
                        
                        coeff = W_OVERLAP * sp_grad / dist
                        grad[3*i] -= coeff * dx
                        grad[3*i+1] -= coeff * dy
                        grad[3*i+2] += W_OVERLAP * sp_grad
                        
                        grad[3*j] += coeff * dx
                        grad[3*j+1] += coeff * dy
                        grad[3*j+2] += W_OVERLAP * sp_grad

        grad_w = 0.0
        for i in range(N):
            yi = circles[3*i+1]
            ri = circles[3*i+2]
            v_top = yi + ri - h
            if v_top > -20:
                grad_w += W_BOUNDARY * sigmoid(v_top)
        grad[0] = grad_w

        return obj_val, grad

    def is_valid(params):
        """Check if configuration is valid."""
        w = np.clip(params[0], 0.1, 1.9)
        h = 2.0 - w
        circles = params[1:]
        
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

    def get_value(params):
        """Return sum of radii if valid, else -inf."""
        if not is_valid(params):
            return -np.inf
        w = np.clip(params[0], 0.1, 1.9)
        h = 2.0 - w
        circles = params[1:]
        xs = circles[0::3]
        ys = circles[1::3]
        rs = circles[2::3]
        min_x = np.min(xs - rs)
        max_x = np.max(xs + rs)
        min_y = np.min(ys - rs)
        max_y = np.max(ys + rs)
        bbox_w = max_x - min_x
        bbox_h = max_y - min_y
        bbox_perim = 2 * (bbox_w + bbox_h)
        if bbox_perim > 4.0 + 1e-5:
            return -np.inf
        return np.sum(rs)

    def gradient_descent(params, lr=0.01, momentum=0.9, max_iter=2000, tol=1e-8):
        """Fast gradient descent with momentum for local optimization."""
        current = params.copy()
        velocity = np.zeros_like(current)
        current_obj, current_grad = objective_and_grad(current)
        
        for it in range(max_iter):
            if time.time() > deadline - 3:
                break
            
            velocity = momentum * velocity - lr * current_grad
            max_vel = 0.1
            velocity = np.clip(velocity, -max_vel, max_vel)
            new_params = current + velocity
            
            new_params[0] = np.clip(new_params[0], 0.3, 1.7)
            for i in range(N):
                new_params[1 + 3*i] = np.clip(new_params[1 + 3*i], -1.0, 3.0)
                new_params[2 + 3*i] = np.clip(new_params[2 + 3*i], -1.0, 3.0)
                new_params[3 + 3*i] = np.clip(new_params[3 + 3*i], 1e-5, 1.0)
            
            new_obj, new_grad = objective_and_grad(new_params)
            
            if new_obj < current_obj - tol or np.all(np.abs(new_grad) < 1e-6):
                current = new_params
                current_obj = new_obj
                current_grad = new_grad
            else:
                lr *= 0.9
                if lr < 1e-6:
                    break
        
        return current

    def delaunay_radius_optimization(xs, ys, w, h, max_iter=500, tol=1e-10):
        """
        Optimize radii given fixed centers using Delaunay triangulation.
        Uses LP to maximize sum of radii subject to non-overlap constraints
        for Delaunay edges plus boundary constraints.
        """
        # Compute Delaunay triangulation
        points = np.column_stack([xs, ys])
        try:
            tri = Delaunay(points)
        except Exception:
            # Fallback: all pairs
            edges = [(i, j) for i in range(N) for j in range(i+1, N)]
            return _solve_radius_lp(xs, ys, w, h, edges, max_iter, tol)
        
        # Extract edges from Delaunay
        edge_set = set()
        for simplex in tri.simplices:
            for i in range(len(simplex)):
                for j in range(i+1, len(simplex)):
                    a, b = simplex[i], simplex[j]
                    edge_set.add((min(a,b), max(a,b)))
        
        # Also add boundary constraints as virtual edges
        # For each circle, distance to each wall
        edges = list(edge_set)
        
        return _solve_radius_lp(xs, ys, w, h, edges, max_iter, tol)

    def _solve_radius_lp(xs, ys, w, h, edges, max_iter=500, tol=1e-10):
        """
        Solve LP: maximize sum(r_i) subject to:
        - r_i + r_j <= d_ij for all edges (i,j)
        - r_i <= min(x_i, w-x_i, y_i, h-y_i) for all i
        - r_i >= 0
        """
        n_vars = N
        # Objective: maximize sum(r_i) -> minimize -sum(r_i)
        c = -np.ones(n_vars)
        
        # Inequality constraints: A_ub @ r <= b_ub
        A_rows = []
        b_vals = []
        
        # Pair constraints from edges
        for (i, j) in edges:
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            d = np.sqrt(dx*dx + dy*dy)
            row = np.zeros(n_vars)
            row[i] = 1.0
            row[j] = 1.0
            A_rows.append(row)
            b_vals.append(d)
        
        # Boundary constraints: r_i <= min(x_i, w-x_i, y_i, h-y_i)
        for i in range(N):
            max_r = min(xs[i], w - xs[i], ys[i], h - ys[i])
            if max_r < 1e-6:
                return None  # Infeasible
            row = np.zeros(n_vars)
            row[i] = 1.0
            A_rows.append(row)
            b_vals.append(max_r)
        
        if not A_rows:
            return np.zeros(N)
        
        A_ub = np.array(A_rows)
        b_ub = np.array(b_vals)
        
        # Bounds: 0 <= r_i <= max_possible
        max_r_bounds = np.array([min(xs[i], w-xs[i], ys[i], h-ys[i]) for i in range(N)])
        bounds = [(0, max_r_bounds[i]) for i in range(N)]
        
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                radii = res.x
                # Verify feasibility
                for (i, j) in edges:
                    d = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                    if radii[i] + radii[j] > d + 1e-8:
                        # Slight violation, scale down
                        scale = d / (radii[i] + radii[j])
                        radii[i] *= scale
                        radii[j] *= scale
                return radii
        except Exception:
            pass
        
        # Fallback: greedy radius assignment
        radii = np.zeros(N)
        for i in range(N):
            max_r = min(xs[i], w-xs[i], ys[i], h-ys[i])
            for (a, b) in edges:
                if a == i or b == i:
                    other = b if a == i else a
                    d = np.sqrt((xs[i]-xs[other])**2 + (ys[i]-ys[other])**2)
                    max_r = min(max_r, d - radii[other])
            radii[i] = max(0, max_r)
        return radii

    def delaunay_refine(params, max_lp_iter=50):
        """
        Refine a configuration by iteratively:
        1. Fix centers, solve LP for optimal radii
        2. Use gradient to adjust centers to allow larger radii
        """
        current = params.copy()
        w = np.clip(current[0], 0.3, 1.7)
        h = 2.0 - w
        
        for iteration in range(max_lp_iter):
            if time.time() > deadline - 5:
                break
            
            circles = current[1:]
            xs = circles[0::3].copy()
            ys = circles[1::3].copy()
            
            # Solve LP for radii given current centers
            radii = delaunay_radius_optimization(xs, ys, w, h)
            if radii is None:
                break
            
            # Check if radii are valid
            valid = True
            for i in range(N):
                if radii[i] < 1e-6:
                    valid = False
                    break
                if xs[i] - radii[i] < -1e-5 or xs[i] + radii[i] > w + 1e-5:
                    valid = False
                    break
                if ys[i] - radii[i] < -1e-5 or ys[i] + radii[i] > h + 1e-5:
                    valid = False
                    break
            
            if not valid:
                break
            
            # Build new params with optimized radii
            new_params = current.copy()
            for i in range(N):
                new_params[3 + 3*i] = radii[i]
            
            # Get value
            new_val = get_value(new_params)
            cur_val = get_value(current)
            
            if new_val > cur_val + 1e-8:
                current = new_params
            else:
                # Try small center perturbations to enable larger radii
                # Use gradient of sum of radii w.r.t. centers
                # For each circle, push away from neighbors to increase gap
                improved = False
                for _ in range(20):
                    if time.time() > deadline - 5:
                        break
                    # Small random perturbation of centers
                    perturbed = current.copy()
                    for i in range(N):
                        perturbed[1 + 3*i] += np.random.randn() * 0.005
                        perturbed[2 + 3*i] += np.random.randn() * 0.005
                    perturbed[1 + 3*i] = np.clip(perturbed[1 + 3*i], 0, w)
                    perturbed[2 + 3*i] = np.clip(perturbed[2 + 3*i], 0, h)
                    
                    # Solve LP again
                    p_xs = perturbed[1::3][0::3]
                    p_ys = perturbed[1::3][1::3]
                    p_radii = delaunay_radius_optimization(p_xs, p_ys, w, h)
                    if p_radii is None:
                        continue
                    
                    p_valid = True
                    for i in range(N):
                        if p_radii[i] < 1e-6:
                            p_valid = False
                            break
                        if p_xs[i] - p_radii[i] < -1e-5 or p_xs[i] + p_radii[i] > w + 1e-5:
                            p_valid = False
                            break
                        if p_ys[i] - p_radii[i] < -1e-5 or p_ys[i] + p_radii[i] > h + 1e-5:
                            p_valid = False
                            break
                    
                    if p_valid:
                        p_params = perturbed.copy()
                        for i in range(N):
                            p_params[3 + 3*i] = p_radii[i]
                        p_val = get_value(p_params)
                        if p_val > get_value(current) + 1e-8:
                            current = p_params
                            improved = True
                            break
                
                if not improved:
                    break
        
        return current

    best_value = -np.inf
    best_params = None

    random.seed(42)
    np.random.seed(42)
    
    candidates = []
    
    for w_try in [0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15]:
        h_try = 2.0 - w_try
        arr = make_initial(w_try, h_try, N)
        params = np.zeros(1 + 3*N)
        params[0] = w_try
        params[1:] = arr
        candidates.append(params.copy())
    
    for _ in range(30):
        w_try = random.uniform(0.8, 1.2)
        h_try = 2.0 - w_try
        arr = make_initial(w_try, h_try, N)
        arr += np.random.randn(len(arr)) * 0.03
        params = np.zeros(1 + 3*N)
        params[0] = w_try
        params[1:] = arr
        candidates.append(params.copy())

    for w_try in [0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2]:
        h_try = 2.0 - w_try
        side = 5
        xs = np.zeros(N)
        ys = np.zeros(N)
        rs = np.zeros(N)
        for i in range(N):
            row = i // side
            col = i % side
            xs[i] = (col + 0.5) / side * w_try
            ys[i] = (row + 0.5) / side * h_try
            rs[i] = 0.48 / side * min(w_try, h_try)
        arr = np.concatenate([xs, ys, rs])
        params = np.zeros(1 + 3*N)
        params[0] = w_try
        params[1:] = arr
        candidates.append(params.copy())

    for w_try in [0.9, 1.0, 1.1]:
        h_try = 2.0 - w_try
        for rows, cols in [(3, 7), (4, 6), (5, 5)]:
            if rows * cols < N:
                continue
            xs = np.zeros(N)
            ys = np.zeros(N)
            rs = np.zeros(N)
            for i in range(N):
                row = i // cols
                col = i % cols
                xs[i] = (col + 0.5) / cols * w_try
                ys[i] = (row + 0.5) / rows * h_try
                rs[i] = 0.48 * min(w_try / cols, h_try / rows)
            arr = np.concatenate([xs, ys, rs])
            params = np.zeros(1 + 3*N)
            params[0] = w_try
            params[1:] = arr
            candidates.append(params.copy())

    bounds = [(0.3, 1.7)] + [(-1.0, 3.0)] * (2*N) + [(1e-5, 1.0)] * N
    
    for idx, init_params in enumerate(candidates):
        if time.time() > deadline - 5:
            break
        
        val = get_value(init_params)
        if val > best_value:
            best_value = val
            best_params = init_params.copy()
        
        # Try Delaunay-based radius optimization
        try:
            del_params = delaunay_refine(init_params, max_lp_iter=30)
            del_val = get_value(del_params)
            if del_val > best_value:
                best_value = del_val
                best_params = del_params.copy()
        except Exception:
            pass
        
        # Try gradient descent
        try:
            gd_params = gradient_descent(init_params, lr=0.02, momentum=0.85, max_iter=2000)
            gd_val = get_value(gd_params)
            if gd_val > best_value:
                best_value = gd_val
                best_params = gd_params.copy()
        except Exception:
            pass
        
        # Try L-BFGS-B
        if time.time() < deadline - 10:
            try:
                result = minimize(
                    objective_and_grad,
                    init_params,
                    method='L-BFGS-B',
                    jac=True,
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-8}
                )
                opt_params = result.x
                opt_val = get_value(opt_params)
                if opt_val > best_value:
                    best_value = opt_val
                    best_params = opt_params.copy()
            except Exception:
                pass
    
    # Final refinement passes
    if best_params is not None and time.time() < deadline - 10:
        for pass_num in range(6):
            if time.time() > deadline - 3:
                break
            
            # Delaunay refinement
            try:
                del_params = delaunay_refine(best_params, max_lp_iter=40)
                del_val = get_value(del_params)
                if del_val > best_value:
                    best_value = del_val
                    best_params = del_params.copy()
            except Exception:
                pass
            
            # Perturb and re-optimize
            perturbed = best_params + np.random.randn(len(best_params)) * 0.008
            perturbed[0] = np.clip(perturbed[0], 0.3, 1.7)
            
            try:
                gd_params = gradient_descent(perturbed, lr=0.015, momentum=0.9, max_iter=1500)
                gd_val = get_value(gd_params)
                if gd_val > best_value:
                    best_value = gd_val
                    best_params = gd_params.copy()
            except Exception:
                pass
            
            # SLSQP polish
            if time.time() < deadline - 5:
                try:
                    result = minimize(
                        objective_and_grad,
                        best_params,
                        method='SLSQP',
                        jac=True,
                        bounds=bounds,
                        options={'maxiter': 300, 'ftol': 1e-10}
                    )
                    opt_params = result.x
                    opt_val = get_value(opt_params)
                    if opt_val > best_value:
                        best_value = opt_val
                        best_params = opt_params.copy()
                except Exception:
                    pass

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

    w_final = np.clip(best_params[0], 0.1, 1.9)
    h_final = 2.0 - w_final
    circles = best_params[1:]
    
    result = np.zeros((N, 3))
    for i in range(N):
        result[i, 0] = circles[3*i]
        result[i, 1] = circles[3*i+1]
        result[i, 2] = circles[3*i+2]
    
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
