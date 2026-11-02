"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii."""
import numpy as np
import time
import random
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 21

def _softplus(x):
    return np.where(x > 20, x, np.log1p(np.exp(x)))

def _sigmoid(x):
    return np.where(x > 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

def _delaunay_edges(points):
    """Get unique edges from Delaunay triangulation."""
    try:
        tri = Delaunay(points)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                a, b = simplex[i], simplex[(i+1) % 3]
                edges.add((min(a,b), max(a,b)))
        return list(edges)
    except Exception:
        # Fallback: all pairs
        n = len(points)
        return [(i, j) for i in range(n) for j in range(i+1, n)]

def _solve_lp_radii(xs, ys, w, h):
    """
    Solve for maximum sum of radii given fixed centers using LP.
    Constraints:
      r_i + r_j <= dist(i,j)  for all pairs
      r_i <= x_i, r_i <= w - x_i, r_i <= y_i, r_i <= h - y_i
      r_i >= 0
    Maximize sum(r_i)
    """
    from scipy.optimize import linprog
    n = len(xs)
    # Objective: maximize sum(r) => minimize -sum(r)
    c = -np.ones(n)
    
    A_ub = []
    b_ub = []
    
    # Pairwise distance constraints: r_i + r_j <= d_ij
    for i in range(n):
        for j in range(i+1, n):
            d = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(d)
    
    # Boundary constraints: r_i <= min(x_i, w-x_i, y_i, h-y_i)
    for i in range(n):
        max_r = min(xs[i], w - xs[i], ys[i], h - ys[i])
        max_r = max(max_r, 0.0)
        row = np.zeros(n)
        row[i] = 1.0
        A_ub.append(row)
        b_ub.append(max_r)
    
    bounds = [(0, None)] * n
    
    res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), bounds=bounds, method='highs')
    if res.success:
        return res.x
    return np.zeros(n)

def _solve_lp_equal_radius(xs, ys, w, h):
    """
    Solve for maximum equal radius t such that all circles of radius t
    fit in the rectangle with centers at (xs, ys).
    Constraints:
      2t <= dist(i,j)  for all pairs
      t <= x_i, t <= w - x_i, t <= y_i, t <= h - y_i
      t >= 0
    Maximize t
    Returns (t, radii_array) or (None, None) if infeasible.
    """
    from scipy.optimize import linprog
    n = len(xs)
    # Objective: maximize t => minimize -t
    c = np.zeros(n + 1)
    c[-1] = -1.0  # t is the last variable
    
    A_ub = []
    b_ub = []
    
    # Pairwise: 2t <= d_ij  =>  -2t >= -d_ij  =>  0*r + (-2)*t >= -d_ij
    # In linprog form (A_ub @ x <= b_ub):  0*r + 2*t <= d_ij
    for i in range(n):
        for j in range(i+1, n):
            d = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
            row = np.zeros(n + 1)
            row[-1] = 2.0
            A_ub.append(row)
            b_ub.append(d)
    
    # Boundary: t <= min(x_i, w-x_i, y_i, h-y_i) for each i
    # This means t <= x_i AND t <= w-x_i AND t <= y_i AND t <= h-y_i
    # So: -t <= -x_i, t <= w-x_i, -t <= -y_i, t <= h-y_i
    for i in range(n):
        # t <= x_i  =>  -t >= -x_i  =>  in linprog: 0*r + (-1)*t <= -x_i ... no
        # Actually: t <= x_i means t - x_i <= 0, i.e., 0*r + 1*t <= x_i
        row = np.zeros(n + 1)
        row[-1] = 1.0
        A_ub.append(row)
        b_ub.append(xs[i])
        
        row = np.zeros(n + 1)
        row[-1] = 1.0
        A_ub.append(row)
        b_ub.append(w - xs[i])
        
        row = np.zeros(n + 1)
        row[-1] = 1.0
        A_ub.append(row)
        b_ub.append(ys[i])
        
        row = np.zeros(n + 1)
        row[-1] = 1.0
        A_ub.append(row)
        b_ub.append(h - ys[i])
    
    bounds = [(0, None)] * n + [(0, None)]
    
    res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), bounds=bounds, method='highs')
    if res.success and res.x[-1] > 1e-8:
        t = res.x[-1]
        radii = np.full(n, t)
        return t, radii
    return None, None

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill
    
    def _lp_radii_for_centers(xs, ys, w, h):
        """Solve LP for optimal radii given fixed centers using Delaunay edges."""
        from scipy.optimize import linprog
        n = len(xs)
        c = -np.ones(n)
        A_ub = []
        b_ub = []
        
        # Get Delaunay edges for pairwise constraints
        points = np.column_stack([xs, ys])
        edges = _delaunay_edges(points)
        
        for (i, j) in edges:
            d = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(d)
        
        # Add non-Delaunay pairs that are close (within 3x min edge length)
        if edges:
            min_edge = min(np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2) for (i,j) in edges)
            threshold = 3.0 * min_edge
            edge_set = set(edges)
            for i in range(n):
                for j in range(i+1, n):
                    if (i,j) not in edge_set:
                        d = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                        if d < threshold:
                            row = np.zeros(n)
                            row[i] = 1.0
                            row[j] = 1.0
                            A_ub.append(row)
                            b_ub.append(d)
        
        # Boundary constraints
        for i in range(n):
            max_r = min(xs[i], w - xs[i], ys[i], h - ys[i])
            max_r = max(max_r, 0.0)
            row = np.zeros(n)
            row[i] = 1.0
            A_ub.append(row)
            b_ub.append(max_r)
        
        bounds_lp = [(0, None)] * n
        try:
            res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), bounds=bounds_lp, method='highs')
            if res.success:
                return res.x
        except Exception:
            pass
        return None

    def _lp_equal_radius_for_centers(xs, ys, w, h):
        """Solve LP for max equal radius using Delaunay edges."""
        from scipy.optimize import linprog
        n = len(xs)
        c = np.zeros(n + 1)
        c[-1] = -1.0
        
        A_ub = []
        b_ub = []
        
        points = np.column_stack([xs, ys])
        edges = _delaunay_edges(points)
        
        for (i, j) in edges:
            d = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
            row = np.zeros(n + 1)
            row[-1] = 2.0
            A_ub.append(row)
            b_ub.append(d)
        
        if edges:
            min_edge = min(np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2) for (i,j) in edges)
            threshold = 3.0 * min_edge
            edge_set = set(edges)
            for i in range(n):
                for j in range(i+1, n):
                    if (i,j) not in edge_set:
                        d = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                        if d < threshold:
                            row = np.zeros(n + 1)
                            row[-1] = 2.0
                            A_ub.append(row)
                            b_ub.append(d)
        
        for i in range(n):
            for val in [xs[i], w - xs[i], ys[i], h - ys[i]]:
                row = np.zeros(n + 1)
                row[-1] = 1.0
                A_ub.append(row)
                b_ub.append(max(val, 0.0))
        
        bounds_lp = [(0, None)] * n + [(0, None)]
        try:
            res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), bounds=bounds_lp, method='highs')
            if res.success and res.x[-1] > 1e-8:
                t = res.x[-1]
                radii = np.full(n, t)
                return t, radii
        except Exception:
            pass
        return None, None

    def _lp_shadow_price_gradient(xs, ys, w, h):
        """
        Compute gradient of optimal sum-of-radii w.r.t. centers using LP dual values.
        Returns (grad_x, grad_y, sum_r, radii) where grad is d(sum_r)/d(x_i), d(sum_r)/d(y_i).
        """
        from scipy.optimize import linprog
        n = len(xs)
        c = -np.ones(n)
        A_ub = []
        b_ub = []
        constraint_map = []  # (type, i, j) or (type, i)
        
        points = np.column_stack([xs, ys])
        edges = _delaunay_edges(points)
        
        for (i, j) in edges:
            d = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(d)
            constraint_map.append(('pair', i, j))
        
        if edges:
            min_edge = min(np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2) for (i,j) in edges)
            threshold = 3.0 * min_edge
            edge_set = set(edges)
            for i in range(n):
                for j in range(i+1, n):
                    if (i,j) not in edge_set:
                        d = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                        if d < threshold:
                            row = np.zeros(n)
                            row[i] = 1.0
                            row[j] = 1.0
                            A_ub.append(row)
                            b_ub.append(d)
                            constraint_map.append(('pair', i, j))
        
        for i in range(n):
            max_r = min(xs[i], w - xs[i], ys[i], h - ys[i])
            max_r = max(max_r, 0.0)
            row = np.zeros(n)
            row[i] = 1.0
            A_ub.append(row)
            b_ub.append(max_r)
            constraint_map.append(('bound', i))
        
        bounds_lp = [(0, None)] * n
        try:
            res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), bounds=bounds_lp, method='highs')
            if not res.success:
                return None
            radii = res.x
            sum_r = np.sum(radii)
            
            # Get dual values (shadow prices) for A_ub constraints
            # In HiGHS, dual values are in res.ineqlin.marginals (for A_ub x <= b_ub)
            duals = None
            if hasattr(res, 'ineqlin') and res.ineqlin is not None:
                duals = res.ineqlin.marginals
            elif hasattr(res, 'slacks'):
                # Fallback: use slacks to identify active constraints
                duals = None
            
            grad_x = np.zeros(n)
            grad_y = np.zeros(n)
            
            if duals is not None and len(duals) == len(constraint_map):
                for k, cm in enumerate(constraint_map):
                    dual_val = abs(duals[k]) if duals[k] is not None else 0.0
                    if dual_val < 1e-10:
                        continue
                    if cm[0] == 'pair':
                        _, i, j = cm
                        dx = xs[i] - xs[j]
                        dy = ys[i] - ys[j]
                        dist = np.sqrt(dx*dx + dy*dy)
                        if dist < 1e-12:
                            continue
                        # d(r_i + r_j)/d(x_i) = -d_ij/dx_i = -(x_i - x_j)/d_ij = -dx/d
                        # Shadow price tells us how much objective improves per unit relaxation
                        # For constraint r_i + r_j <= d_ij, if active, increasing d_ij by eps
                        # increases sum_r by dual_val * eps
                        # d(d_ij)/d(x_i) = (x_i - x_j)/d_ij = dx/d
                        # So d(sum_r)/d(x_i) += dual_val * dx/d
                        coeff = dual_val / dist
                        grad_x[i] += coeff * dx
                        grad_x[j] -= coeff * dx
                        grad_y[i] += coeff * dy
                        grad_y[j] -= coeff * dy
                    elif cm[0] == 'bound':
                        _, i = cm
                        # For boundary constraint r_i <= min(x_i, w-x_i, y_i, h-y_i)
                        # The tightest boundary determines the constraint
                        # d(sum_r)/d(x_i) depends on which boundary is tight
                        # If x_i is tight: d(sum_r)/d(x_i) += dual_val
                        # If w-x_i is tight: d(sum_r)/d(x_i) -= dual_val
                        # etc.
                        # We need to figure out which boundary is the binding one
                        vals = [xs[i], w - xs[i], ys[i], h - ys[i]]
                        min_val = min(vals)
                        # Check which are close to min
                        for k2, val in enumerate(vals):
                            if abs(val - min_val) < 1e-8:
                                if k2 == 0:  # x_i
                                    grad_x[i] += dual_val
                                elif k2 == 1:  # w - x_i
                                    grad_x[i] -= dual_val
                                elif k2 == 2:  # y_i
                                    grad_y[i] += dual_val
                                elif k2 == 3:  # h - y_i
                                    grad_y[i] -= dual_val
            
            return (grad_x, grad_y, sum_r, radii)
        except Exception:
            pass
        return None
    
    def softplus(x):
        return _softplus(x)

    def sigmoid(x):
        return _sigmoid(x)

    W_BOUNDARY = 100.0
    W_OVERLAP = 50.0
    
    # We optimize over: w (width), and for each circle (x_i, y_i, r_i)
    # h = 2 - w, perimeter = 2*(w+h) = 4
    # Total variables: 1 + 3*N = 64
    
    def make_hex(w, h, n_circles):
        """Create hexagonal packing initialization."""
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

    def make_grid(w, h, n_circles, side=5):
        """Create grid packing initialization."""
        xs = np.zeros(n_circles)
        ys = np.zeros(n_circles)
        rs = np.zeros(n_circles)
        for i in range(n_circles):
            row = i // side
            col = i % side
            xs[i] = (col + 0.5) / side * w
            ys[i] = (row + 0.5) / side * h
            rs[i] = 0.48 / side * min(w, h)
        return np.concatenate([xs, ys, rs])

    def make_random(w, h, n_circles):
        """Create random initialization."""
        xs = np.random.uniform(0.1*w, 0.9*w, n_circles)
        ys = np.random.uniform(0.1*h, 0.9*h, n_circles)
        rs = np.full(n_circles, 0.05 * min(w, h))
        return np.concatenate([xs, ys, rs])

    def refine_slsqp(params, maxiter=1000, ftol=1e-10):
        """Final SLSQP polish pass."""
        bounds = [(0.3, 1.7)] + [(-1.0, 3.0)] * (2*N) + [(1e-5, 1.0)] * N
        try:
            result = minimize(
                objective_and_grad,
                params,
                method='SLSQP',
                jac=True,
                bounds=bounds,
                options={'maxiter': maxiter, 'ftol': ftol}
            )
            return result.x
        except Exception:
            return params

    def make_initial(w, h, n_circles):
        """Create a hexagonal-like packing initialization."""
        if n_circles == 0:
            return np.zeros(0)
        
        # Hexagonal packing: rows offset
        # For n circles, find best grid
        best_score = -1
        best_arr = None
        
        for rows in range(1, n_circles + 1):
            cols = (n_circles + rows - 1) // rows  # ceil division
            # Try this grid
            # Hexagonal: each row offset by 0.5 * spacing_x
            # spacing_x = w / cols (approx)
            # spacing_y = h / rows
            # For hexagonal, vertical spacing = 0.866 * horizontal spacing (for equilateral)
            # But we're in a rectangle, so we just use grid with hex offset
            
            xs = np.zeros(n_circles)
            ys = np.zeros(n_circles)
            rs = np.zeros(n_circles)
            
            for i in range(n_circles):
                row = i // cols
                col = i % cols
                # Center of cell
                x = (col + 0.5) * (w / cols)
                y = (row + 0.5) * (h / rows)
                # Hex offset: shift alternate rows
                if row % 2 == 1:
                    x += 0.5 * (w / cols)
                # Clamp
                x = np.clip(x, 0.1 * w, 0.9 * w)
                y = np.clip(y, 0.1 * h, 0.9 * h)
                
                # Radius: half the min of cell width and height, times safety factor
                cell_w = w / cols
                cell_h = h / rows
                r = 0.45 * min(cell_w, cell_h)
                
                xs[i] = x
                ys[i] = y
                rs[i] = r
            
            # Quick validity check
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
            # Fallback: simple grid
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
        """Penalty-based objective with analytic gradient. Uses softplus for smoothness."""
        w = params[0]
        w_clipped = np.clip(w, 0.3, 1.7)
        h = 2.0 - w_clipped
        
        circles = params[1:]
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
                        obj_val += W_OVERLAP * softplus(v)
                        
                        coeff = W_OVERLAP * sp_grad / dist
                        grad[3*i] -= coeff * dx
                        grad[3*i+1] -= coeff * dy
                        grad[3*i+2] += W_OVERLAP * sp_grad
                        
                        grad[3*j] += coeff * dx
                        grad[3*j+1] += coeff * dy
                        grad[3*j+2] += W_OVERLAP * sp_grad

        # Gradient for w
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
        # Verify bounding box perimeter
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

    best_value = -np.inf
    best_params = None

    # Try multiple aspect ratios and initializations
    random.seed(42)
    np.random.seed(42)
    
    # Generate a set of candidate initial configurations
    candidates = []
    
    # 1. Square-ish with hexagonal pattern
    for w_try in [0.9, 1.0, 1.1, 0.95, 1.05]:
        h_try = 2.0 - w_try
        arr = make_hex(w_try, h_try, N)
        params = np.zeros(1 + 3*N)
        params[0] = w_try
        params[1:] = arr
        candidates.append(params.copy())
    
    # 2. Random perturbations of hexagonal
    for _ in range(30):
        w_try = random.uniform(0.85, 1.15)
        h_try = 2.0 - w_try
        arr = make_hex(w_try, h_try, N)
        # Perturb
        arr += np.random.randn(len(arr)) * 0.02
        params = np.zeros(1 + 3*N)
        params[0] = w_try
        params[1:] = arr
        candidates.append(params.copy())
    
    # 5. More diverse aspect ratios with hexagonal
    for w_try in [0.85, 0.92, 1.08, 1.15, 0.88, 1.12]:
        h_try = 2.0 - w_try
        arr = make_hex(w_try, h_try, N)
        params = np.zeros(1 + 3*N)
        params[0] = w_try
        params[1:] = arr
        candidates.append(params.copy())
        # LP-enhanced version
        xs = arr[0:N]
        ys = arr[N:2*N]
        rs_lp = _lp_radii_for_centers(xs, ys, w_try, h_try)
        if rs_lp is not None:
            arr_lp = np.concatenate([xs, ys, rs_lp])
            params_lp = np.zeros(1 + 3*N)
            params_lp[0] = w_try
            params_lp[1:] = arr_lp
            candidates.append(params_lp.copy())

    # 3. Simple grid initialization
    for w_try in [0.8, 0.9, 1.0, 1.1, 1.2]:
        h_try = 2.0 - w_try
        arr = make_grid(w_try, h_try, N)
        params = np.zeros(1 + 3*N)
        params[0] = w_try
        params[1:] = arr
        candidates.append(params.copy())

    # 4. LP-enhanced: take hex centers, solve LP for optimal radii
    for w_try in [0.9, 1.0, 1.1, 0.95, 1.05]:
        h_try = 2.0 - w_try
        arr = make_hex(w_try, h_try, N)
        xs = arr[0:N]
        ys = arr[N:2*N]
        rs_lp = _lp_radii_for_centers(xs, ys, w_try, h_try)
        if rs_lp is not None:
            arr_lp = np.concatenate([xs, ys, rs_lp])
            params = np.zeros(1 + 3*N)
            params[0] = w_try
            params[1:] = arr_lp
            candidates.append(params.copy())

    # Evaluate and improve each candidate
    # We will use SLSQP with analytic gradients for faster and more reliable convergence
    bounds = [(0.3, 1.7)] + [(-1.0, 3.0)] * (2*N) + [(1e-5, 1.0)] * N
    
    def _polish_with_lp_gradient(params, max_steps=50, step_size=0.01):
        """
        Polish centers using LP shadow prices as gradient signal.
        Moves centers in direction that increases optimal sum-of-radii.
        """
        current = params.copy()
        w = np.clip(current[0], 0.3, 1.7)
        h = 2.0 - w
        
        for step in range(max_steps):
            if time.time() > deadline - 5:
                break
            
            xs = current[1:1+3*N:3]
            ys = current[2:2+3*N:3]
            
            result = _lp_shadow_price_gradient(xs, ys, w, h)
            if result is None:
                break
            grad_x, grad_y, sum_r, radii = result
            
            # Check gradient magnitude
            grad_norm = np.sqrt(np.sum(grad_x**2) + np.sum(grad_y**2))
            if grad_norm < 1e-8:
                break
            
            # Adaptive step size
            ss = step_size * min(1.0, 0.1 / (grad_norm + 1e-8))
            
            # Proposed new centers
            new_xs = xs + ss * grad_x
            new_ys = ys + ss * grad_y
            
            # Clamp to valid range
            new_xs = np.clip(new_xs, 0.01, w - 0.01)
            new_ys = np.clip(new_ys, 0.01, h - 0.01)
            
            # Solve LP with new centers
            new_radii = _lp_radii_for_centers(new_xs, new_ys, w, h)
            if new_radii is None:
                break
            
            new_sum = np.sum(new_radii)
            
            if new_sum > sum_r + 1e-10:
                current[1:1+3*N:3] = new_xs
                current[2:2+3*N:3] = new_ys
                current[3::3] = new_radii
            else:
                # Reduce step size
                step_size *= 0.5
                if step_size < 1e-6:
                    break
        
        return current
    
    for idx, init_params in enumerate(candidates):
        if time.time() > deadline - 10:
            break
        
        # First, check if valid as-is
        val = get_value(init_params)
        if val > best_value:
            best_value = val
            best_params = init_params.copy()
        
        # Try local optimization using L-BFGS-B (faster)
        try:
            result = minimize(
                objective_and_grad,
                init_params,
                method='L-BFGS-B',
                jac=True, # Indicates that objective returns (f, g)
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-8}
            )
            
            opt_params = result.x
            opt_val = get_value(opt_params)
            if opt_val > best_value:
                best_value = opt_val
                best_params = opt_params.copy()
        except Exception:
            pass
        
        # After optimization, try LP-based radius refinement
        if best_params is not None and time.time() < deadline - 15:
            w_lp = np.clip(best_params[0], 0.3, 1.7)
            h_lp = 2.0 - w_lp
            xs_lp = best_params[1:1+3*N:3]
            ys_lp = best_params[2:2+3*N:3]
            rs_lp = _lp_radii_for_centers(xs_lp, ys_lp, w_lp, h_lp)
            if rs_lp is not None:
                new_params = best_params.copy()
                new_params[3::3] = rs_lp
                new_val = get_value(new_params)
                if new_val > best_value:
                    best_value = new_val
                    best_params = new_params.copy()
    
    # If we have a valid best, try to refine with more iterations
    if best_params is not None and time.time() < deadline - 10:
        # First polish with SLSQP
        if time.time() < deadline - 20:
            best_params = refine_slsqp(best_params, maxiter=1000, ftol=1e-10)
            opt_val = get_value(best_params)
            if opt_val > best_value:
                best_value = opt_val
        
        # LP-based radius refinement after SLSQP
        if time.time() < deadline - 15:
            w_lp = np.clip(best_params[0], 0.3, 1.7)
            h_lp = 2.0 - w_lp
            xs_lp = best_params[1:1+3*N:3]
            ys_lp = best_params[2:2+3*N:3]
            rs_lp = _lp_radii_for_centers(xs_lp, ys_lp, w_lp, h_lp)
            if rs_lp is not None:
                new_params = best_params.copy()
                new_params[3::3] = rs_lp
                new_val = get_value(new_params)
                if new_val > best_value:
                    best_value = new_val
                    best_params = new_params.copy()
        
        # Use LP shadow price gradient to polish centers
        if time.time() < deadline - 12:
            best_params = _polish_with_lp_gradient(best_params, max_steps=80, step_size=0.02)
            opt_val = get_value(best_params)
            if opt_val > best_value:
                best_value = opt_val
        
        # Second polish: tighter tolerance
        if time.time() < deadline - 10:
            best_params = refine_slsqp(best_params, maxiter=2000, ftol=1e-12)
            opt_val = get_value(best_params)
            if opt_val > best_value:
                best_value = opt_val
        
        # Final LP-based radius refinement
        if time.time() < deadline - 5:
            w_lp = np.clip(best_params[0], 0.3, 1.7)
            h_lp = 2.0 - w_lp
            xs_lp = best_params[1:1+3*N:3]
            ys_lp = best_params[2:2+3*N:3]
            rs_lp = _lp_radii_for_centers(xs_lp, ys_lp, w_lp, h_lp)
            if rs_lp is not None:
                new_params = best_params.copy()
                new_params[3::3] = rs_lp
                new_val = get_value(new_params)
                if new_val > best_value:
                    best_value = new_val
                    best_params = new_params.copy()
        
        # Third polish: very tight
        if time.time() < deadline - 3:
            best_params = refine_slsqp(best_params, maxiter=3000, ftol=1e-14)
            opt_val = get_value(best_params)
            if opt_val > best_value:
                best_value = opt_val
        
        # Final LP gradient polish
        if time.time() < deadline - 2:
            best_params = _polish_with_lp_gradient(best_params, max_steps=50, step_size=0.005)
            opt_val = get_value(best_params)
            if opt_val > best_value:
                best_value = opt_val

    # If still no valid solution, create a guaranteed valid one
    if best_params is None or best_value < 0:
        # Fallback: 5x5 grid in a 1x1 square (perimeter 4)
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
    w_final = np.clip(best_params[0], 0.1, 1.9)
    h_final = 2.0 - w_final
    circles = best_params[1:]
    
    result = np.zeros((N, 3))
    for i in range(N):
        result[i, 0] = circles[3*i]
        result[i, 1] = circles[3*i+1]
        result[i, 2] = circles[3*i+2]
    
    # Final verification: ensure bounding box perimeter <= 4
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
        # Scale down if needed
        scale = 4.0 / bbox_perim
        # Scale positions and radii relative to center of bbox
        cx = (min_x + max_x) / 2
        cy = (min_y + max_y) / 2
        for i in range(N):
            result[i, 0] = cx + (result[i, 0] - cx) * scale
            result[i, 1] = cy + (result[i, 1] - cy) * scale
            result[i, 2] = result[i, 2] * scale
    
    return result
