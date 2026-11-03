import numpy as np
from scipy.optimize import minimize, linprog
from scipy.spatial import Delaunay
import time

N = 21

def _softplus(x, beta=50.0):
    """Smooth approximation of max(0, x) using softplus."""
    out = np.zeros_like(x)
    large = beta * x > 50
    small = beta * x < -50
    mid = ~(large | small)
    out[large] = x[large]
    out[mid] = (1.0 / beta) * np.log1p(np.exp(beta * x[mid]))
    return out

def _softplus_grad(x, beta=50.0):
    """Gradient of the stable softplus above."""
    out = np.zeros_like(x)
    large = beta * x > 50
    small = beta * x < -50
    mid = ~(large | small)
    out[large] = 1.0
    out[mid] = 1.0 / (1.0 + np.exp(-beta * x[mid]))
    return out

def _delaunay_edges(xs, ys):
    """
    Compute the Delaunay triangulation edges from points.
    Returns a list of (i, j) pairs representing edges.
    """
    pts = np.column_stack([xs, ys])
    try:
        tri = Delaunay(pts)
        edges = set()
        for simplex in tri.simplices:
            for i in range(len(simplex)):
                for j in range(i + 1, len(simplex)):
                    a, b = simplex[i], simplex[j]
                    if a > b:
                        a, b = b, a
                    edges.add((a, b))
        return list(edges)
    except Exception:
        # Fallback: all pairs if Delaunay fails (e.g., < 3 points or degenerate)
        return [(i, j) for i in range(N) for j in range(i + 1, N)]

def _solve_lp_radii(xs, ys, w, h, edges):
    """
    Given fixed centers (xs, ys) and rectangle (w, h), solve for the maximum
    sum of radii such that circles do not overlap and stay within bounds.
    
    This is a Linear Program:
    Maximize: sum(r_i)
    Subject to:
    r_i + r_j <= dist_ij  for all (i, j) in edges (and potentially more if we want global)
    r_i <= x_i
    r_i <= w - x_i
    r_i <= y_i
    r_i <= h - y_i
    r_i >= 0
    
    Note: Using only Delaunay edges is a relaxation. It might allow overlaps
    for non-adjacent circles if the geometry is tricky, but for "packing"
    where we start from a valid config or near-valid, it's a good local approximation.
    However, to be safe and ensure validity, we should probably include all pairs
    or at least check validity afterwards. For N=21, all pairs is 210 constraints,
    which is tiny for an LP. So let's just use ALL pairs for correctness in the LP step.
    """
    n = N
    # Variables: r_0, ..., r_{n-1}
    c = -np.ones(n) # Minimize -sum(r)
    
    # Inequality constraints: A_ub @ r <= b_ub
    # 1. Overlap constraints: r_i + r_j <= dist_ij
    # We will use ALL pairs to ensure global non-overlap in the LP solution.
    # This is O(N^2) constraints, which is fine for N=21.
    
    A_ub_list = []
    b_ub_list = []
    
    # Pairwise constraints
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx*dx + dy*dy)
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub_list.append(row)
            b_ub_list.append(dist)
            
    # Boundary constraints
    # r_i <= x_i  =>  r_i <= x_i
    # r_i <= w - x_i => r_i <= w - x_i
    # r_i <= y_i => r_i <= y_i
    # r_i <= h - y_i => r_i <= h - y_i
    
    for i in range(n):
        # r_i <= x_i
        row = np.zeros(n)
        row[i] = 1.0
        A_ub_list.append(row)
        b_ub_list.append(xs[i])
        
        # r_i <= w - x_i
        row = np.zeros(n)
        row[i] = 1.0
        A_ub_list.append(row)
        b_ub_list.append(w - xs[i])
        
        # r_i <= y_i
        row = np.zeros(n)
        row[i] = 1.0
        A_ub_list.append(row)
        b_ub_list.append(ys[i])
        
        # r_i <= h - y_i
        row = np.zeros(n)
        row[i] = 1.0
        A_ub_list.append(row)
        b_ub_list.append(h - ys[i])
        
    A_ub = np.array(A_ub_list)
    b_ub = np.array(b_ub_list)
    
    bounds = [(0, None)] * n
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.x
        else:
            return None
    except Exception:
        return None

def _is_valid(params, w, h, tol=1e-7):
    xs = params[0:N]
    ys = params[N:2*N]
    rs = params[2*N:3*N]
    
    if np.any(rs < -tol): return False
    if np.any(xs < -tol) or np.any(w - xs < -tol): return False
    if np.any(ys < -tol) or np.any(h - ys < -tol): return False
    if np.any(rs - xs < -tol) or np.any(rs - (w - xs) < -tol): return False
    if np.any(rs - ys < -tol) or np.any(rs - (h - ys) < -tol): return False
    
    for i in range(N):
        for j in range(i+1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < rs[i] + rs[j] - tol:
                return False
    return True

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill
    
    def get_wh(w):
        h = 2.0 - w
        return w, h

    def objective_and_grad(params, w, h, beta=50.0, lambda_pen=100.0):
        """
        params: [x_0, ..., x_{N-1}, y_0, ..., y_{N-1}, r_0, ..., r_{N-1}]
        Objective: -sum(r) + penalties for overlaps and boundary violations.
        """
        n = N
        xs = params[0:n]
        ys = params[n:2*n]
        rs = params[2*n:3*n]
        
        f = -np.sum(rs)
        g = np.zeros(3*n)
        g[2*n:3*n] = -1.0  # Gradient of -sum(r) w.r.t r is -1
        
        # 1. Boundary Penalties
        v_x0 = rs - xs
        v_x1 = rs - (w - xs)
        v_y0 = rs - ys
        v_y1 = rs - (h - ys)
        
        pen_x0 = _softplus(v_x0, beta)
        pen_x1 = _softplus(v_x1, beta)
        pen_y0 = _softplus(v_y0, beta)
        pen_y1 = _softplus(v_y1, beta)
        
        f += lambda_pen * (np.sum(pen_x0) + np.sum(pen_x1) + np.sum(pen_y0) + np.sum(pen_y1))
        
        g_x0 = _softplus_grad(v_x0, beta)
        g_x1 = _softplus_grad(v_x1, beta)
        g_y0 = _softplus_grad(v_y0, beta)
        g_y1 = _softplus_grad(v_y1, beta)
        
        g[0:n] -= lambda_pen * g_x0
        g[0:n] += lambda_pen * g_x1
        g[n:2*n] -= lambda_pen * g_y0
        g[n:2*n] += lambda_pen * g_y1
        g[2*n:3*n] += lambda_pen * (g_x0 + g_x1 + g_y0 + g_y1)
        
        # 2. Overlap Penalties
        dx = xs[:, None] - xs[None, :]
        dy = ys[:, None] - ys[None, :]
        dist_sq = dx*dx + dy*dy
        np.fill_diagonal(dist_sq, 1e-12)
        dist = np.sqrt(dist_sq)
        
        r_sum = rs[:, None] + rs[None, :]
        viol = r_sum - dist
        np.fill_diagonal(viol, 0.0)
        
        iu = np.triu_indices(n, 1)
        viol_pairs = viol[iu]
        
        pen_overlap = _softplus(viol_pairs, beta)
        f += lambda_pen * np.sum(pen_overlap)
        
        g_sp = _softplus_grad(viol_pairs, beta)
        
        i_idx, j_idx = iu
        dist_pairs = dist[iu]
        dx_pairs = dx[iu]
        dy_pairs = dy[iu]
        
        w_x = g_sp * (-dx_pairs / dist_pairs)
        w_y = g_sp * (-dy_pairs / dist_pairs)
        
        np.add.at(g[0:n], i_idx, lambda_pen * w_x)
        np.add.at(g[0:n], j_idx, -lambda_pen * w_x)
        
        np.add.at(g[n:2*n], i_idx, lambda_pen * w_y)
        np.add.at(g[n:2*n], j_idx, -lambda_pen * w_y)
        
        np.add.at(g[2*n:3*n], i_idx, lambda_pen * g_sp)
        np.add.at(g[2*n:3*n], j_idx, lambda_pen * g_sp)
        
        return f, g

    def make_grid_init(w, h, rng, noise_scale=0.01):
        best_init = None
        best_sum = -1
        for rows in range(3, 8):
            cols = int(np.ceil(N / rows))
            if cols < 2: continue
            
            xs = np.zeros(N)
            ys = np.zeros(N)
            rs = np.zeros(N)
            
            for i in range(N):
                row = i // cols
                col = i % cols
                xs[i] = (col + 0.5) * (w / cols)
                ys[i] = (row + 0.5) * (h / rows)
                r_est = 0.5 * min(xs[i], w - xs[i], ys[i], h - ys[i])
                rs[i] = r_est
            
            xs += rng.normal(0, noise_scale, N)
            ys += rng.normal(0, noise_scale, N)
            rs = np.clip(rs, 0.01, 0.5)
            
            params = np.concatenate([xs, ys, rs])
            s = np.sum(rs)
            if s > best_sum:
                best_sum = s
                best_init = params
        return best_init

    def make_hex_init(w, h, rng):
        patterns = [
            [5, 4, 5, 4, 3],
            [4, 5, 4, 5, 3],
            [5, 5, 5, 4, 2],
            [3, 4, 5, 4, 5],
            [4, 4, 5, 4, 4],
            [7, 7, 7],
            [4, 4, 4, 4, 5],
        ]
        best_init = None
        best_sum = -1
        
        for pattern in patterns:
            if sum(pattern) != N: continue
            
            xs = np.zeros(N)
            ys = np.zeros(N)
            rs = np.zeros(N)
            
            idx = 0
            n_rows = len(pattern)
            row_h = h / n_rows
            
            for row_idx, n_cols in enumerate(pattern):
                col_w = w / n_cols
                offset = 0.0 if row_idx % 2 == 0 else col_w / 2
                for col_idx in range(n_cols):
                    if idx >= N: break
                    x = (col_idx + 0.5) * col_w + offset
                    y = (row_idx + 0.5) * row_h
                    xs[idx] = x
                    ys[idx] = y
                    r_est = 0.5 * min(x, w - x, y, h - y)
                    rs[idx] = r_est
                    idx += 1
            
            if idx != N: continue
            
            xs += rng.normal(0, 0.01, N)
            ys += rng.normal(0, 0.01, N)
            rs = np.clip(rs, 0.01, 0.5)
            
            params = np.concatenate([xs, ys, rs])
            s = np.sum(rs)
            if s > best_sum:
                best_sum = s
                best_init = params
        return best_init

    def make_random_init(w, h, rng):
        xs = rng.uniform(0.1, w - 0.1, N)
        ys = rng.uniform(0.1, h - 0.1, N)
        rs = rng.uniform(0.05, 0.2, N)
        return np.concatenate([xs, ys, rs])

    best_val = 0.0
    best_params = None
    best_w = 1.0
    
    rng = np.random.default_rng(42)
    
    # Search over aspect ratio
    w_values = np.linspace(0.6, 1.4, 15)
    
    for w in w_values:
        if time.time() - start_time > time_limit - 20:
            break
            
        h = 2.0 - w
        
        n_restarts = 10
        for k in range(n_restarts):
            if time.time() - start_time > time_limit - 10:
                break
                
            if k % 3 == 0:
                params0 = make_grid_init(w, h, rng, noise_scale=0.02)
            elif k % 3 == 1:
                params0 = make_hex_init(w, h, rng)
                if params0 is None:
                    params0 = make_grid_init(w, h, rng, noise_scale=0.03)
            else:
                params0 = make_random_init(w, h, rng)
                
            if params0 is None:
                continue
                
            try:
                max_r = min(w, h) / 2.0
                bounds = []
                for i in range(N):
                    bounds.append((0.0, w))
                for i in range(N):
                    bounds.append((0.0, h))
                for i in range(N):
                    bounds.append((0.0, max_r))
                    
                # Phase 1: General optimization
                res = minimize(objective_and_grad, params0, args=(w, h), 
                               jac=True, method='L-BFGS-B',
                               bounds=bounds,
                               options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50})
                
                current_params = res.x if res.success else params0
                current_val = np.sum(current_params[2*N:3*N])
                
                # Phase 2: LP-based radius polishing
                # Fix centers, optimize radii via LP
                xs_cur = current_params[0:N].copy()
                ys_cur = current_params[N:2*N].copy()
                
                # Ensure centers are strictly inside to help LP
                xs_cur = np.clip(xs_cur, 1e-6, w - 1e-6)
                ys_cur = np.clip(ys_cur, 1e-6, h - 1e-6)
                
                # Get Delaunay edges (though we use all pairs in LP for safety, 
                # Delaunay might be useful for a sparser LP if N was larger, 
                # but for N=21 all pairs is fine and safer)
                
                rs_lp = _solve_lp_radii(xs_cur, ys_cur, w, h, None)
                
                if rs_lp is not None:
                    new_params = np.concatenate([xs_cur, ys_cur, rs_lp])
                    if _is_valid(new_params, w, h):
                        new_val = np.sum(rs_lp)
                        if new_val > current_val:
                            current_params = new_params
                            current_val = new_val
                            
                            # Phase 3: Re-optimize centers with the new radii as starting point
                            # This can sometimes unlock better center positions
                            try:
                                res2 = minimize(objective_and_grad, current_params, args=(w, h), 
                                               jac=True, method='L-BFGS-B',
                                               bounds=bounds,
                                               options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50})
                                if res2.success:
                                    current_params = res2.x
                                    # Another LP polish
                                    xs_cur2 = current_params[0:N].copy()
                                    ys_cur2 = current_params[N:2*N].copy()
                                    xs_cur2 = np.clip(xs_cur2, 1e-6, w - 1e-6)
                                    ys_cur2 = np.clip(ys_cur2, 1e-6, h - 1e-6)
                                    rs_lp2 = _solve_lp_radii(xs_cur2, ys_cur2, w, h, None)
                                    if rs_lp2 is not None:
                                        new_params2 = np.concatenate([xs_cur2, ys_cur2, rs_lp2])
                                        if _is_valid(new_params2, w, h):
                                            new_val2 = np.sum(rs_lp2)
                                            if new_val2 > current_val:
                                                current_params = new_params2
                                                current_val = new_val2
                            except Exception:
                                pass
                            
                # Check validity and update best
                if _is_valid(current_params, w, h):
                    final_val = np.sum(current_params[2*N:3*N])
                    if final_val > best_val:
                        best_val = final_val
                        best_params = current_params.copy()
                        best_w = w
                        
            except Exception:
                pass

    # Final Polish with high precision
    if best_params is not None and time.time() - start_time < time_limit - 5:
        w = best_w
        h = 2.0 - w
        max_r = min(w, h) / 2.0
        bounds = []
        for i in range(N):
            bounds.append((0.0, w))
        for i in range(N):
            bounds.append((0.0, h))
        for i in range(N):
            bounds.append((0.0, max_r))
            
        try:
            # Higher precision polish
            res = minimize(objective_and_grad, best_params, args=(w, h), 
                           jac=True, method='L-BFGS-B',
                           bounds=bounds,
                           options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-10, 'maxls': 100})
            
            if res.success:
                current_params = res.x
                # LP Polish
                xs_cur = current_params[0:N].copy()
                ys_cur = current_params[N:2*N].copy()
                xs_cur = np.clip(xs_cur, 1e-6, w - 1e-6)
                ys_cur = np.clip(ys_cur, 1e-6, h - 1e-6)
                rs_lp = _solve_lp_radii(xs_cur, ys_cur, w, h, None)
                if rs_lp is not None:
                    new_params = np.concatenate([xs_cur, ys_cur, rs_lp])
                    if _is_valid(new_params, w, h):
                        new_val = np.sum(rs_lp)
                        if new_val > best_val:
                            best_val = new_val
                            best_params = new_params.copy()
        except Exception:
            pass

    if best_params is not None:
        out = best_params.reshape(N, 3)
        out[:, 2] = np.maximum(0, out[:, 2])
        return out
    else:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
