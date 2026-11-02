import numpy as np
from scipy.optimize import minimize, minimize_scalar, linprog
from scipy.spatial import Delaunay
import time

N = 21

def _softplus(x, s=1.0):
    """Smooth approximation of max(0, x)."""
    return s * np.log1p(np.exp(x / s))

def _sigmoid(x, s=1.0):
    """Sigmoid function for smooth constraints."""
    return 1.0 / (1.0 + np.exp(-x / s))

def _delaunay_edges(points):
    """Get unique edges from Delaunay triangulation."""
    if len(points) < 3:
        return list(range(len(points)))
    try:
        tri = Delaunay(points)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i+1, 3):
                    a, b = simplex[i], simplex[j]
                    edges.add((min(a, b), max(a, b)))
        return list(edges)
    except Exception:
        # Fallback: all pairs if Delaunay fails
        return [(i, j) for i in range(len(points)) for j in range(i+1, len(points))]

def _solve_lp_radii(xs, ys, w, h, edges):
    """
    Solve LP to maximize sum of radii given positions.
    Variables: r_0, ..., r_{N-1}
    Maximize: sum(r_i)
    Subject to:
      r_i + r_j <= dist(i,j) for all edges (i,j)
      r_i <= x_i, r_i <= w - x_i, r_i <= y_i, r_i <= h - y_i
      r_i >= 0
    """
    n = N
    # Objective: maximize sum(r) => minimize -sum(r)
    c = -np.ones(n)
    
    # Inequality constraints: A_ub @ r <= b_ub
    A_ub_rows = []
    b_ub_rows = []
    
    # Pairwise constraints for edges
    for (i, j) in edges:
        dx = xs[i] - xs[j]
        dy = ys[i] - ys[j]
        dist = np.sqrt(dx*dx + dy*dy)
        row = np.zeros(n)
        row[i] = 1.0
        row[j] = 1.0
        A_ub_rows.append(row)
        b_ub_rows.append(dist)
    
    # Boundary constraints
    for i in range(n):
        # r_i <= x_i
        row = np.zeros(n)
        row[i] = 1.0
        A_ub_rows.append(row)
        b_ub_rows.append(xs[i])
        
        # r_i <= w - x_i
        row = np.zeros(n)
        row[i] = 1.0
        A_ub_rows.append(row)
        b_ub_rows.append(w - xs[i])
        
        # r_i <= y_i
        row = np.zeros(n)
        row[i] = 1.0
        A_ub_rows.append(row)
        b_ub_rows.append(ys[i])
        
        # r_i <= h - y_i
        row = np.zeros(n)
        row[i] = 1.0
        A_ub_rows.append(row)
        b_ub_rows.append(h - ys[i])
    
    if A_ub_rows:
        A_ub = np.array(A_ub_rows)
        b_ub = np.array(b_ub_rows)
    else:
        A_ub = None
        b_ub = None
    
    # Bounds: 0 <= r_i <= min(w, h)/2
    bounds = [(0.0, min(w, h)/2) for _ in range(n)]
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return -res.fun, res.x
        else:
            return 0.0, np.zeros(n)
    except Exception:
        return 0.0, np.zeros(n)

def _max_equal_radius(xs, ys, w, h):
    """Find maximum equal radius such that all circles fit."""
    # Binary search on radius
    lo = 0.0
    hi = min(w, h) / 2.0
    
    for _ in range(50):
        mid = (lo + hi) / 2.0
        valid = True
        # Check boundaries
        for i in range(N):
            if xs[i] < mid or w - xs[i] < mid or ys[i] < mid or h - ys[i] < mid:
                valid = False
                break
        if not valid:
            hi = mid
            continue
        # Check pairwise
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < 2.0 * mid - 1e-10:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            lo = mid
        else:
            hi = mid
    
    return lo

def _make_hex(w, h):
    """Create a hexagonal packing pattern."""
    positions = []
    r = min(w, h) / 10.0
    # Simple hex grid
    row = 0
    y = r
    while y <= h - r:
        x = r
        offset = 0 if row % 2 == 0 else r
        while x + offset <= w - r:
            positions.append([x + offset, y])
            x += 2 * r
        y += r * np.sqrt(3)
        row += 1
    # Pad or truncate to N
    while len(positions) < N:
        positions.append([w/2, h/2])
    return np.array(positions[:N])

def _make_grid(w, h):
    """Create a grid packing pattern."""
    rows = int(np.ceil(np.sqrt(N * h / w)))
    cols = int(np.ceil(N / rows))
    positions = []
    for i in range(N):
        row = i // cols
        col = i % cols
        x = (col + 0.5) * w / cols
        y = (row + 0.5) * h / rows
        positions.append([x, y])
    return np.array(positions)

def _make_random(w, h, rng):
    """Create random positions."""
    xs = rng.uniform(0.1, w - 0.1, N)
    ys = rng.uniform(0.1, h - 0.1, N)
    return np.column_stack([xs, ys])

def _refine_slsqp(xs, ys, w, h, time_budget):
    """
    Refine positions using SLSQP to maximize sum of radii.
    We parameterize: for each circle, position (x_i, y_i) and radius r_i.
    But LP gives optimal r for given positions, so we can use that.
    """
    start_time = time.time()
    
    def objective_and_grad(params):
        # params: [x_0, y_0, r_0, ..., x_{N-1}, y_{N-1}, r_{N-1}]
        pos = params.reshape(N, 3)
        xs_p = pos[:, 0]
        ys_p = pos[:, 1]
        rs_p = pos[:, 2]
        
        # Clip to valid range
        xs_p = np.clip(xs_p, 0, w)
        ys_p = np.clip(ys_p, 0, h)
        rs_p = np.clip(rs_p, 0, min(w, h)/2)
        
        # Get Delaunay edges
        points = np.column_stack([xs_p, ys_p])
        edges = _delaunay_edges(points)
        
        # Solve LP for optimal radii
        obj_val, rs_opt = _solve_lp_radii(xs_p, ys_p, w, h, edges)
        
        # We want to maximize obj_val, so minimize -obj_val
        # But we also need gradient w.r.t. positions
        # Approximate gradient by finite differences on a subset
        # For efficiency, use analytical gradient approximation
        
        # Actually, let's just return the objective and use SLSQP's numerical gradient
        # But that's expensive. Instead, let's use a different approach.
        
        # Return negative of LP objective (we're minimizing)
        return -obj_val
    
    # Initial guess
    params0 = np.zeros(N * 3)
    for i in range(N):
        params0[i*3] = xs[i]
        params0[i*3+1] = ys[i]
        params0[i*3+2] = _max_equal_radius(xs, ys, w, h)
    
    # Constraints for SLSQP
    def constraint_boundary(params):
        pos = params.reshape(N, 3)
        xs_p = pos[:, 0]
        ys_p = pos[:, 1]
        rs_p = pos[:, 2]
        # r_i <= x_i, r_i <= w - x_i, etc.
        cons = []
        for i in range(N):
            cons.append(rs_p[i] - xs_p[i])  # <= 0 => r_i - x_i <= 0
            cons.append(rs_p[i] - (w - xs_p[i]))
            cons.append(rs_p[i] - ys_p[i])
            cons.append(rs_p[i] - (h - ys_p[i]))
        return cons
    
    # This is getting complex. Let me use a simpler approach:
    # Optimize positions only, and compute radii via LP at the end.
    
    def pos_objective(pos_params):
        pos_params = pos_params.reshape(N, 2)
        xs_p = pos_params[:, 0]
        ys_p = pos_params[:, 1]
        
        # Clip
        xs_p = np.clip(xs_p, 0, w)
        ys_p = np.clip(ys_p, 0, h)
        
        points = np.column_stack([xs_p, ys_p])
        edges = _delaunay_edges(points)
        obj_val, _ = _solve_lp_radii(xs_p, ys_p, w, h, edges)
        return -obj_val
    
    pos0 = np.column_stack([xs, ys]).flatten()
    
    # Constraints: positions in [0, w] x [0, h]
    bounds = []
    for i in range(N):
        bounds.append((0, w))
        bounds.append((0, h))
    
    try:
        res = minimize(pos_objective, pos0, method='SLSQP', bounds=bounds,
                       options={'maxiter': 100, 'ftol': 1e-8})
        if res.success:
            pos_opt = res.x.reshape(N, 2)
            xs_opt = pos_opt[:, 0]
            ys_opt = pos_opt[:, 1]
            points = np.column_stack([xs_opt, ys_opt])
            edges = _delaunay_edges(points)
            _, rs_opt = _solve_lp_radii(xs_opt, ys_opt, w, h, edges)
            return xs_opt, ys_opt, rs_opt
    except Exception:
        pass
    
    return xs, ys, np.zeros(N)

def _polish_gradient(xs, ys, w, h, time_budget=5.0):
    """
    Polish using gradient descent on positions with LP-computed radii.
    """
    start_time = time.time()
    step_size = 0.01
    lr = 0.01
    
    for it in range(200):
        if time.time() - start_time > time_budget:
            break
        
        points = np.column_stack([xs, ys])
        edges = _delaunay_edges(points)
        obj_val, rs = _solve_lp_radii(xs, ys, w, h, edges)
        
        # Compute gradient of -obj_val w.r.t. positions
        # Use finite differences
        grad = np.zeros((N, 2))
        eps = 1e-5
        for i in range(N):
            for d in range(2):
                xs_p = xs.copy()
                ys_p = ys.copy()
                if d == 0:
                    xs_p[i] += eps
                else:
                    ys_p[i] += eps
                points_p = np.column_stack([xs_p, ys_p])
                edges_p = _delaunay_edges(points_p)
                obj_p, _ = _solve_lp_radii(xs_p, ys_p, w, h, edges_p)
                grad[i, d] = (obj_p - obj_val) / eps
        
        # Update
        xs -= lr * grad[:, 0]
        ys -= lr * grad[:, 1]
        
        # Clip
        xs = np.clip(xs, 0, w)
        ys = np.clip(ys, 0, h)
        
        # Decay learning rate
        lr *= 0.95
    
    points = np.column_stack([xs, ys])
    edges = _delaunay_edges(points)
    _, rs = _solve_lp_radii(xs, ys, w, h, edges)
    return xs, ys, rs

def _make_result_from_lp(xs, ys, w, h):
    """Create result array from positions using LP radii."""
    points = np.column_stack([xs, ys])
    edges = _delaunay_edges(points)
    _, rs = _solve_lp_radii(xs, ys, w, h, edges)
    return np.column_stack([xs, ys, rs])

def _make_result_equal(xs, ys, w, h):
    """Create result array with equal radii."""
    r = _max_equal_radius(xs, ys, w, h)
    return np.column_stack([xs, ys, np.full(N, r)])

def construct():
    start_time = time.time()
    time_limit = 140.0  # Safety margin before 150s kill
    
    best_val = 0.0
    best_result = None
    best_a = 1.0
    
    # Aspect ratio search
    a_values = np.linspace(0.7, 1.4, 25)
    
    rng = np.random.default_rng(42)
    
    for a in a_values:
        if time.time() - start_time > time_limit:
            break
        
        h = 2.0 / (1.0 + a)
        w = 2.0 * a / (1.0 + a)
        
        # Multiple initializations
        inits = []
        
        # Grid
        inits.append(_make_grid(w, h))
        
        # Hex
        inits.append(_make_hex(w, h))
        
        # Random
        for _ in range(5):
            inits.append(_make_random(w, h, rng))
        
        for pos_init in inits:
            if time.time() - start_time > time_limit:
                break
            
            xs = pos_init[:, 0]
            ys = pos_init[:, 1]
            
            # Polish with gradient
            xs, ys, rs = _polish_gradient(xs, ys, w, h, time_budget=3.0)
            
            # Try SLSQP refinement
            xs2, ys2, rs2 = _refine_slsqp(xs, ys, w, h, time_budget=5.0)
            
            # Evaluate both
            for (xs_e, ys_e, rs_e) in [(xs, ys, rs), (xs2, ys2, rs2)]:
                # Validate
                valid = True
                for i in range(N):
                    if rs_e[i] < -1e-8 or xs_e[i] < -1e-8 or w - xs_e[i] < -1e-8:
                        valid = False
                        break
                    if ys_e[i] < -1e-8 or h - ys_e[i] < -1e-8:
                        valid = False
                        break
                    if rs_e[i] > xs_e[i] + 1e-8 or rs_e[i] > w - xs_e[i] + 1e-8:
                        valid = False
                        break
                    if rs_e[i] > ys_e[i] + 1e-8 or rs_e[i] > h - ys_e[i] + 1e-8:
                        valid = False
                        break
                if valid:
                    for i in range(N):
                        for j in range(i+1, N):
                            dx = xs_e[i] - xs_e[j]
                            dy = ys_e[i] - ys_e[j]
                            dist = np.sqrt(dx*dx + dy*dy)
                            if dist < rs_e[i] + rs_e[j] - 1e-8:
                                valid = False
                                break
                        if not valid:
                            break
                
                if valid:
                    val = np.sum(rs_e)
                    if val > best_val:
                        best_val = val
                        best_result = np.column_stack([xs_e, ys_e, rs_e])
                        best_a = a
    
    # Final polish on best result
    if best_result is not None and time.time() - start_time < time_limit - 10:
        xs = best_result[:, 0]
        ys = best_result[:, 1]
        h = 2.0 / (1.0 + best_a)
        w = 2.0 * best_a / (1.0 + best_a)
        
        # Gradient polish with more iterations
        xs, ys, rs = _polish_gradient(xs, ys, w, h, time_budget=10.0)
        
        # Validate
        valid = True
        for i in range(N):
            if rs[i] < -1e-8 or xs[i] < -1e-8 or w - xs[i] < -1e-8:
                valid = False
                break
            if ys[i] < -1e-8 or h - ys[i] < -1e-8:
                valid = False
                break
            if rs[i] > xs[i] + 1e-8 or rs[i] > w - xs[i] + 1e-8:
                valid = False
                break
            if rs[i] > ys[i] + 1e-8 or rs[i] > h - ys[i] + 1e-8:
                valid = False
                break
        if valid:
            for i in range(N):
                for j in range(i+1, N):
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < rs[i] + rs[j] - 1e-8:
                        valid = False
                        break
                if not valid:
                    break
        
        if valid:
            val = np.sum(rs)
            if val > best_val:
                best_result = np.column_stack([xs, ys, rs])
    
    if best_result is not None:
        # Ensure non-negative radii
        best_result[:, 2] = np.maximum(0, best_result[:, 2])
        return best_result
    else:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
