import numpy as np
from scipy.optimize import minimize, linprog
from scipy.spatial import Delaunay
import time

N = 21

def _softplus(x, k=10.0):
    """Numerically stable softplus."""
    return np.where(x > 0, x + np.log1p(np.exp(-np.abs(x)))/k, np.log1p(np.exp(x))/k)

def _delaunay_edges(pts):
    """Get unique edge pairs from Delaunay triangulation."""
    try:
        tri = Delaunay(pts)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                a, b = simplex[i], simplex[(i+1) % 3]
                edges.add((min(a,b), max(a,b)))
        return list(edges)
    except Exception:
        return []

def _solve_lp_radii(x, y, w, h, edges):
    """
    Solve LP: maximize sum(r_i) subject to:
    - For each edge (i,j): |x_i-x_j|*r_i + |y_i-y_j|*r_j >= ... 
      Actually linearize: (x_j-x_i)*(r_i+r_j) ... no.
      
    Better: use the standard tangent-circle linearization.
    For circles i,j with fixed centers, the constraint is:
      (x_i-x_j)^2 + (y_i-y_j)^2 >= (r_i + r_j)^2
    This is non-linear. But we can use a first-order approximation or
    the "supporting hyperplane" approach.
    
    Alternative: use the fact that for fixed centers, we want to maximize
    sum(r) s.t. r_i + r_j <= d_ij for all pairs. This IS linear!
    The constraint is simply r_i + r_j <= d_ij where d_ij is the center distance.
    """
    n = len(x)
    # c_i = r_i (all positive)
    # max sum(r)
    # s.t. r_i + r_j <= d_ij for all i < j
    #      r_i >= 0
    #      r_i <= min(w, h) / 2 (box constraint)
    #      x_i - r_i >= 0, x_i + r_i <= w, y_i - r_i >= 0, y_i + r_i <= h
    
    # Build constraint matrix
    # Only use edges from Delaunay + nearby pairs for efficiency
    # But for correctness, use ALL pairs (N=21, so 210 pairs)
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
            pairs.append((i, j, d))
    
    m = len(pairs)
    A_ub = np.zeros((m, n))
    b_ub = np.zeros(m)
    for k, (i, j, d) in enumerate(pairs):
        A_ub[k, i] = 1.0
        A_ub[k, j] = 1.0
        b_ub[k] = d
    
    # Box constraints: 0 <= r_i <= min(w,h)/2
    r_max = min(w, h) / 2.0
    # Also from boundary: r_i <= x_i, r_i <= w - x_i, r_i <= y_i, r_i <= h - y_i
    r_bound = np.minimum(r_max, np.minimum(np.minimum(x, w - x), np.minimum(y, h - y)))
    r_bound = np.maximum(r_bound, 0.0)
    
    bounds = [(0.0, r_bound[i]) for i in range(n)]
    c = -np.ones(n)  # minimize -sum(r)
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.x
        else:
            return None
    except Exception:
        return None

def _make_hex(w, h):
    """Generate hexagonal packing initial configuration."""
    pts = []
    r_est = 0.15
    dy = 2 * r_est
    dx = np.sqrt(3) * r_est
    row = 0
    while True:
        y = r_est + row * dy
        if y + r_est > h:
            break
        offset = (dx / 2) if row % 2 == 0 else 0
        x = r_est + offset
        col = 0
        while True:
            if x + r_est > w:
                break
            pts.append([x, y])
            x += dx
            col += 1
        row += 1
    # Pad or trim to N
    while len(pts) < N:
        pts.append([w/2, h/2])
    pts = pts[:N]
    return np.array(pts)

def _make_grid(w, h):
    """Generate grid initial configuration."""
    cols = int(np.ceil(np.sqrt(N * w / h)))
    rows = int(np.ceil(N / cols))
    pts = []
    for i in range(rows):
        for j in range(cols):
            if len(pts) >= N:
                break
            x = (j + 0.5) * w / cols
            y = (i + 0.5) * h / rows
            pts.append([x, y])
    while len(pts) < N:
        pts.append([w/2, h/2])
    return np.array(pts[:N])

def _make_random(w, h, seed=0):
    """Generate random initial configuration."""
    rng = np.random.RandomState(seed)
    pts = rng.uniform(0.1, np.array([w-0.1, h-0.1]), size=(N, 2))
    return pts

def _refine_slsqp(x, y, r, w, h, time_remaining):
    """Refine with SLSQP using hard constraints."""
    n = N
    params0 = np.concatenate([x, y, r])
    
    def obj(p):
        x_ = p[:n]
        y_ = p[n:2*n]
        r_ = np.maximum(p[2*n:], 1e-8)
        return -np.sum(r_)
    
    def jac(p):
        x_ = p[:n]
        y_ = p[n:2*n]
        r_ = np.maximum(p[2*n:], 1e-8)
        return np.concatenate([-np.ones(n), -np.ones(n), -np.ones(n)])
    
    # Constraints: non-overlap (equality at boundary for active set, inequality otherwise)
    # For SLSQP, use inequality: dist - (r_i + r_j) >= 0
    def make_overlap_constraints():
        cons = []
        for i in range(n):
            for j in range(i+1, n):
                def c(p, i=i, j=j):
                    x_ = p[:n]
                    y_ = p[n:2*n]
                    r_ = np.maximum(p[2*n:], 1e-8)
                    d = np.sqrt((x_[i]-x_[j])**2 + (y_[i]-y_[j])**2)
                    return d - (r_[i] + r_[j])
                cons.append({'type': 'ineq', 'fun': c})
        # Boundary constraints
        for i in range(n):
            # x_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[i] - np.maximum(p[2*n+i], 1e-8)})
            # w - x_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda p, i=i: w - p[i] - np.maximum(p[2*n+i], 1e-8)})
            # y_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[n+i] - np.maximum(p[2*n+i], 1e-8)})
            # h - y_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda p, i=i: h - p[n+i] - np.maximum(p[2*n+i], 1e-8)})
        return cons
    
    cons = make_overlap_constraints()
    
    bounds = []
    for i in range(n):
        bounds.append((0.0, w))       # x
    for i in range(n):
        bounds.append((0.0, h))       # y
    for i in range(n):
        bounds.append((0.0, min(w,h)/2))  # r
    
    try:
        res = minimize(obj, params0, jac=jac, method='SLSQP', 
                       bounds=bounds, constraints=cons,
                       options={'maxiter': 200, 'ftol': 1e-10, 'iprint': 0})
        if res.success or res.fun < -np.sum(r):
            x_new = res.x[:n]
            y_new = res.x[n:2*n]
            r_new = np.maximum(res.x[2*n:], 0.0)
            return x_new, y_new, r_new
    except Exception:
        pass
    
    return x, y, r

def _scale_to_perimeter(x, y, r):
    """Scale coordinates so bounding box perimeter is exactly 4."""
    min_x = np.min(x - r)
    max_x = np.max(x + r)
    min_y = np.min(y - r)
    max_y = np.max(y + r)
    W = max_x - min_x
    H = max_y - min_y
    perim = 2 * (W + H)
    if perim <= 0:
        return x, y, r
    scale = 4.0 / perim
    # Scale about center
    cx = (min_x + max_x) / 2
    cy = (min_y + max_y) / 2
    x_new = (x - cx) * scale + cx
    y_new = (y - cy) * scale + cy
    r_new = r * scale
    return x_new, y_new, r_new

def _validate(x, y, r, tol=1e-7):
    """Check if configuration is valid (disjoint, in some rect with perimeter <= 4)."""
    if np.any(r < -tol):
        return False
    r = np.maximum(r, 0.0)
    
    # Check overlaps
    for i in range(N):
        for j in range(i+1, N):
            d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
            if d < r[i] + r[j] - tol:
                return False
    
    # Check bounding box perimeter
    min_x = np.min(x - r)
    max_x = np.max(x + r)
    min_y = np.min(y - r)
    max_y = np.max(y + r)
    W = max_x - min_x
    H = max_y - min_y
    if 2*(W + H) > 4.0 + 1e-6:
        return False
    
    return True

def _score(x, y, r):
    """Compute the score: sum of radii, after ensuring perimeter constraint."""
    r = np.maximum(r, 0.0)
    min_x = np.min(x - r)
    max_x = np.max(x + r)
    min_y = np.min(y - r)
    max_y = np.max(y + r)
    W = max_x - min_x
    H = max_y - min_y
    perim = 2*(W + H)
    if perim > 4.0:
        # Scale down
        scale = 4.0 / perim
        cx = (min_x + max_x) / 2
        cy = (min_y + max_y) / 2
        x = (x - cx) * scale + cx
        y = (y - cy) * scale + cy
        r = r * scale
    return np.sum(r)

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin (150s kill)
    
    best_score = -np.inf
    best_result = None
    
    # Aspect ratios to try
    aspect_ratios = [1.0, 1.2, 0.8, 1.5, 0.67, 1.33, 0.5, 2.0, 1.1, 0.9]
    
    # Generate diverse initial configurations
    configs = []
    for idx, w in enumerate(aspect_ratios):
        h = 2.0 - w
        if h <= 0.1 or w <= 0.1:
            continue
        configs.append(('hex', _make_hex(w, h), w, h))
        configs.append(('grid', _make_grid(w, h), w, h))
        configs.append(('rand', _make_random(w, h, seed=idx*100), w, h))
    
    # Also add some random aspect ratios
    rng = np.random.RandomState(42)
    for idx in range(10):
        w = rng.uniform(0.3, 1.7)
        h = 2.0 - w
        if h <= 0.1:
            continue
        configs.append(('rand2', _make_random(w, h, seed=500+idx), w, h))
    
    # Phase 1: LP-based radii optimization for each config
    for config_type, pts, w, h in configs:
        if time.time() - start_time > time_limit:
            break
        
        x, y = pts[:, 0].copy(), pts[:, 1].copy()
        
        # Get Delaunay edges for potential contacts
        edges = _delaunay_edges(pts)
        
        # Solve LP for optimal radii
        r = _solve_lp_radii(x, y, w, h, edges)
        if r is None:
            # Fallback: equal radii based on minimum distance
            min_d = np.inf
            for i in range(N):
                for j in range(i+1, N):
                    d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                    min_d = min(min_d, d)
            r = np.full(N, min_d / 2.0 * 0.95)
            # Also respect boundaries
            r = np.minimum(r, np.minimum(np.minimum(x, w-x), np.minimum(y, h-y)))
        
        r = np.maximum(r, 0.0)
        
        # Scale to perimeter 4
        x_s, y_s, r_s = _scale_to_perimeter(x, y, r)
        
        # Validate and score
        if _validate(x_s, y_s, r_s):
            s = _score(x_s, y_s, r_s)
            if s > best_score:
                best_score = s
                best_result = np.column_stack([x_s, y_s, r_s])
    
    # Phase 2: SLSQP refinement on top candidates
    # Re-run LP for best configs and refine
    for config_type, pts, w, h in configs[:15]:  # Limit to save time
        if time.time() - start_time > time_limit - 30:
            break
        
        x, y = pts[:, 0].copy(), pts[:, 1].copy()
        r = _solve_lp_radii(x, y, w, h, None)
        if r is None:
            min_d = np.inf
            for i in range(N):
                for j in range(i+1, N):
                    d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                    min_d = min(min_d, d)
            r = np.full(N, min_d / 2.0 * 0.95)
            r = np.minimum(r, np.minimum(np.minimum(x, w-x), np.minimum(y, h-y)))
        r = np.maximum(r, 1e-6)
        
        # SLSQP refinement
        x_new, y_new, r_new = _refine_slsqp(x, y, r, w, h, time_limit - (time.time() - start_time))
        
        # Scale to perimeter 4
        x_s, y_s, r_s = _scale_to_perimeter(x_new, y_new, r_new)
        
        if _validate(x_s, y_s, r_s):
            s = _score(x_s, y_s, r_s)
            if s > best_score:
                best_score = s
                best_result = np.column_stack([x_s, y_s, r_s])
    
    # Phase 3: Joint position+radius optimization with SLSQP from best LP solutions
    if time.time() - start_time < time_limit - 20 and best_result is not None:
        x0 = best_result[:, 0]
        y0 = best_result[:, 1]
        r0 = best_result[:, 2]
        
        # Infer bounding box
        min_x = np.min(x0 - r0)
        max_x = np.max(x0 + r0)
        min_y = np.min(y0 - r0)
        max_y = np.max(y0 + r0)
        w0 = max_x - min_x
        h0 = max_y - min_y
        # Shift to [0,w]x[0,h]
        x0 = x0 - min_x
        y0 = y0 - min_y
        
        def obj_joint(p):
            x_ = p[:N]
            y_ = p[N:2*N]
            r_ = np.maximum(p[2*N:], 1e-8)
            return -np.sum(r_)
        
        cons = []
        for i in range(N):
            for j in range(i+1, N):
                def c(p, i=i, j=j):
                    x_ = p[:N]
                    y_ = p[N:2*N]
                    r_ = np.maximum(p[2*N:], 1e-8)
                    d = np.sqrt((x_[i]-x_[j])**2 + (y_[i]-y_[j])**2)
                    return d - (r_[i] + r_[j])
                cons.append({'type': 'ineq', 'fun': c})
        for i in range(N):
            cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[i] - np.maximum(p[2*N+i], 1e-8)})
            cons.append({'type': 'ineq', 'fun': lambda p, i=i: w0 - p[i] - np.maximum(p[2*N+i], 1e-8)})
            cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[N+i] - np.maximum(p[2*N+i], 1e-8)})
            cons.append({'type': 'ineq', 'fun': lambda p, i=i: h0 - p[N+i] - np.maximum(p[2*N+i], 1e-8)})
        
        bounds = [(0.0, w0)] * N + [(0.0, h0)] * N + [(0.0, min(w0,h0)/2)] * N
        
        try:
            res = minimize(obj_joint, np.concatenate([x0, y0, r0]), method='SLSQP',
                          bounds=bounds, constraints=cons,
                          options={'maxiter': 300, 'ftol': 1e-10, 'iprint': 0})
            if res.fun < -best_score:
                x_n = res.x[:N]
                y_n = res.x[N:2*N]
                r_n = np.maximum(res.x[2*N:], 0.0)
                x_s, y_s, r_s = _scale_to_perimeter(x_n, y_n, r_n)
                if _validate(x_s, y_s, r_s):
                    s = _score(x_s, y_s, r_s)
                    if s > best_score:
                        best_score = s
                        best_result = np.column_stack([x_s, y_s, r_s])
        except Exception:
            pass
    
    # Fallback
    if best_result is None:
        # Simple grid fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        best_result = np.array(cells[:N])
        # Scale to perimeter 4
        x, y, r = best_result[:,0], best_result[:,1], best_result[:,2]
        best_result = np.column_stack(_scale_to_perimeter(x, y, r))
    
    return best_result
