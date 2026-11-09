import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _radius_score(centers):
    """Compute the radius score for a given set of centers."""
    if len(centers) != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    
    # Check bounds
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return 0.0
        
    # Boundary distances
    d_left = centers[:, 0] + 0.5
    d_right = 0.5 - centers[:, 0]
    d_bottom = centers[:, 1] + 0.5
    d_top = 0.5 - centers[:, 1]
    
    min_boundary = np.min(np.min(np.stack([d_left, d_right, d_bottom, d_top]), axis=0))
    
    # Pairwise distances
    min_pairwise = np.inf
    for i in range(N):
        for j in range(i + 1, N):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < min_pairwise:
                min_pairwise = dist
                
    min_pairwise_radius = min_pairwise / 2.0
    
    return min(min_boundary, min_pairwise_radius)

def _objective(x, t_init):
    """
    Objective function for SLSQP.
    We want to maximize t.
    The constraints enforce that all distances are >= 2*t.
    The objective is to minimize -t (i.e., maximize t).
    
    x = [x0, y0, ..., x49, y49, t]
    """
    t = x[-1]
    return -t

def _make_constraints():
    """
    Generate constraint functions.
    SLSQP expects a list of dictionaries: {'type': 'ineq', 'fun': lambda x: ...}
    fun(x) >= 0
    """
    cons = []
    
    # 1. Pairwise constraints: dist(i, j) - 2*t >= 0
    # dist(i, j) = sqrt((xi-xj)^2 + (yi-yj)^2)
    # Constraint: sqrt(...) - 2*t >= 0
    for i in range(N):
        for j in range(i + 1, N):
            def con_pair(x, i=i, j=j):
                dx = x[2*i] - x[2*j]
                dy = x[2*i+1] - x[2*j+1]
                dist = np.sqrt(dx*dx + dy*dy)
                t = x[-1]
                return dist - 2.0 * t
            cons.append({'type': 'ineq', 'fun': con_pair})
            
    # 2. Boundary constraints:
    # Left: x_i + 0.5 - 2*t >= 0
    # Right: 0.5 - x_i - 2*t >= 0
    # Bottom: y_i + 0.5 - 2*t >= 0
    # Top: 0.5 - y_i - 2*t >= 0
    
    for i in range(N):
        # Left
        def con_left(x, i=i):
            return x[2*i] + 0.5 - 2.0 * x[-1]
        cons.append({'type': 'ineq', 'fun': con_left})
        
        # Right
        def con_right(x, i=i):
            return 0.5 - x[2*i] - 2.0 * x[-1]
        cons.append({'type': 'ineq', 'fun': con_right})
        
        # Bottom
        def con_bottom(x, i=i):
            return x[2*i+1] + 0.5 - 2.0 * x[-1]
        cons.append({'type': 'ineq', 'fun': con_bottom})
        
        # Top
        def con_top(x, i=i):
            return 0.5 - x[2*i+1] - 2.0 * x[-1]
        cons.append({'type': 'ineq', 'fun': con_top})
        
    return cons

def _hex_init(m, n, offset_x=0.0, offset_y=0.0):
    """
    Generate a hexagonal grid initialization.
    m: columns, n: rows.
    """
    points = []
    # Approximate spacing
    # For N=50, 7x8 grid is 56, 8x7 is 56.
    # Hex grid is denser.
    # Let's just use a grid for simplicity and reliability, or a proper hex.
    # A proper hex grid for 50 circles:
    # Row 0: 8 circles
    # Row 1: 7 circles (offset)
    # ...
    # 8+7+8+7+8+7 = 45. Need 5 more.
    # Let's stick to a simple grid or random for robustness, 
    # but the prompt says "hex_init" in inspirations.
    # Let's generate a hex grid that fits 50 points.
    
    # Spacing
    # Square side 1.
    # If we use 8 columns, spacing ~ 1/8 = 0.125.
    # Hex height = 0.125 * sqrt(3)/2 ~ 0.108.
    # Rows: 1 / 0.108 ~ 9.2 rows.
    
    # Let's generate a grid of points and take the first 50?
    # Or just use a standard grid.
    # Actually, a grid is a valid initialization.
    
    # Let's do a grid of 7x8 = 56 points, take 50.
    # Or 8x7.
    
    # Let's implement a generic grid init for reliability.
    pts = []
    for i in range(n):
        for j in range(m):
            if len(pts) >= N:
                break
            # Center of cell
            x = (j + 0.5) / m - 0.5
            y = (i + 0.5) / n - 0.5
            # Add small random jitter to break symmetry? 
            # No, deterministic is better for reproducibility, 
            # but symmetry breaking helps optimization.
            # SLSQP might get stuck in symmetric local minima.
            # We will add jitter in _construct.
            pts.append([x, y])
    return np.array(pts[:N])

def _optimize_from(init_centers, time_limit=10.0):
    """
    Run SLSQP optimization from a given initialization.
    """
    # Flatten centers
    x0 = np.zeros(N * 2 + 1)
    x0[:N*2] = init_centers.flatten()
    
    # Initial guess for t
    # Estimate radius from init
    r_init = _radius_score(init_centers)
    t0 = r_init
    
    # If r_init is 0, set a small positive value
    if t0 <= 0:
        t0 = 0.01
        
    x0[-1] = t0
    
    constraints = _make_constraints()
    
    # Bounds for x, y: [-0.5, 0.5]
    # Bounds for t: [0, 0.5] (max possible radius is 0.5 / 2? No, max radius is 0.5 if 1 circle. For 50, it's small. But 0.5 is a safe upper bound for t? 
    # If t=0.5, 2t=1. Distances must be >= 1. In a unit square, max distance is sqrt(2)~1.41. 
    # So t can be up to 0.5.
    bounds = [(-0.5, 0.5)] * (N * 2) + [(0.0, 0.5)]
    
    start_time = time.time()
    
    # SLSQP options
    # maxiter: limit iterations to avoid hanging
    # ftol: tolerance on objective
    options = {
        'maxiter': 1000,
        'ftol': 1e-9,
        'disp': False
    }
    
    # Run minimize
    # We might need to handle time limit. SLSQP doesn't have a time limit option directly.
    # We can run it and hope it's fast, or use a wrapper.
    # For N=50, constraints are ~1300. SLSQP might be slow.
    # Let's try. If it's too slow, we need a different solver or approach.
    
    # Alternative: Use a simpler local search if SLSQP is too slow.
    # But SLSQP is generally good for this.
    
    try:
        res = minimize(_objective, x0, method='SLSQP', bounds=bounds, 
                       constraints=constraints, options=options)
    except Exception:
        # Fallback to current best
        return init_centers, r_init
        
    x_opt = res.x
    centers_opt = x_opt[:N*2].reshape(N, 2)
    t_opt = x_opt[-1]
    
    # Clip centers to valid range to ensure validity
    centers_opt = np.clip(centers_opt, -0.5, 0.5)
    
    # Calculate actual score
    r_opt = _radius_score(centers_opt)
    
    return centers_opt, r_opt

def _construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    best_centers = None
    best_radius = -np.inf
    
    # Generate initializations
    inits = []
    
    # 1. Grid 7x8
    inits.append(_hex_init(7, 8))
    # 2. Grid 8x7
    inits.append(_hex_init(8, 7))
    
    # 3. Random initializations
    rng = np.random.default_rng(42)
    for _ in range(5):
        # Uniform random
        inits.append(rng.uniform(-0.45, 0.45, size=(N, 2)))
        
    # 4. Hexagonal-ish pattern (manually constructed)
    # Try to place 50 circles in a hex pattern
    # Spacing s.
    # Let's just use the grid ones with jitter.
    
    # Add jitter to all inits to break symmetry
    for i, init in enumerate(inits):
        # Small jitter
        jitter = rng.normal(0, 0.01, size=(N, 2))
        init = init + jitter
        init = np.clip(init, -0.49, 0.49)
        inits[i] = init
        
    # Run optimization on each init
    for i, init in enumerate(inits):
        if time.time() > deadline:
            break
            
        remaining = deadline - time.time()
        # Allocate time for this run
        # SLSQP is deterministic, so we don't need to split time, 
        # but we should check if it's taking too long.
        # Since we can't interrupt SLSQP easily, we rely on maxiter.
        
        try:
            centers, r = _optimize_from(init)
            if r > best_radius:
                best_radius = r
                best_centers = centers
        except Exception:
            pass
            
    # If we have time, do a polish on the best result
    if best_centers is not None and time.time() < deadline - 5.0:
        try:
            # Polish with tighter tolerance
            # We can't easily change tolerance in _optimize_from without refactoring
            # But we can just run it again. It should converge quickly if already good.
            centers, r = _optimize_from(best_centers)
            if r > best_radius:
                best_radius = r
                best_centers = centers
        except Exception:
            pass
            
    # Fallback
    if best_centers is None:
        best_centers = _hex_init(7, 8)
        
    # Final validity check and clipping
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers

def construct():
    return _construct()
