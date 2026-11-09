import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import itertools

_DEADLINE = time.time() + 600.0

def _min_triangle_area(points):
    n = len(points)
    idx = np.array(list(itertools.combinations(range(n), 3)))
    if len(idx) == 0:
        return 0.0
    a = points[idx[:, 0]]
    b = points[idx[:, 1]]
    c = points[idx[:, 2]]
    areas = 0.5 * np.abs(a[:, 0] * (b[:, 1] - c[:, 1])
                         + b[:, 0] * (c[:, 1] - a[:, 1])
                         + c[:, 0] * (a[:, 1] - b[:, 1]))
    return float(np.min(areas))

def _hull_area(points):
    try:
        hull = ConvexHull(points)
        return float(hull.volume)
    except Exception:
        return 0.0

def _score(points):
    ha = _hull_area(points)
    if ha <= 0:
        return 0.0
    ma = _min_triangle_area(points)
    if ma <= 0:
        return 0.0
    return ma / ha

def _regular_polygon(n, radius=0.5, center=(0.5, 0.5)):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([center[0] + radius * np.cos(angles),
                            center[1] + radius * np.sin(angles)])

def _hex_lattice(n, spacing=0.2, center=(0.5, 0.5)):
    """Generate n points from a hexagonal lattice centered at center."""
    pts = []
    # Hex lattice: rows offset by half spacing
    max_row = 5
    for row in range(-max_row, max_row + 1):
        for col in range(-max_row, max_row + 1):
            x = center[0] + col * spacing + (0.5 * spacing if row % 2 else 0)
            y = center[1] + row * spacing * np.sqrt(3) / 2
            if 0.05 <= x <= 0.95 and 0.05 <= y <= 0.95:
                pts.append([x, y])
    pts = np.array(pts)
    if len(pts) < n:
        # Pad with random points
        rng = np.random.default_rng(42)
        extra = rng.uniform(0.1, 0.9, (n - len(pts), 2))
        pts = np.vstack([pts, extra])
    # Take first n
    return pts[:n]

def _concentric_rings(n, center=(0.5, 0.5)):
    """Generate n points in concentric rings."""
    pts = []
    # Center
    pts.append([center[0], center[1]])
    # Ring 1: 6 points
    for i in range(6):
        angle = 2 * np.pi * i / 6
        pts.append([center[0] + 0.15 * np.cos(angle), center[1] + 0.15 * np.sin(angle)])
    # Ring 2: 6 more points
    for i in range(6):
        angle = 2 * np.pi * i / 6 + np.pi / 6
        pts.append([center[0] + 0.3 * np.cos(angle), center[1] + 0.3 * np.sin(angle)])
    pts = np.array(pts)
    if len(pts) < n:
        rng = np.random.default_rng(43)
        extra = rng.uniform(0.1, 0.9, (n - len(pts), 2))
        pts = np.vstack([pts, extra])
    return pts[:n]

def _soft_min_ratio(x, beta):
    """
    Objective: minimize -soft_min_ratio where soft_min_ratio = soft_min_area / hull_area.
    x is a flat array of 2n coordinates.
    """
    n = len(x) // 2
    pts = x.reshape(n, 2)
    
    # Check bounds
    if np.any(pts < 0.01) or np.any(pts > 0.99):
        return 1e10  # Penalty
    
    # Compute all triangle areas
    idx = np.array(list(itertools.combinations(range(n), 3)))
    a = pts[idx[:, 0]]
    b = pts[idx[:, 1]]
    c = pts[idx[:, 2]]
    areas = 0.5 * np.abs(a[:, 0] * (b[:, 1] - c[:, 1])
                         + b[:, 0] * (c[:, 1] - a[:, 1])
                         + c[:, 0] * (a[:, 1] - b[:, 1]))
    
    # Soft min via log-sum-exp
    # soft_min = -log(sum(exp(-beta * areas))) / beta
    # For numerical stability, subtract max
    max_area = np.max(areas)
    shifted = areas - max_area
    log_sum_exp = np.log(np.sum(np.exp(-beta * shifted))) + beta * max_area
    soft_min = -log_sum_exp / beta
    
    # Hull area
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
    except Exception:
        return 1e10
    
    if hull_area <= 0:
        return 1e10
    
    ratio = soft_min / hull_area
    return -ratio

def _soft_min_ratio_grad(x, beta):
    """
    Numerical gradient of _soft_min_ratio.
    """
    n = len(x) // 2
    eps = 1e-7
    grad = np.zeros_like(x)
    f0 = _soft_min_ratio(x, beta)
    for i in range(len(x)):
        xp = x.copy()
        xp[i] += eps
        fp = _soft_min_ratio(xp, beta)
        xm = x.copy()
        xm[i] -= eps
        fm = _soft_min_ratio(xm, beta)
        grad[i] = (fp - fm) / (2 * eps)
    return grad

def _optimize_slsqp(x0, beta_seq, max_iter=100, time_limit=30):
    """
    Run SLSQP optimization with progressive beta annealing.
    """
    x = x0.copy()
    n = len(x) // 2
    bounds = [(0.01, 0.99)] * (2 * n)
    
    for beta in beta_seq:
        if time.time() > _DEADLINE - 20:
            break
        start_time = time.time()
        try:
            result = minimize(
                _soft_min_ratio,
                x,
                args=(beta,),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-12, 'disp': False}
            )
            if result.success or result.fun < _soft_min_ratio(x, beta):
                x = result.x
        except Exception:
            pass
        # Check time
        if time.time() - start_time > time_limit:
            break
    
    return x

def heilbronn_convex13() -> np.ndarray:
    n = 13
    rng = np.random.default_rng(seed=12345)
    
    best_points = None
    best_score = -np.inf
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Regular 13-gon
    initializations.append(_regular_polygon(n, radius=0.48, center=(0.5, 0.5)))
    
    # 2. Hexagonal lattice
    initializations.append(_hex_lattice(n, spacing=0.2, center=(0.5, 0.5)))
    
    # 3. Concentric rings
    initializations.append(_concentric_rings(n, center=(0.5, 0.5)))
    
    # 4. Random points
    for _ in range(3):
        initializations.append(rng.uniform(0.1, 0.9, (n, 2)))
    
    # 5. 9-gon outer + 4 inner
    outer9 = _regular_polygon(9, radius=0.48, center=(0.5, 0.5))
    angles4 = np.linspace(0, 2 * np.pi, 4, endpoint=False)
    inner4 = np.column_stack([0.5 + 0.2 * np.cos(angles4), 0.5 + 0.2 * np.sin(angles4)])
    initializations.append(np.vstack([outer9, inner4]))
    
    # Beta sequence: start smooth, end sharp
    beta_seq = [20, 50, 100, 200, 500, 1000]
    
    for init_pts in initializations:
        if time.time() > _DEADLINE - 30:
            break
        
        x0 = init_pts.flatten()
        
        # Run SLSQP with progressive beta
        x_opt = _optimize_slsqp(x0, beta_seq, max_iter=80, time_limit=25)
        pts_opt = x_opt.reshape(n, 2)
        
        # Final polish with hard min using coordinate ascent
        # Quick coordinate ascent to polish
        for sweep in range(50):
            if time.time() > _DEADLINE - 10:
                break
            improved = False
            for i in range(n):
                x0_i, y0_i = pts_opt[i]
                best_x, best_y = x0_i, y0_i
                best_val = _min_triangle_area(pts_opt)
                
                for step in [0.02, 0.01, 0.005, 0.002]:
                    for dx, dy in [(step,0),(-step,0),(0,step),(0,-step),
                                   (step*0.707,step*0.707),(-step*0.707,step*0.707),
                                   (step*0.707,-step*0.707),(-step*0.707,-step*0.707)]:
                        nx, ny = x0_i + dx, y0_i + dy
                        if nx < 0.01 or nx > 0.99 or ny < 0.01 or ny > 0.99:
                            continue
                        new_pts = pts_opt.copy()
                        new_pts[i] = [nx, ny]
                        ma = _min_triangle_area(new_pts)
                        if ma > best_val + 1e-12:
                            best_val = ma
                            best_x, best_y = nx, ny
                            improved = True
                pts_opt[i] = [best_x, best_y]
            if not improved:
                break
        
        s = _score(pts_opt)
        if s > best_score:
            best_score = s
            best_points = pts_opt.copy()
    
    if best_points is None:
        best_points = _regular_polygon(n, radius=0.48, center=(0.5, 0.5))
    
    if not np.all(np.isfinite(best_points)):
        best_points = _regular_polygon(n, radius=0.48, center=(0.5, 0.5))
    
    return best_points
