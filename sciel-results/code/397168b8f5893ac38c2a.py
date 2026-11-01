import numpy as np
import scipy.spatial as spat
import scipy.optimize as opt
import time

N = 14
DEADLINE = 145  # 5s safety margin before 150s kill

def _triangle_min_area(points):
    """Compute minimum triangle area over all triples."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
    return min_area

def _hull_area(points):
    """Compute convex hull area."""
    try:
        hull = spat.ConvexHull(points)
        return hull.volume  # in 2D, volume is area
    except Exception:
        return 1e-10

def _score(points):
    """Score = min triangle area / hull area. Higher is better."""
    hull_a = _hull_area(points)
    if hull_a < 1e-10:
        return 0.0
    min_tri = _triangle_min_area(points)
    return min_tri / hull_a

def _neg_score(points):
    """Negative score for minimization."""
    return -_score(points)

def _heilbronn_rows(n):
    """
    Generate points in two rows, staggered to maximize minimum triangle area.
    This is a known heuristic for Heilbronn-type problems.
    Top row: x = i / (n_top - 1)
    Bottom row: x = (i + 0.5) / (n_bottom - 1) if n_bottom > 1 else 0.5
    """
    n_top = n // 2
    n_bottom = n - n_top
    
    top_xs = np.linspace(0, 1, n_top) if n_top > 1 else np.array([0.5])
    top_ys = np.zeros(n_top)
    
    # Stagger bottom row
    if n_bottom > 1:
        bottom_xs = np.linspace(0.5 / (n_bottom - 1), 1 - 0.5 / (n_bottom - 1), n_bottom)
    else:
        bottom_xs = np.array([0.5])
    bottom_ys = np.ones(n_bottom)
    
    xs = np.concatenate([top_xs, bottom_xs])
    ys = np.concatenate([top_ys, bottom_ys])
    
    # Center and scale to unit area hull
    pts = np.column_stack([xs, ys])
    hull_a = _hull_area(pts)
    if hull_a > 0:
        pts /= np.sqrt(hull_a)
    return pts

def _regular_polygon(n, radius=1.0, center=(0.0, 0.0)):
    """Generate n points on a regular polygon."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    xs = center[0] + radius * np.cos(angles)
    ys = center[1] + radius * np.sin(angles)
    return np.column_stack([xs, ys])

def _grid_2rows(n):
    """2 rows of points, alternating."""
    half = n // 2
    rem = n - half
    pts = []
    # Top row
    for i in range(rem):
        pts.append([i / max(rem - 1, 1), 1.0])
    # Bottom row
    for i in range(half):
        pts.append([i / max(half - 1, 1), 0.0])
    return np.array(pts)

def _hex_lattice_subset(n):
    """Subset of hexagonal lattice."""
    pts = []
    # Generate hex lattice points
    for row in range(-2, 3):
        for col in range(-2, 3):
            x = col + (row % 2) * 0.5
            y = row * 0.866
            pts.append([x, y])
    pts = np.array(pts)
    # Center and scale
    pts -= pts.mean(axis=0)
    scale = 1.0 / (np.abs(pts).max() + 1e-10)
    pts *= scale
    # Pick n closest to center or use all if n >= len(pts)
    if len(pts) >= n:
        dists = np.linalg.norm(pts, axis=1)
        idx = np.argsort(dists)[:n]
        pts = pts[idx]
    return pts

def construct():
    start_time = time.time()
    
    best_points = None
    best_score = 0.0
    
    # Initial configurations to try
    # 1. Heilbronn rows (staggered)
    # 2. Regular polygon
    # 3. Grid 2 rows (uniform)
    # 4. Hex lattice
    # 5. Random
    
    initial_configs = [
        _heilbronn_rows(N),
        _regular_polygon(N, radius=1.0),
        _grid_2rows(N),
        _hex_lattice_subset(N),
    ]
    
    rng = np.random.RandomState(42)
    for _ in range(3):
        pts = rng.random((N, 2))
        initial_configs.append(pts)
    
    for init_pts in initial_configs:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 30:  # Need at least 30s for optimization
            break
        
        # Normalize initial points to unit area hull
        hull_a = _hull_area(init_pts)
        if hull_a < 1e-10:
            continue
        init_pts = init_pts / np.sqrt(hull_a)  # scale so hull area ~ 1
        
        # Run Nelder-Mead optimization
        try:
            result = opt.minimize(
                _neg_score,
                init_pts,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
        except Exception:
            result = None
        
        if result is not None and result.fun is not None:
            score = -result.fun
            if score > best_score:
                best_score = score
                best_points = result.x
        
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 10:
            break
    
    # If we didn't find anything good, use the best initial config
    if best_points is None:
        best_points = initial_configs[0]
    
    # Final check and return
    return np.array(best_points)
