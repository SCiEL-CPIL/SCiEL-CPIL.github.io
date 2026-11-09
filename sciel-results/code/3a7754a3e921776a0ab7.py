# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import itertools

_START_TIME = time.time()
_DEADLINE = _START_TIME + 580.0  # 50s safety margin before 630s limit


def _min_triangle_area(points):
    """Compute the minimum area among all triangles formed by triples of points."""
    n = len(points)
    if n < 3:
        return 0.0
    indices = np.array(list(itertools.combinations(range(n), 3)))
    p1 = points[indices[:, 0]]
    p2 = points[indices[:, 1]]
    p3 = points[indices[:, 2]]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return float(np.min(areas))


def _hull_area(points):
    """Compute the area of the convex hull of the points."""
    try:
        if np.allclose(points[:, 0], points[0, 0]) or np.allclose(points[:, 1], points[0, 1]):
            return 0.0
        hull = ConvexHull(points)
        return float(hull.volume)
    except Exception:
        return 0.0


def _score(points):
    """Compute A_min / A_hull."""
    if not np.all(np.isfinite(points)):
        return 0.0
    a_min = _min_triangle_area(points)
    a_hull = _hull_area(points)
    if a_hull <= 1e-12 or a_min <= 1e-12:
        return 0.0
    return a_min / a_hull


def _soft_min_triangle_area(points, beta=20.0):
    """
    Smooth approximation of the minimum triangle area using log-sum-exp.
    As beta -> infinity, this approaches the true min.
    """
    n = len(points)
    if n < 3:
        return 0.0
    indices = np.array(list(itertools.combinations(range(n), 3)))
    p1 = points[indices[:, 0]]
    p2 = points[indices[:, 1]]
    p3 = points[indices[:, 2]]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Soft min: -log-sum-exp(-beta * areas) / beta
    # This is a smooth lower bound that approaches min(areas) as beta increases
    max_a = np.max(areas)
    if max_a <= 1e-15:
        return 0.0
    log_sum = np.log(np.sum(np.exp(-beta * (areas - max_a)))) + beta * max_a
    soft_min = np.exp(log_sum) / beta
    return float(soft_min)


def _soft_ratio(points, beta=20.0):
    """Compute soft_min_area / hull_area."""
    if not np.all(np.isfinite(points)):
        return 0.0
    a_soft = _soft_min_triangle_area(points, beta=beta)
    a_hull = _hull_area(points)
    if a_hull <= 1e-12 or a_soft <= 1e-12:
        return 0.0
    return a_soft / a_hull


def _neg_soft_ratio(x, beta=20.0):
    """Objective for minimization: negative soft ratio."""
    if time.time() > _DEADLINE - 15:
        return 1e10
    points = x.reshape(-1, 2)
    s = _soft_ratio(points, beta=beta)
    if s < 1e-12:
        return 1e10
    return -s


def _neg_exact_ratio(x):
    """Objective for minimization: negative exact ratio."""
    if time.time() > _DEADLINE - 15:
        return 1e10
    points = x.reshape(-1, 2)
    s = _score(points)
    if s < 1e-12:
        return 1e10
    return -s


def _regular_polygon(n, radius=1.0, center=(0.0, 0.0)):
    """Generate a regular n-gon."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([center[0] + radius * np.cos(angles),
                            center[1] + radius * np.sin(angles)])


def _hexagonal_grid_points(n, spacing=1.0):
    """
    Generate points in a hexagonal (close-packed) pattern.
    For n=13, we can use a hexagonal arrangement.
    """
    if n == 13:
        # Hexagonal arrangement: 1 center + 6 in first ring + 6 in second ring
        # Or: 3x3 grid with corners removed + extras
        # Better: use a known good hexagonal packing
        # Ring 0: 1 point at center
        # Ring 1: 6 points at distance 1
        # Ring 2: 6 points at distance 2 (alternating)
        pts = []
        pts.append([0.0, 0.0])
        # Ring 1: 6 points at 60-degree intervals
        for k in range(6):
            angle = k * np.pi / 3
            pts.append([spacing * np.cos(angle), spacing * np.sin(angle)])
        # Ring 2: 6 points, offset by 30 degrees, at distance ~2
        for k in range(6):
            angle = k * np.pi / 3 + np.pi / 6
            r = spacing * np.sqrt(3)
            pts.append([r * np.cos(angle), r * np.sin(angle)])
        return np.array(pts[:n])
    # Fallback for other n
    return _regular_polygon(n, radius=1.0)


def _generate_structured_configs():
    """
    Generate diverse structured initial configurations based on known
    optimal topologies for the Heilbronn problem.
    """
    configs = []
    
    # Config 1: Regular 13-gon
    configs.append(_regular_polygon(13, radius=1.0))
    
    # Config 2: Hexagonal packing (13 points in hex pattern)
    configs.append(_hexagonal_grid_points(13, spacing=1.0))
    
    # Config 3: 7 outer (heptagon) + 6 inner (hexagon)
    outer = _regular_polygon(7, radius=1.0)
    inner = _regular_polygon(6, radius=0.45)
    # Rotate inner to interleave
    angle = np.pi / 7
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    inner = inner @ rot.T
    configs.append(np.vstack([outer, inner]))
    
    # Config 4: 8 outer + 5 inner
    outer = _regular_polygon(8, radius=1.0)
    inner = _regular_polygon(5, radius=0.4)
    angle = np.pi / 8
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    inner = inner @ rot.T
    configs.append(np.vstack([outer, inner]))
    
    # Config 5: 6 outer + 7 inner
    outer = _regular_polygon(6, radius=1.0)
    inner = _regular_polygon(7, radius=0.5)
    angle = np.pi / 6
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    inner = inner @ rot.T
    configs.append(np.vstack([outer, inner]))
    
    # Config 6: 9 outer + 4 inner
    outer = _regular_polygon(9, radius=1.0)
    inner = _regular_polygon(4, radius=0.45)
    angle = np.pi / 9
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    inner = inner @ rot.T
    configs.append(np.vstack([outer, inner]))
    
    # Config 7: Square-ish grid (4x4 minus one corner)
    xs = [-1, -0.33, 0.33, 1.0]
    ys = [-1, -0.33, 0.33, 1.0]
    grid = []
    for x in xs:
        for y in ys:
            grid.append([x, y])
    # Take 13 points: remove one corner
    grid = grid[:13]
    configs.append(np.array(grid))
    
    # Config 8: Concentric rings with 4+4+5
    ring1 = _regular_polygon(4, radius=0.3)
    ring2 = _regular_polygon(4, radius=0.6)
    ring3 = _regular_polygon(5, radius=1.0)
    angle1 = np.pi / 4
    rot1 = np.array([[np.cos(angle1), -np.sin(angle1)], [np.sin(angle1), np.cos(angle1)]])
    ring2 = ring2 @ rot1.T
    configs.append(np.vstack([ring1, ring2, ring3]))
    
    # Config 9: Triangular lattice patch
    # Points in a triangular lattice
    s = 0.7
    pts = []
    for i in range(4):
        for j in range(4):
            x = i * s + (j % 2) * s / 2
            y = j * s * np.sqrt(3) / 2
            pts.append([x, y])
    # Take 13
    pts = pts[:13]
    # Center them
    center = np.mean(pts, axis=0)
    pts = np.array(pts) - center
    configs.append(pts)
    
    # Config 10: Another hexagonal variant
    s = 0.8
    pts = []
    # Center
    pts.append([0.0, 0.0])
    # First ring: 6 points
    for k in range(6):
        angle = k * np.pi / 3
        pts.append([s * np.cos(angle), s * np.sin(angle)])
    # Second ring: 6 points at distance 2s
    for k in range(6):
        angle = k * np.pi / 3
        pts.append([2 * s * np.cos(angle), 2 * s * np.sin(angle)])
    pts = pts[:13]
    configs.append(np.array(pts))
    
    return configs


def _refine_soft(x0, beta=20.0, maxiter=300):
    """
    Refine using Powell's method with the soft-min surrogate objective.
    """
    if time.time() > _DEADLINE - 60:
        return x0
    
    def obj(x):
        return _neg_soft_ratio(x, beta=beta)
    
    try:
        res = minimize(obj, x0, method='Powell',
                       options={'maxiter': maxiter, 'xtol': 1e-6, 'ftol': 1e-6})
        return res.x.reshape(-1, 2)
    except Exception:
        return x0


def _refine_exact(x0, maxiter=100):
    """
    Refine using Powell's method with the exact objective.
    """
    if time.time() > _DEADLINE - 30:
        return x0
    
    def obj(x):
        return _neg_exact_ratio(x)
    
    try:
        res = minimize(obj, x0, method='Powell',
                       options={'maxiter': maxiter, 'xtol': 1e-8, 'ftol': 1e-8})
        return res.x.reshape(-1, 2)
    except Exception:
        return x0


def _refine_adaptive(x0):
    """
    Multi-stage refinement:
    1. Coarse with soft-min (low beta)
    2. Medium with soft-min (higher beta)
    3. Fine with exact objective
    """
    if time.time() > _DEADLINE - 90:
        return x0
    
    # Stage 1: Coarse soft-min
    refined1 = _refine_soft(x0, beta=10.0, maxiter=200)
    
    if time.time() > _DEADLINE - 60:
        return refined1
    
    # Stage 2: Medium soft-min
    refined2 = _refine_soft(refined1, beta=30.0, maxiter=200)
    
    if time.time() > _DEADLINE - 30:
        return refined2
    
    # Stage 3: Exact refinement
    refined3 = _refine_exact(refined2, maxiter=150)
    
    return refined3


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points on or inside a convex region to maximize
    the area of the smallest triangle formed by these points.
    
    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    best_points = None
    best_score = -np.inf
    
    rng = np.random.default_rng(seed=42)
    
    # Generate structured initial configurations
    initial_configs = _generate_structured_configs()
    
    # Add some random perturbations of the best structured configs
    for config in initial_configs[:4]:
        perturbed = config + rng.normal(0, 0.05, size=config.shape)
        initial_configs.append(perturbed)
    
    # Add purely random configs
    for _ in range(3):
        pts = rng.uniform(-1, 1, size=(n, 2))
        initial_configs.append(pts)
    
    # Process each initial configuration with adaptive refinement
    for config in initial_configs:
        if time.time() > _DEADLINE - 120:
            break
        
        # Adaptive multi-stage refinement
        refined = _refine_adaptive(config)
        
        # Evaluate
        score = _score(refined)
        if score > best_score:
            best_score = score
            best_points = refined.copy()
        
        # Also try direct exact refinement from original
        if time.time() > _DEADLINE - 60:
            break
        refined_direct = _refine_exact(config, maxiter=80)
        score_direct = _score(refined_direct)
        if score_direct > best_score:
            best_score = score_direct
            best_points = refined_direct.copy()
    
    # If we have time, do additional random restarts with refinement
    remaining_time = _DEADLINE - time.time()
    if remaining_time > 120:
        n_restarts = min(10, int(remaining_time // 15))
        for _ in range(n_restarts):
            if time.time() > _DEADLINE - 30:
                break
            
            # Random perturbation of best found
            if best_points is not None:
                start = best_points + rng.normal(0, 0.1, size=(n, 2))
            else:
                start = rng.uniform(-1, 1, size=(n, 2))
            
            refined = _refine_adaptive(start)
            score = _score(refined)
            if score > best_score:
                best_score = score
                best_points = refined.copy()
    
    # If no valid result found, fall back to regular polygon
    if best_points is None:
        best_points = _regular_polygon(13, radius=1.0)
    
    # Final verification and cleanup
    best_points = np.asarray(best_points, dtype=np.float64)
    if not np.all(np.isfinite(best_points)):
        best_points = rng.uniform(-1, 1, size=(n, 2))
    
    return best_points

# EVOLVE-BLOCK-END
