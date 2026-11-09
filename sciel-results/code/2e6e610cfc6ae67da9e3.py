# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time
import itertools

def compute_score(points):
    """Compute the Heilbronn score: A_min / A_hull (vectorized)"""
    n = len(points)
    if n < 3:
        return 0.0
    
    # Check for collinearity and valid hull
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 0.0
    
    if A_hull <= 0:
        return 0.0
    
    # Vectorized triangle area computation
    # For all triples (i,j,k), compute area using cross product
    idx = np.array(list(itertools.combinations(range(n), 3)))
    if len(idx) == 0:
        return 0.0
    
    p1 = points[idx[:, 0]]  # shape (m, 2)
    p2 = points[idx[:, 1]]
    p3 = points[idx[:, 2]]
    
    # Area = 0.5 * |cross(p2-p1, p3-p1)|
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = np.abs(cross) / 2.0
    
    min_area = np.min(areas)
    
    return min_area / A_hull

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points in a convex region to maximize the area of the
    smallest triangle formed by these points.
    
    Uses a multi-start Nelder-Mead optimization with structured initial configurations.
    
    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    safe_deadline = time.time() + 570.0
    
    def nelder_mead_optimize(points, max_iter=2000, tol=1e-8):
        """Nelder-Mead optimization for the Heilbronn score"""
        n = len(points)
        dim = n * 2
        
        def flatten(p):
            return p.flatten()
        
        def unflatten(v):
            return v.reshape(n, 2)
        
        def objective(v):
            p = unflatten(v)
            return compute_score(p)
        
        # Initialize simplex
        simplex = [flatten(points.copy())]
        initial_step = 0.1
        for i in range(dim):
            v = flatten(points.copy())
            v[i] += initial_step
            simplex.append(v)
        
        # Nelder-Mead parameters
        alpha = 0.5  # reflection
        gamma = 2.0  # expansion
        rho = 0.5    # contraction
        sigma = 0.5  # shrink
        
        best_points = points.copy()
        best_score = objective(flatten(points))
        
        for iteration in range(max_iter):
            if time.time() > safe_deadline:
                break
            
            # Evaluate all simplex vertices
            fvals = [objective(v) for v in simplex]
            
            # Sort simplex by function value (minimize negative score)
            order = np.argsort([-f for f in fvals])
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]
            
            # Check convergence
            if abs(fvals[-1] - fvals[0]) < tol:
                break
            
            # Centroid of all but worst point
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            fr = objective(xr)
            
            if fr < fvals[0]:
                # Expansion
                xe = centroid + gamma * (centroid - simplex[-1])
                fe = objective(xe)
                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
            elif fr < fvals[-2]:
                # Accept reflection
                simplex[-1] = xr
                fvals[-1] = fr
            else:
                # Contraction
                if fr < fvals[-1]:
                    # Outside contraction
                    xc = centroid + rho * (simplex[-1] - centroid)
                    fc = objective(xc)
                else:
                    # Inside contraction
                    xc = centroid + rho * (centroid - simplex[-1])
                    fc = objective(xc)
                
                if fc < fvals[-1]:
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fvals[i] = objective(simplex[i])
            
            # Track best
            for v, f in zip(simplex, fvals):
                if f > best_score:
                    best_score = f
                    best_points = unflatten(v).copy()
        
        return best_points, best_score
    
    # Strategy 1: Regular dodecagon + center
    def make_regular_polygon_with_center(n_outer=12, n_inner=1):
        angles = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
        radius = 1.0
        outer_points = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
        center = np.zeros((n_inner, 2))
        points = np.vstack([outer_points, center])
        return points
    
    # Strategy 2: Two concentric regular polygons
    def make_concentric_polygons(n1=7, n2=6, r1=1.0, r2=0.5):
        angles1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
        angles2 = np.linspace(0, 2 * np.pi, n2, endpoint=False) + np.pi / n2
        p1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
        p2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
        return np.vstack([p1, p2])
    
    # Strategy 3: Structured random with polygon bias
    def make_structured_random(seed):
        rng = np.random.default_rng(seed)
        base = make_regular_polygon_with_center(12, 1)
        perturb = rng.normal(0, 0.05, size=base.shape)
        return base + perturb
    
    # Strategy 4: Grid-like arrangement
    def make_grid_arrangement():
        # 4x3 grid with one extra point
        xs = np.array([0, 1, 2, 3])
        ys = np.array([0, 1, 2, 3])
        # Take 13 points from a 4x4 grid
        points = []
        for i in range(4):
            for j in range(4):
                if len(points) < 13:
                    points.append([i, j])
        points = np.array(points)
        # Center and scale
        points = (points - points.mean(axis=0)) / points.std(axis=0)
        return points
    
    # Strategy 5: Fibonacci sphere-like distribution projected to 2D
    def make_fibonacci_2d():
        n = 13
        golden_ratio = (1 + np.sqrt(5)) / 2
        angles = 2 * np.pi * np.arange(n) / golden_ratio
        radii = np.sqrt(np.arange(n) / n)
        points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        return points
    
    # Initial configurations
    initial_configs = [
        make_regular_polygon_with_center(12, 1),
        make_concentric_polygons(7, 6, 1.0, 0.5),
        make_concentric_polygons(6, 7, 1.0, 0.6),
        make_structured_random(12345),
        make_grid_arrangement(),
        make_fibonacci_2d(),
    ]
    
    # Add some random starts
    rng = np.random.default_rng(seed=42)
    for _ in range(3):
        initial_configs.append(rng.random((13, 2)) * 2 - 1)
    
    best_points = None
    best_score = 0.0
    
    for config in initial_configs:
        if time.time() > safe_deadline:
            break
        
        points, score = nelder_mead_optimize(config, max_iter=1500)
        if score > best_score:
            best_score = score
            best_points = points.copy()
    
    # Additional random restarts if time permits
    restart_count = 0
    while time.time() < safe_deadline and restart_count < 20:
        rng = np.random.default_rng(seed=restart_count * 7 + 100)
        config = rng.random((13, 2)) * 2 - 1
        points, score = nelder_mead_optimize(config, max_iter=1000)
        if score > best_score:
            best_score = score
            best_points = points.copy()
        restart_count += 1
    
    # Final fallback
    if best_points is None:
        best_points = make_regular_polygon_with_center(12, 1)
    
    # Ensure all points are finite
    if not np.all(np.isfinite(best_points)):
        best_points = make_regular_polygon_with_center(12, 1)
    
    return best_points
    
    # Strategy 1: Start with a regular dodecagon (12-gon) plus center
    def make_regular_polygon_with_center(n_outer=12, n_inner=1):
        """Create regular polygon with center point(s)"""
        angles = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
        radius = 1.0
        outer_points = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
        center = np.zeros((n_inner, 2))
        points = np.vstack([outer_points, center])
        return points
    
    # Strategy 2: Two concentric regular polygons
    def make_concentric_polygons(n1=7, n2=6, r1=1.0, r2=0.5):
        angles1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
        angles2 = np.linspace(0, 2 * np.pi, n2, endpoint=False) + np.pi / n2  # offset
        p1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
        p2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
        return np.vstack([p1, p2])
    
    # Strategy 3: Random initialization with structured bias
    def make_bstructured_random():
        rng = np.random.default_rng(seed=999)
        # Mix of polygonal and random
        base = make_regular_polygon_with_center(12, 1)
        # Add small random perturbations
        perturb = rng.normal(0, 0.05, size=base.shape)
        return base + perturb
    
    # Initialize with multiple starting configurations
    initial_configs = [
        make_regular_polygon_with_center(12, 1),
        make_concentric_polygons(7, 6, 1.0, 0.5),
        make_concentric_polygons(6, 7, 1.0, 0.6),
        make_bstructured_random(),
    ]
    
    # Also try a fully random start
    rng = np.random.default_rng(seed=42)
    initial_configs.append(rng.random((13, 2)) * 2 - 1)
    
    best_points = None
    best_score = 0.0
    
    for config in initial_configs:
        if time.time() > safe_deadline:
            break
        
        points, score = try_configuration(config)
        if score > best_score:
            best_score = score
            best_points = points.copy()
    
    # If we haven't found a good solution, try more random restarts
    if best_score < 0.01 and time.time() < safe_deadline:
        for restart in range(20):
            if time.time() > safe_deadline:
                break
            
            rng = np.random.default_rng(seed=restart * 7 + 1)
            config = rng.random((13, 2)) * 2 - 1
            points, score = try_configuration(config)
            if score > best_score:
                best_score = score
                best_points = points.copy()
    
    # Final fallback: if best_points is None, use a basic configuration
    if best_points is None:
        best_points = make_regular_polygon_with_center(12, 1)
    
    # Ensure all points are finite
    if not np.all(np.isfinite(best_points)):
        best_points = make_regular_polygon_with_center(12, 1)
    
    return best_points

# EVOLVE-BLOCK-END
