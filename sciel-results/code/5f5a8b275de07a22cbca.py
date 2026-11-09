# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
import itertools
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = _START_TIME + 580.0  # 20s safety margin

def _min_triangle_area(points):
    """
    Compute the minimum area among all triangles formed by triples of points.
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    # Generate all combinations of 3 indices
    indices = np.array(list(itertools.combinations(range(n), 3)))
    
    # Extract points
    p1 = points[indices[:, 0]]
    p2 = points[indices[:, 1]]
    p3 = points[indices[:, 2]]
    
    # Calculate areas: 0.5 * | (p2-p1) x (p3-p1) |
    v1 = p2 - p1
    v2 = p3 - p1
    # Cross product in 2D is a scalar: v1_x * v2_y - v1_y * v2_x
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _hull_area(points):
    """
    Compute the area of the convex hull of the points.
    """
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """
    Calculate the metric: A_min / A_hull.
    """
    a_hull = _hull_area(points)
    if a_hull <= 1e-12:
        return 0.0
    a_min = _min_triangle_area(points)
    return a_min / a_hull

def _objective(points_flat):
    """
    Objective function for minimization (we minimize negative score).
    """
    points = points_flat.reshape(13, 2)
    s = _score(points)
    # If score is 0, it's invalid (collinear or degenerate), penalize heavily
    if s <= 1e-12:
        return 1e6
    return -s

def _generate_regular_polygon(n=13, radius=1.0, seed=0):
    """
    Generate points on a regular n-gon.
    """
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    points = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
    # Add small noise to break symmetry
    rng = np.random.default_rng(seed)
    points += rng.normal(0, 0.01, (n, 2))
    return points

def _generate_hex_lattice(n=13, seed=0):
    """
    Generate points from a hexagonal (triangular) lattice.
    """
    rng = np.random.default_rng(seed)
    # Create a larger grid of hex lattice points
    # Basis vectors for hex lattice
    a1 = np.array([1.0, 0.0])
    a2 = np.array([0.5, np.sqrt(3)/2])
    
    points = []
    # Iterate over a range of lattice coordinates
    for i in range(-3, 4):
        for j in range(-3, 4):
            p = i * a1 + j * a2
            points.append(p)
    points = np.array(points)
    
    # Shuffle and take n points
    idx = rng.permutation(len(points))
    selected = points[idx[:n]].copy()
    
    # Add small noise
    selected += rng.normal(0, 0.01, (n, 2))
    
    # Center and scale to unit circle roughly
    center = selected.mean(axis=0)
    selected = selected - center
    max_r = np.max(np.linalg.norm(selected, axis=1))
    if max_r > 0:
        selected = selected / max_r * 0.8
    return selected

def _generate_center_rings(n=13, seed=0):
    """
    Generate 1 center point, 6 in first ring, 6 in second ring.
    """
    rng = np.random.default_rng(seed)
    points = []
    # Center
    points.append([0.0, 0.0])
    # Ring 1: 6 points
    angles1 = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    r1 = 0.4
    for a in angles1:
        points.append([r1 * np.cos(a), r1 * np.sin(a)])
    # Ring 2: 6 points (offset by 30 degrees)
    angles2 = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi/6
    r2 = 0.8
    for a in angles2:
        points.append([r2 * np.cos(a), r2 * np.sin(a)])
    
    points = np.array(points)
    points += rng.normal(0, 0.01, (n, 2))
    return points

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 13.

    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    best_score = -1.0
    best_points = np.random.default_rng(0).random((n, 2))
    
    # List of initial generation strategies
    generators = [
        _generate_regular_polygon,
        _generate_hex_lattice,
        _generate_center_rings
    ]
    
    current_seed = 0
    max_restarts = 500 # High cap, limited by time
    
    while time.time() < _DEADLINE and current_seed < max_restarts:
        # Check deadline
        if time.time() > _DEADLINE - 5:
            break
            
        # Cycle through generators
        gen_idx = current_seed % len(generators)
        gen_func = generators[gen_idx]
        
        # Use a unique seed for each restart to ensure diversity
        restart_seed = current_seed * 12345 + 67890
        
        try:
            initial_points = gen_func(n=n, seed=restart_seed)
            initial_flat = initial_points.flatten()
            
            # Run Powell optimization
            res = minimize(
                _objective, 
                initial_flat, 
                method='Powell', 
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'disp': False}
            )
        except Exception:
            current_seed += 1
            continue
        
        res_points = res.x.reshape(n, 2)
        
        # Validate result
        if not np.all(np.isfinite(res_points)):
            current_seed += 1
            continue
            
        res_score = _score(res_points)
        
        if res_score > best_score:
            best_score = res_score
            best_points = res_points.copy()
            
        current_seed += 1
        
    return best_points

# EVOLVE-BLOCK-END
