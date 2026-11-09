import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 14

# Precompute triple indices for vectorized area calculation
_i_list, _j_list, _k_list = [], [], []
for _i in range(N):
    for _j in range(_i + 1, N):
        for _k in range(_j + 1, N):
            _i_list.append(_i)
            _j_list.append(_j)
            _k_list.append(_k)
_TRIPLE_I = np.array(_i_list)
_TRIPLE_J = np.array(_j_list)
_TRIPLE_K = np.array(_k_list)
_N_TRIPLES = len(_i_list)

def _compute_areas(points):
    """Vectorized triangle areas for all triples."""
    p1 = points[_TRIPLE_I]
    p2 = points[_TRIPLE_J]
    p3 = points[_TRIPLE_K]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def _hull_area(points):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def get_score(points):
    """Score = min triangle area / convex hull area."""
    ha = _hull_area(points)
    if ha <= 1e-12:
        return 0.0
    areas = _compute_areas(points)
    ma = np.min(areas)
    if ma < 1e-15:
        return 0.0
    return ma / ha

def objective(x_flat):
    """Objective function for minimization (negative score)."""
    points = x_flat.reshape(N, 2)
    # Add a small penalty if points are too close to avoid numerical issues in hull
    # Though ConvexHull handles it, score becomes 0 if hull area is 0
    score = get_score(points)
    return -score

def _regular_polygon(radius=1.0):
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    return np.column_stack((x, y)).flatten()

def _perturbed_polygon(radius=1.0, noise=0.1, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    angles = np.linspace(0, 2*np.pi, N, endpoint=False) + rng.normal(0, noise, N)
    angles = np.sort(angles)
    radii = radius + rng.normal(0, noise*0.5, N)
    radii = np.clip(radii, 0.5, 1.5)
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.column_stack((x, y)).flatten()

def _concentric_rings(n_inner, n_outer, r_inner=0.5, r_outer=1.0, rng=None):
    """Generate points on two concentric rings."""
    if rng is None:
        rng = np.random.default_rng()
    
    # Inner ring
    angles_in = np.linspace(0, 2*np.pi, n_inner, endpoint=False) + rng.normal(0, 0.1, n_inner)
    x_in = r_inner * np.cos(angles_in)
    y_in = r_inner * np.sin(angles_in)
    
    # Outer ring
    angles_out = np.linspace(0, 2*np.pi, n_outer, endpoint=False) + rng.normal(0, 0.1, n_outer)
    x_out = r_outer * np.cos(angles_out)
    y_out = r_outer * np.sin(angles_out)
    
    x = np.concatenate([x_in, x_out])
    y = np.concatenate([y_in, y_out])
    return np.column_stack((x, y)).flatten()

def _three_rings(n1, n2, n3, r1=0.3, r2=0.7, r3=1.0, rng=None):
    """Generate points on three concentric rings."""
    if rng is None:
        rng = np.random.default_rng()
    
    x_list = []
    y_list = []
    for n_pts, r in zip([n1, n2, n3], [r1, r2, r3]):
        angles = np.linspace(0, 2*np.pi, n_pts, endpoint=False) + rng.normal(0, 0.1, n_pts)
        x_list.append(r * np.cos(angles))
        y_list.append(r * np.sin(angles))
        
    x = np.concatenate(x_list)
    y = np.concatenate(y_list)
    return np.column_stack((x, y)).flatten()

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    rng = np.random.default_rng(42)
    
    best_points = None
    best_score = 0.0
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Regular polygon
    initializations.append(_regular_polygon(1.0))
    
    # 2. Perturbed regular polygons
    for noise in [0.05, 0.1, 0.2]:
        initializations.append(_perturbed_polygon(1.0, noise, rng))
        
    # 3. Concentric rings (2 rings)
    # Split 14 points into various combinations
    for n_in in [4, 5, 6, 7]:
        n_out = N - n_in
        initializations.append(_concentric_rings(n_in, n_out, 0.5, 1.0, rng))
        initializations.append(_concentric_rings(n_in, n_out, 0.6, 1.0, rng))
        
    # 4. Three rings
    initializations.append(_three_rings(4, 5, 5, 0.3, 0.7, 1.0, rng))
    initializations.append(_three_rings(5, 5, 4, 0.3, 0.7, 1.0, rng))
    
    # 5. Random points in unit disk
    for _ in range(10):
        angles = rng.uniform(0, 2*np.pi, N)
        radii = rng.uniform(0.5, 1.0, N)
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        initializations.append(np.column_stack((x, y)).flatten())

    # Optimization loop
    # We use Powell's method which is derivative-free and robust for non-smooth objectives
    # Limit iterations to prevent long runs on a single start point
    options = {'maxiter': 500, 'xtol': 1e-8, 'ftol': 1e-8}
    
    for idx, init_flat in enumerate(initializations):
        if time.time() > deadline - 5.0:
            break
            
        # Check time for this optimization
        if time.time() > deadline - 2.0:
            break
            
        try:
            # Run Powell optimization
            result = minimize(objective, init_flat, method='Powell', options=options)
            
            if result.success or result.fun < 0:
                points = result.x.reshape(N, 2)
                score = -result.fun
                if score > best_score:
                    best_score = score
                    best_points = points.copy()
        except Exception:
            pass

    # If no valid points found, fallback to regular polygon
    if best_points is None:
        best_points = _regular_polygon(1.0).reshape(N, 2)

    return best_points
