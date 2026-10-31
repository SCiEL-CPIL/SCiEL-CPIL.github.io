import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
import warnings
warnings.filterwarnings('ignore')

N = 13

# Precompute all triangle index triples
_TRIPLE_IDX = np.array([
    (i, j, k)
    for i in range(N)
    for j in range(i + 1, N)
    for k in range(j + 1, N)
], dtype=np.int32)

def _triangle_areas_vectorized(points):
    """Compute all triangle areas using precomputed indices."""
    pi = points[_TRIPLE_IDX[:, 0]]
    pj = points[_TRIPLE_IDX[:, 1]]
    pk = points[_TRIPLE_IDX[:, 2]]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def _hull_area(points):
    """Compute convex hull area."""
    try:
        if len(points) < 3:
            return 0.0
        if np.allclose(points, points[0]):
            return 0.0
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Compute the score: min triangle area / hull area."""
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    min_area = np.min(areas)
    if min_area < 1e-15:
        return 0.0
    return min_area / hull_area

def _normalize(points):
    """Center and normalize points to unit hull area."""
    centroid = np.mean(points, axis=0)
    pts = points - centroid
    hull_area = _hull_area(pts)
    if hull_area > 1e-12:
        pts = pts / np.sqrt(hull_area)
    return pts

def _objective(x_flat):
    """
    Objective function for optimization.
    1. Reshape to N x 2
    2. Center (remove translation)
    3. Rotate (remove rotation)
    4. Compute score and return negative
    """
    points = x_flat.reshape(N, 2)
    
    # Center
    centroid = np.mean(points, axis=0)
    points = points - centroid
    
    # Check if points are all the same (degenerate)
    if np.allclose(points, 0):
        return 1e10
    
    # Rotate such that point 0 is on the positive x-axis
    p0 = points[0]
    norm_p0 = np.linalg.norm(p0)
    if norm_p0 > 1e-9:
        theta = np.arctan2(p0[1], p0[0])
        c = np.cos(-theta)
        s = np.sin(-theta)
        R = np.array([[c, -s], [s, c]])
        points = points @ R.T
    else:
        p1 = points[1]
        norm_p1 = np.linalg.norm(p1)
        if norm_p1 > 1e-9:
            theta = np.arctan2(p1[1], p1[0])
            c = np.cos(-theta)
            s = np.sin(-theta)
            R = np.array([[c, -s], [s, c]])
            points = points @ R.T
        else:
            return 1e10

    score = _score(points)
    if score <= 0:
        return 1e10
    return -score

def _smooth_objective(x_flat, p=20):
    """
    Smooth surrogate objective using a power mean (soft-min) of triangle areas.
    """
    points = x_flat.reshape(N, 2)
    
    centroid = np.mean(points, axis=0)
    points = points - centroid
    
    if np.allclose(points, 0):
        return 1e10
    
    p0 = points[0]
    norm_p0 = np.linalg.norm(p0)
    if norm_p0 > 1e-9:
        theta = np.arctan2(p0[1], p0[0])
        c = np.cos(-theta)
        s = np.sin(-theta)
        R = np.array([[c, -s], [s, c]])
        points = points @ R.T
    else:
        p1 = points[1]
        norm_p1 = np.linalg.norm(p1)
        if norm_p1 > 1e-9:
            theta = np.arctan2(p1[1], p1[0])
            c = np.cos(-theta)
            s = np.sin(-theta)
            R = np.array([[c, -s], [s, c]])
            points = points @ R.T
        else:
            return 1e10

    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 1e10
        
    areas = _triangle_areas_vectorized(points)
    areas = np.maximum(areas, 1e-15)
    
    inv_areas = 1.0 / areas
    soft_min = (np.mean(inv_areas ** p)) ** (-1.0 / p)
    
    score = soft_min / hull_area
    if score <= 0:
        return 1e10
    return -score

def _fourier_to_points(params):
    """
    Convert Fourier parameters to 13 points.
    params: [a1, b1, a2, b2]
    """
    k = np.arange(N)
    theta = 2 * np.pi * k / N
    a1, b1, a2, b2 = params
    r = 1.0 + a1 * np.cos(theta) + b1 * np.sin(theta) + a2 * np.cos(2 * theta) + b2 * np.sin(2 * theta)
    r = np.maximum(r, 0.1)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack([x, y])

def _fourier_objective(params):
    """Objective for Fourier parameter optimization."""
    points = _fourier_to_points(params)
    if np.allclose(points, points[0]):
        return 1e10
    score = _score(points)
    if score <= 0:
        return 1e10
    return -score

def _get_grid_bases():
    """Generate various grid-based initial configurations."""
    bases = []
    
    # 1. 4x4 grid minus one corner
    xs = np.array([-1.5, -0.5, 0.5, 1.5])
    ys = np.array([-1.5, -0.5, 0.5, 1.5])
    gx, gy = np.meshgrid(xs, ys)
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    pts = pts[:-1] # Remove last point (top-right)
    pts = pts - np.mean(pts, axis=0)
    bases.append(pts)
    
    # 2. 3x5 grid
    xs = np.array([-1.0, 0.0, 1.0])
    ys = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    gx, gy = np.meshgrid(xs, ys)
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    pts = pts - np.mean(pts, axis=0)
    bases.append(pts)
    
    # 3. 5x3 grid
    xs = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    ys = np.array([-1.0, 0.0, 1.0])
    gx, gy = np.meshgrid(xs, ys)
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    pts = pts - np.mean(pts, axis=0)
    bases.append(pts)
    
    # 4. Hexagonal lattice subset
    # Generate a larger hex grid and pick 13 points
    hex_pts = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            # Axial to Cartesian
            x = i + j/2.0
            y = j * np.sqrt(3)/2.0
            hex_pts.append([x, y])
    hex_pts = np.array(hex_pts)
    # Sort by distance to center and take closest 13? Or just take a block.
    # Let's take a block: i in [-1, 1], j in [-1, 1] gives 9 points.
    # Need 4 more.
    # Let's just use the first 13 from a sorted list by radius
    dists = np.linalg.norm(hex_pts, axis=1)
    order = np.argsort(dists)
    pts = hex_pts[order[:13]]
    pts = pts - np.mean(pts, axis=0)
    bases.append(pts)

    # 5. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    bases.append(pts)
    
    # 6. Two rings (7 outer, 6 inner)
    r_out, r_in = 1.0, 0.5
    ang_out = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    ang_in = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi/6
    pts = np.vstack([
        np.column_stack([r_out * np.cos(ang_out), r_out * np.sin(ang_out)]),
        np.column_stack([r_in * np.cos(ang_in), r_in * np.sin(ang_in)])
    ])
    bases.append(pts)
    
    return bases

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_points = None
    best_score = 0.0
    
    def _update_best(points):
        nonlocal best_points, best_score
        if points is None:
            return
        if len(points) != N:
            return
        s = _score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()

    # 1. Generate initial candidates
    inits = []
    grid_bases = _get_grid_bases()
    
    for base in grid_bases:
        inits.append(base.flatten())
        
    # Add some random starts
    rng = np.random.default_rng(42)
    for _ in range(5):
        r = np.sqrt(rng.random(N))
        theta = rng.random(N) * 2 * np.pi
        pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        inits.append(pts.flatten())

    # 2. Optimization Loop
    # Strategy: For each initial point, do a quick smooth optimization, then a hard polish.
    # Also run a global DE search in Fourier space.
    
    # First, try to improve each grid base with local optimization
    for i, x0 in enumerate(inits):
        if time.time() > deadline - 20:
            break
            
        # Phase 1: Smooth optimization (Powell is good for non-smooth/steep landscapes)
        try:
            res = minimize(
                lambda x: _smooth_objective(x, p=20), 
                x0, 
                method='Powell',
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            opt_pts = res.x.reshape(N, 2)
            _update_best(opt_pts)
        except Exception:
            pass
            
        # Phase 2: Hard objective polish with Nelder-Mead
        if best_score > 0 and time.time() < deadline - 5:
            try:
                start_x = res.x if 'res' in locals() and res.success else x0
                res_nm = minimize(
                    _objective,
                    start_x,
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
                )
                opt_pts_nm = res_nm.x.reshape(N, 2)
                _update_best(opt_pts_nm)
            except Exception:
                pass

    # 3. Fourier Series Optimization (Global Search in Low-Dim Space)
    # This captures the "structural" insight from the inspiration that low-dimensional
    # parameterization might find better basins.
    if time.time() < deadline - 15:
        try:
            bounds = [(-0.4, 0.4), (-0.4, 0.4), (-0.4, 0.4), (-0.4, 0.4)]
            
            de_res = differential_evolution(
                _fourier_objective,
                bounds,
                seed=42,
                maxiter=50,
                tol=1e-6,
                popsize=15,
                polish=False # We will polish manually to save time
            )
            
            if de_res.fun < 1e9:
                fourier_points = _fourier_to_points(de_res.x)
                _update_best(fourier_points)
                
                # Polish the Fourier result with full Cartesian optimization
                if best_points is not None and time.time() < deadline - 5:
                    res_polish = minimize(
                        _objective,
                        fourier_points.flatten(),
                        method='Nelder-Mead',
                        options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
                    )
                    opt_pts_polish = res_polish.x.reshape(N, 2)
                    _update_best(opt_pts_polish)
                    
        except Exception:
            pass

    # 4. Final High-Precision Polish on the best candidate found
    if best_points is not None and time.time() < deadline - 2:
        try:
            res_final = minimize(
                _objective,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            opt_pts_final = res_final.x.reshape(N, 2)
            _update_best(opt_pts_final)
        except Exception:
            pass

    if best_points is None:
        # Fallback to regular polygon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return best_points
