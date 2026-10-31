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

def _support_lines_to_polygon(n_points):
    """
    Generate n_points as vertices of a convex polygon defined by support lines.
    Each support line has a fixed normal angle (evenly spaced) and an offset.
    The polygon is the intersection of half-planes.
    """
    # Evenly spaced normal angles
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    return normals, angles

def _polygon_from_support_lines(offsets, normals=None):
    """
    Given offsets for support lines with fixed normal angles, compute the polygon vertices.
    offsets: array of shape (n_points,) - distance from origin along each normal
    normals: array of shape (n_points, 2) - fixed normal directions
    Returns: array of shape (n_points, 2) - polygon vertices in order
    """
    n = len(offsets)
    if normals is None:
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # For each pair of adjacent support lines, find their intersection
    vertices = np.zeros((n, 2))
    for i in range(n):
        n1 = normals[i]
        n2 = normals[(i + 1) % n]
        o1 = offsets[i]
        o2 = offsets[(i + 1) % n]
        
        # Solve: n1 . x = o1, n2 . x = o2
        A = np.array([n1, n2])
        b = np.array([o1, o2])
        try:
            if abs(np.linalg.det(A)) < 1e-12:
                vertices[i] = np.array([o1 * n1[0], o1 * n1[1]])
            else:
                vertices[i] = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            vertices[i] = np.array([o1 * n1[0], o1 * n1[1]])
    
    return vertices

def _support_line_objective(offsets):
    """
    Objective: maximize min triangle area / hull area for polygon from support lines.
    Returns negative score for minimization.
    """
    points = _polygon_from_support_lines(offsets)
    
    # Check if polygon is valid (convex, non-degenerate)
    if np.any(~np.isfinite(points)):
        return 1e10
    
    # Center the polygon
    centroid = np.mean(points, axis=0)
    points = points - centroid
    
    # Check degeneracy
    if np.allclose(points, 0):
        return 1e10
    
    score = _score(points)
    if score <= 0:
        return 1e10
    return -score

def _simulated_annealing_support_lines(n_points, max_iters, start_temp, end_temp, 
                                         cooling_rate, seed=42, deadline=None):
    """
    Simulated annealing to optimize support line offsets.
    """
    rng = np.random.default_rng(seed)
    
    # Initialize offsets: start with regular polygon
    initial_offsets = np.ones(n_points)
    
    current_offsets = initial_offsets.copy()
    current_points = _polygon_from_support_lines(current_offsets)
    current_centroid = np.mean(current_points, axis=0)
    current_points_c = current_points - current_centroid
    current_score = _score(current_points_c)
    current_obj = -current_score if current_score > 0 else 1e10
    
    best_offsets = current_offsets.copy()
    best_points = current_points_c.copy()
    best_score = current_score
    best_obj = current_obj
    
    n_evals = 0
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Generate neighbor: perturb one or a few offsets
        n_perturb = rng.integers(1, min(4, n_points))
        idx = rng.choice(n_points, size=n_perturb, replace=False)
        perturbations = rng.normal(0, 0.1, size=n_perturb)
        
        neighbor_offsets = current_offsets.copy()
        neighbor_offsets[idx] += perturbations
        
        # Ensure offsets are positive (for valid convex polygon)
        neighbor_offsets = np.maximum(neighbor_offsets, 0.1)
        
        neighbor_points = _polygon_from_support_lines(neighbor_offsets)
        
        if np.any(~np.isfinite(neighbor_points)):
            continue
        
        neighbor_centroid = np.mean(neighbor_points, axis=0)
        neighbor_points_c = neighbor_points - neighbor_centroid
        
        neighbor_score = _score(neighbor_points_c)
        neighbor_obj = -neighbor_score if neighbor_score > 0 else 1e10
        
        n_evals += 1
        
        # Accept or reject
        if neighbor_obj < current_obj:
            accept = True
        else:
            delta = neighbor_obj - current_obj
            temp = start_temp * (cooling_rate ** it)
            if temp > 1e-10:
                accept = rng.random() < np.exp(-delta / temp)
            else:
                accept = False
        
        if accept:
            current_offsets = neighbor_offsets
            current_points_c = neighbor_points_c
            current_score = neighbor_score
            current_obj = neighbor_obj
            
            if current_score > best_score:
                best_score = current_score
                best_obj = current_obj
                best_offsets = current_offsets.copy()
                best_points = current_points_c.copy()
    
    return best_points, best_score, n_evals

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
    hex_pts = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            x = i + j/2.0
            y = j * np.sqrt(3)/2.0
            hex_pts.append([x, y])
    hex_pts = np.array(hex_pts)
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

def _smooth_objective(x_flat, p=20):
    """
    Smooth surrogate objective using a power mean (soft-min) of triangle areas.
    """
    points = x_flat.reshape(N, 2)
    
    centroid = np.mean(points, axis=0)
    points = points - centroid
    
    if np.allclose(points, 0):
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

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
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

    # 2. Quick local optimization on grid bases
    for i, x0 in enumerate(inits):
        if time.time() > deadline - 40:
            break
            
        # Phase 1: Smooth optimization with Powell
        try:
            res = minimize(
                lambda x: _smooth_objective(x, p=20), 
                x0, 
                method='Powell',
                options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            opt_pts = res.x.reshape(N, 2)
            _update_best(opt_pts)
        except Exception:
            pass
            
        # Phase 2: Hard objective polish with Nelder-Mead
        if best_score > 0 and time.time() < deadline - 30:
            try:
                start_x = res.x if 'res' in locals() and res.success else x0
                res_nm = minimize(
                    _objective,
                    start_x,
                    method='Nelder-Mead',
                    options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
                )
                opt_pts_nm = res_nm.x.reshape(N, 2)
                _update_best(opt_pts_nm)
            except Exception:
                pass

    # 3. Simulated Annealing in support-line space (STRUCTURAL CHANGE)
    # This is a different representation: 13 offsets for 13 support lines
    # with fixed normal angles, forming a convex polygon.
    if time.time() < deadline - 20:
        # Multiple SA runs with different seeds and temperature schedules
        sa_configs = [
            {'max_iters': 5000, 'start_temp': 1.0, 'cooling_rate': 0.999, 'seed': 1},
            {'max_iters': 5000, 'start_temp': 0.5, 'cooling_rate': 0.998, 'seed': 2},
            {'max_iters': 3000, 'start_temp': 2.0, 'cooling_rate': 0.9995, 'seed': 3},
        ]
        
        for cfg in sa_configs:
            if time.time() > deadline - 15:
                break
            try:
                sa_points, sa_score, n_evals = _simulated_annealing_support_lines(
                    N, 
                    cfg['max_iters'], 
                    cfg['start_temp'], 
                    0.01, 
                    cfg['cooling_rate'], 
                    seed=cfg['seed'],
                    deadline=deadline - 10
                )
                _update_best(sa_points)
                
                # Polish the SA result with Cartesian optimization
                if best_points is not None and time.time() < deadline - 5:
                    res_polish = minimize(
                        _objective,
                        best_points.flatten(),
                        method='Nelder-Mead',
                        options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
                    )
                    opt_pts_polish = res_polish.x.reshape(N, 2)
                    _update_best(opt_pts_polish)
                    
            except Exception:
                pass

    # 4. Final High-Precision Polish on the best candidate found
    if best_points is not None and time.time() < deadline - 3:
        try:
            res_final = minimize(
                _objective,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 1500, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
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
