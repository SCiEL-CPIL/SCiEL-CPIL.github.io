"""Heilbronn triangles: place 11 points to maximize the minimum triangle area."""
import numpy as np
import time
import itertools
from scipy.optimize import minimize, differential_evolution

N = 11
S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4  # area of the equilateral triangle with side 1
DEADLINE = 145.0

# Precompute all triplets for N=11 points
_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])
_N_TRIP = len(_TRIPLETS)

def _cartesian_to_barycentric(pts):
    """Convert (x,y) points to (a,b,c) barycentric coords with a+b+c=1."""
    n = len(pts)
    abc = np.zeros((n, 3))
    c = pts[:, 1] / S3H
    b = pts[:, 0] - 0.5 * c
    a = 1.0 - b - c
    abc[:, 0] = a
    abc[:, 1] = b
    abc[:, 2] = c
    return abc

def _barycentric_to_cartesian(abc):
    """Convert (a,b,c) barycentric coords to (x,y) cartesian points."""
    n = len(abc)
    pts = np.zeros((n, 2))
    pts[:, 0] = abc[:, 1] + 0.5 * abc[:, 2]
    pts[:, 1] = abc[:, 2] * S3H
    return pts

def _all_triangle_areas(pts):
    """Compute all triangle areas (vectorized)."""
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    return areas

def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by three points."""
    return np.min(_all_triangle_areas(pts))

def _min_area_normalized(points):
    """Compute minimum triangle area normalized by the equilateral triangle area."""
    return _min_triangle_area(points) / TRI_AREA

def _soft_min_area(pts, beta):
    """Soft minimum of triangle areas using log-sum-exp (differentiable)."""
    areas = _all_triangle_areas(pts)
    max_a = np.max(areas)
    shifted = -beta * (areas - max_a)
    # Clip to avoid overflow
    shifted = np.clip(shifted, -700, 700)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min = max_a - log_sum / beta
    return soft_min

def _is_valid(points):
    """Check if all points are inside the triangle."""
    bary = _cartesian_to_barycentric(points)
    return np.all(bary >= -1e-9)

def _project_to_triangle(pts):
    """Project points into the equilateral triangle."""
    pts = pts.copy()
    for i in range(len(pts)):
        x, y = pts[i]
        y = max(0.0, min(S3H, y))
        x_min = y / S3
        x_max = 1 - y / S3
        x = max(x_min, min(x_max, x))
        pts[i] = [x, y]
    return pts

def _objective_bary_2d(flat_bc, beta=100.0):
    """
    Objective in 2D barycentric space (b, c) where a = 1-b-c.
    Uses soft-min for differentiability and penalty for constraint violation.
    """
    bc = flat_bc.reshape(-1, 2)
    b = bc[:, 0]
    c = bc[:, 1]
    a = 1.0 - b - c
    
    # Penalty for constraint violation (a,b,c >= 0)
    pen = 0.0
    if np.any(a < 0):
        pen += np.sum(a[a < 0] ** 2) * 500
    if np.any(b < 0):
        pen += np.sum(b[b < 0] ** 2) * 500
    if np.any(c < 0):
        pen += np.sum(c[c < 0] ** 2) * 500
    
    # Convert to Cartesian
    abc = np.column_stack([a, b, c])
    pts = _barycentric_to_cartesian(abc)
    
    # Soft min area
    ma = _soft_min_area(pts, beta)
    
    return -ma + pen

def _objective_bary_hard(flat_bc):
    """Hard min objective in 2D barycentric space."""
    bc = flat_bc.reshape(-1, 2)
    b = bc[:, 0]
    c = bc[:, 1]
    a = 1.0 - b - c
    
    pen = 0.0
    if np.any(a < 0):
        pen += np.sum(a[a < 0] ** 2) * 500
    if np.any(b < 0):
        pen += np.sum(b[b < 0] ** 2) * 500
    if np.any(c < 0):
        pen += np.sum(c[c < 0] ** 2) * 500
    
    abc = np.column_stack([a, b, c])
    pts = _barycentric_to_cartesian(abc)
    
    ma = _min_triangle_area(pts)
    
    return -ma + pen

def _random_point_in_triangle(rng):
    """Generate a random point in the triangle."""
    u = rng.random()
    v = rng.random()
    if u + v > 1:
        u = 1 - u
        v = 1 - v
    x = u + 0.5 * v
    y = S3H * v
    return np.array([x, y])

def _hexagonal_initialization(n_points, rng):
    """
    Generate a hexagonal grid of points inside the triangle.
    """
    points = []
    best_pts = None
    best_score = -1
    
    for scale in [0.1, 0.15, 0.2, 0.25, 0.3]:
        pts = []
        k = 0
        while True:
            y = k * np.sqrt(3)/2 * scale
            if y > S3H:
                break
            j = -10
            while j < 10:
                x = (j + (k % 2) * 0.5) * scale
                w = y / S3H
                u = 1.0 - x - 0.5 * w
                v = x - 0.5 * w
                if u >= -1e-9 and v >= -1e-9 and w >= -1e-9:
                    pts.append([x, y])
                j += 1
            k += 1
        
        pts = np.array(pts)
        if len(pts) == 0:
            continue
        
        if len(pts) >= n_points:
            indices = rng.choice(len(pts), n_points, replace=False)
            selected = pts[indices]
        else:
            selected = pts.copy()
            while len(selected) < n_points:
                p = _random_point_in_triangle(rng)
                selected = np.vstack([selected, [p]])
        
        score = _min_area_normalized(selected)
        if score > best_score:
            best_score = score
            best_pts = selected
    
    if best_pts is None:
        # Fallback to random
        return np.array([_random_point_in_triangle(rng) for _ in range(n_points)])
    
    return best_pts

def _farthest_point_sampling(rng, n_points):
    """
    Generate n_points in the triangle using Farthest Point Sampling (FPS).
    """
    points = [_random_point_in_triangle(rng)]
    
    for _ in range(n_points - 1):
        best_dist = -1
        best_pt = None
        
        for _ in range(100):
            cand = _random_point_in_triangle(rng)
            
            dists = np.linalg.norm(np.array(points) - cand, axis=1)
            min_d = np.min(dists)
            
            if min_d > best_dist:
                best_dist = min_d
                best_pt = cand
        
        points.append(best_pt)
        
    return np.array(points)

def _simulated_annealing(points, n_iter=3000, T_start=0.05, T_end=0.001, rng=None):
    """
    Run Simulated Annealing to maximize the minimum triangle area.
    """
    if rng is None:
        rng = np.random.RandomState(42)
    
    current_points = points.copy()
    current_score = _min_area_normalized(current_points)
    best_points = current_points.copy()
    best_score = current_score
    
    n = len(points)
    
    for i in range(n_iter):
        T = T_start * (T_end / T_start) ** (i / n_iter)
        
        idx = rng.randint(n)
        cand_points = current_points.copy()
        
        step = 0.05 * (1.0 + 0.5 * (T / T_start))
        dx = rng.normal(0, step)
        dy = rng.normal(0, step)
        
        cand_points[idx, 0] += dx
        cand_points[idx, 1] += dy
        
        # Project to triangle
        cand_points = _project_to_triangle(cand_points)
        
        cand_score = _min_area_normalized(cand_points)
        
        if cand_score >= current_score:
            current_points = cand_points
            current_score = cand_score
            if cand_score > best_score:
                best_score = cand_score
                best_points = cand_points.copy()
        else:
            delta = current_score - cand_score
            if delta > 0:
                prob = np.exp(-delta / T)
                if rng.random() < prob:
                    current_points = cand_points
                    current_score = cand_score
    
    return best_points, best_score

def construct():
    start_time = time.time()
    
    def elapsed():
        return time.time() - start_time
    
    best_points = None
    best_score = -float('inf')
    
    rng = np.random.RandomState(42)
    
    # Phase 1: Differential Evolution in barycentric space
    if elapsed() < 60:
        try:
            def _de_obj(flat_bc):
                return _objective_bary_2d(flat_bc, beta=100.0)
            
            bounds_de = [(0, 1), (0, 1)] * N
            de_result = differential_evolution(
                _de_obj,
                bounds_de,
                maxiter=20,
                popsize=15,
                seed=42,
                tol=1e-5,
                polish=False,
                init='latinhypercube',
                workers=1
            )
            
            if de_result.success or de_result.fun < 100:
                bc = de_result.x.reshape(N, 2)
                a = 1.0 - bc[:, 0] - bc[:, 1]
                # Check validity
                if np.all(a >= -1e-6) and np.all(bc[:, 0] >= -1e-6) and np.all(bc[:, 1] >= -1e-6):
                    abc = np.column_stack([a, bc[:, 0], bc[:, 1]])
                    # Clip to valid range
                    abc = np.maximum(abc, 0)
                    abc = abc / abc.sum(axis=1, keepdims=True)
                    pts = _barycentric_to_cartesian(abc)
                    score = _min_area_normalized(pts)
                    if score > best_score:
                        best_score = score
                        best_points = pts.copy()
        except Exception:
            pass
    
    # Phase 2: Simulated Annealing from diverse starts
    if elapsed() < 90:
        starts = []
        
        # FPS start
        fps_pts = _farthest_point_sampling(rng, N)
        starts.append(fps_pts)
        
        # Hexagonal start
        hex_pts = _hexagonal_initialization(N, rng)
        starts.append(hex_pts)
        
        # Random starts
        for _ in range(2):
            pts = np.array([_random_point_in_triangle(rng) for _ in range(N)])
            starts.append(pts)
        
        # Add DE result if found
        if best_points is not None:
            starts.append(best_points.copy())
        
        for start_pts in starts:
            if elapsed() > DEADLINE - 30:
                break
            
            try:
                sa_pts, sa_score = _simulated_annealing(start_pts, n_iter=3000, T_start=0.05, T_end=0.001, rng=rng)
                if sa_score > best_score:
                    best_score = sa_score
                    best_points = sa_pts.copy()
            except Exception:
                pass
    
    # Phase 3: Powell optimization from each initial guess
    if elapsed() < DEADLINE - 20:
        initial_guesses = []
        
        # FPS-based start
        fps_pts = _farthest_point_sampling(rng, N)
        bary = _cartesian_to_barycentric(fps_pts)
        initial_guesses.append(np.column_stack([bary[:, 1], bary[:, 2]]).flatten())
        
        # Random starts
        for _ in range(3):
            pts = np.array([_random_point_in_triangle(rng) for _ in range(N)])
            bary = _cartesian_to_barycentric(pts)
            initial_guesses.append(np.column_stack([bary[:, 1], bary[:, 2]]).flatten())
        
        # Add best result so far
        if best_points is not None:
            bary = _cartesian_to_barycentric(best_points)
            initial_guesses.append(np.column_stack([bary[:, 1], bary[:, 2]]).flatten())
        
        for x0 in initial_guesses:
            if elapsed() > DEADLINE - 15:
                break
            
            try:
                res = minimize(
                    _objective_bary_hard,
                    x0,
                    method='Powell',
                    options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                
                bc = res.x.reshape(N, 2)
                a = 1.0 - bc[:, 0] - bc[:, 1]
                
                # Check validity
                if np.all(a >= -1e-6) and np.all(bc[:, 0] >= -1e-6) and np.all(bc[:, 1] >= -1e-6):
                    abc = np.column_stack([a, bc[:, 0], bc[:, 1]])
                    abc = np.maximum(abc, 0)
                    abc = abc / abc.sum(axis=1, keepdims=True)
                    pts = _barycentric_to_cartesian(abc)
                    
                    if _is_valid(pts):
                        score = _min_area_normalized(pts)
                        if score > best_score:
                            best_score = score
                            best_points = pts.copy()
            except Exception:
                continue
    
    # Phase 4: Polish with L-BFGS-B using soft-min objective
    if best_points is not None and elapsed() < DEADLINE - 5:
        bary = _cartesian_to_barycentric(best_points)
        x0 = np.column_stack([bary[:, 1], bary[:, 2]]).flatten()
        
        try:
            res = minimize(
                lambda x: _objective_bary_2d(x, beta=200.0),
                x0,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            
            bc = res.x.reshape(N, 2)
            a = 1.0 - bc[:, 0] - bc[:, 1]
            
            if np.all(a >= -1e-6) and np.all(bc[:, 0] >= -1e-6) and np.all(bc[:, 1] >= -1e-6):
                abc = np.column_stack([a, bc[:, 0], bc[:, 1]])
                abc = np.maximum(abc, 0)
                abc = abc / abc.sum(axis=1, keepdims=True)
                pts = _barycentric_to_cartesian(abc)
                
                if _is_valid(pts):
                    score = _min_area_normalized(pts)
                    if score > best_score:
                        best_score = score
                        best_points = pts.copy()
        except Exception:
            pass
    
    # Phase 5: Nelder-Mead polish in Cartesian space
    if best_points is not None and elapsed() < DEADLINE - 2:
        def cart_objective(flat):
            pts = flat.reshape(N, 2)
            bary = _cartesian_to_barycentric(pts)
            penalty = 1000.0 * (np.sum(np.maximum(0, -bary[:, 0])**2) +
                               np.sum(np.maximum(0, -bary[:, 1])**2) +
                               np.sum(np.maximum(0, -bary[:, 2])**2))
            min_a = _min_area_normalized(pts)
            return -min_a + penalty
        
        try:
            res = minimize(
                cart_objective,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            polished = res.x.reshape(N, 2)
            if _is_valid(polished):
                score = _min_area_normalized(polished)
                if score > best_score:
                    best_score = score
                    best_points = polished.copy()
        except Exception:
            pass
    
    # Fallback
    if best_points is None:
        best_points = _farthest_point_sampling(rng, N)
    
    return best_points
