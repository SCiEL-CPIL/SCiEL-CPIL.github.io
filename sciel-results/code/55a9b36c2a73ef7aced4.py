"""Convex polygon optimization via support function Fourier coefficients."""
N = 14
M = N // 2  # 7 unique points (not used in this new representation, but kept for signature if needed)

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

# Precompute indices for all combinations of 3 points for vectorized area computation
_indices = np.arange(N)
IDX_I, IDX_J, IDX_K = np.meshgrid(_indices, _indices, _indices, indexing='ij')
_mask = (IDX_I < IDX_J) & (IDX_J < IDX_K)
IDX_I = IDX_I[_mask].flatten()
IDX_J = IDX_J[_mask].flatten()
IDX_K = IDX_K[_mask].flatten()

def _all_areas(points):
    """Vectorized computation of all triangle areas."""
    pi = points[IDX_I]
    pj = points[IDX_J]
    pk = points[IDX_K]
    v1x = pj[:, 0] - pi[:, 0]
    v1y = pj[:, 1] - pi[:, 1]
    v2x = pk[:, 0] - pi[:, 0]
    v2y = pk[:, 1] - pi[:, 1]
    areas = 0.5 * np.abs(v1x * v2y - v1y * v2x)
    return areas

def _compute_hull_area(points):
    """Compute convex hull area."""
    try:
        if np.allclose(points[:, 0], points[0, 0]) or np.allclose(points[:, 1], points[0, 1]):
            return 0.0
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Min triangle area / convex hull area."""
    hull_area = _compute_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = np.min(_all_areas(points))
    return min_area / hull_area

def _support_function(theta, coeffs):
    """
    Evaluate support function h(theta) = c0 + sum(c_{2k-1} cos(k theta) + c_{2k} sin(k theta))
    coeffs = [c0, c1, c2, c3, c4, c5, c6]
    """
    h = np.full_like(theta, coeffs[0])
    if len(coeffs) > 1:
        h += coeffs[1] * np.cos(theta) + coeffs[2] * np.sin(theta)
    if len(coeffs) > 3:
        h += coeffs[3] * np.cos(2 * theta) + coeffs[4] * np.sin(2 * theta)
    if len(coeffs) > 5:
        h += coeffs[5] * np.cos(3 * theta) + coeffs[6] * np.sin(3 * theta)
    return h

def _support_function_2nd_deriv(theta, coeffs):
    """
    Evaluate h''(theta)
    h' = -c1 sin + c2 cos - 2c3 sin2 + 2c4 cos2 - 3c5 sin3 + 3c6 cos3
    h'' = -c1 cos - c2 sin - 4c3 cos2 - 4c4 sin2 - 9c5 cos3 - 9c6 sin3
    """
    h2 = np.zeros_like(theta)
    if len(coeffs) > 1:
        h2 += -coeffs[1] * np.cos(theta) - coeffs[2] * np.sin(theta)
    if len(coeffs) > 3:
        h2 += -4 * coeffs[3] * np.cos(2 * theta) - 4 * coeffs[4] * np.sin(2 * theta)
    if len(coeffs) > 5:
        h2 += -9 * coeffs[5] * np.cos(3 * theta) - 9 * coeffs[6] * np.sin(3 * theta)
    return h2

def _is_valid_support(coeffs):
    """Check if support function defines a strictly convex body."""
    if np.any(~np.isfinite(coeffs)):
        return False
    # Sample theta densely to check h > 0 and h + h'' > 0
    theta_check = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    h = _support_function(theta_check, coeffs)
    h2 = _support_function_2nd_deriv(theta_check, coeffs)
    if np.min(h) < 0.01:
        return False
    if np.min(h + h2) < 0.01:
        return False
    return True

def _vertices_from_support(coeffs):
    """
    Compute 14 vertices of the convex polygon defined by support function h(theta)
    at angles theta_i = 2*pi*i/N.
    Vertex i is intersection of lines L_{i-1} and L_i.
    Line L_i: x cos(theta_i) + y sin(theta_i) = h(theta_i)
    """
    theta_vals = np.linspace(0, 2 * np.pi, N, endpoint=False)
    h_vals = _support_function(theta_vals, coeffs)
    
    # Cos and sin of theta
    c = np.cos(theta_vals)
    s = np.sin(theta_vals)
    
    vertices = np.zeros((N, 2))
    for i in range(N):
        i_prev = (i - 1) % N
        # Line i_prev: x c_p + y s_p = h_p
        # Line i: x c_i + y s_i = h_i
        # Solve:
        # | c_p s_p | |x|   |h_p|
        # | c_i s_i | |y| = |h_i|
        det = c[i_prev] * s[i] - c[i] * s[i_prev]
        if np.abs(det) < 1e-10:
            # Parallel lines, degenerate
            return None
        x = (h_vals[i_prev] * s[i] - h_vals[i] * s[i_prev]) / det
        y = (c[i_prev] * h_vals[i] - c[i] * h_vals[i_prev]) / det
        vertices[i] = [x, y]
    return vertices

def _objective_support(coeffs):
    """Objective to minimize: -score"""
    if not _is_valid_support(coeffs):
        return 1e10
    pts = _vertices_from_support(coeffs)
    if pts is None:
        return 1e10
    # Check if points are in convex position and not degenerate
    if np.any(~np.isfinite(pts)):
        return 1e10
    s = _score(pts)
    if s < 1e-12:
        return 1e10
    return -s

def _objective_support_soft(coeffs, beta=50.0):
    """Soft-min objective for support function parameterization."""
    if not _is_valid_support(coeffs):
        return 1e10
    pts = _vertices_from_support(coeffs)
    if pts is None:
        return 1e10
    areas = _all_areas(pts)
    if np.min(areas) < 1e-12:
        return 1e10
    z = -beta * areas
    max_z = np.max(z)
    if max_z < -500:
        log_sum = max_z
    else:
        log_sum = max_z + np.log(np.sum(np.exp(z - max_z)))
    soft_min = -log_sum / beta
    if soft_min < 1e-12:
        return 1e10
    hull_area = _compute_hull_area(pts)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / soft_min

def _refine_support_lbfgs(coeffs_init, deadline, beta=50.0):
    """Refine support function coefficients using L-BFGS-B."""
    if time.time() > deadline:
        return None, -np.inf
    try:
        res = minimize(
            lambda x: _objective_support_soft(x, beta=beta),
            coeffs_init,
            method='L-BFGS-B',
            options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-10, 'maxfun': 100000}
        )
        if time.time() > deadline:
            return None, -np.inf
        if not _is_valid_support(res.x):
            return None, -np.inf
        cand_pts = _vertices_from_support(res.x)
        if cand_pts is None:
            return None, -np.inf
        cand_score = _score(cand_pts)
        return cand_pts, cand_score
    except Exception:
        return None, -np.inf

def _refine_support_nm(coeffs_init, deadline):
    """Refine support function coefficients using Nelder-Mead."""
    if time.time() > deadline:
        return None, -np.inf
    try:
        def obj(coeffs):
            if time.time() > deadline:
                return 1e10
            return _objective_support(coeffs)
        res = minimize(
            obj, coeffs_init, method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        if time.time() > deadline:
            return None, -np.inf
        if res.fun < 1e9 and _is_valid_support(res.x):
            cand_pts = _vertices_from_support(res.x)
            if cand_pts is not None:
                cand_score = _score(cand_pts)
                return cand_pts, cand_score
    except Exception:
        pass
    return None, -np.inf

def _normalize_pts(pts):
    """Normalize points to unit convex hull area."""
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
    except Exception:
        return pts
    if hull_area > 1e-12:
        scale = np.sqrt(1.0 / hull_area)
        return pts * scale
    return pts

def construct():
    """Place 14 points in convex position to maximize min triangle area / hull area."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_points = None
    best_score = 0.0
    
    def update_best(pts, score):
        nonlocal best_points, best_score
        if score > best_score:
            best_score = score
            best_points = pts.copy()
    
    # Initial guess: Regular 14-gon corresponds to h(theta) = R (constant)
    # R is chosen such that the polygon has some area.
    # For a regular n-gon with circumradius R, area = 0.5 * n * R^2 * sin(2*pi/n)
    # We just need a valid starting point.
    # h(theta) = 1.0 -> Circle of radius 1.
    # Vertices of inscribed 14-gon in unit circle.
    init_coeffs = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    if not _is_valid_support(init_coeffs):
        # Fallback to a slightly perturbed constant if invalid (shouldn't happen for 1.0)
        init_coeffs = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
    init_pts = _vertices_from_support(init_coeffs)
    if init_pts is not None:
        init_score = _score(init_pts)
        update_best(init_pts, init_score)
    else:
        # Fallback: Regular 14-gon
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        init_pts = np.column_stack([np.cos(angles), np.sin(angles)])
        init_score = _score(init_pts)
        update_best(init_pts, init_score)

    # Phase 1: Differential Evolution in coefficient space
    if time.time() < deadline - 60.0:
        bounds = [(-0.2, 0.2)] * 7
        bounds[0] = (0.5, 2.0) # c0 must be positive and large enough
        try:
            def de_obj(coeffs):
                if time.time() > deadline - 5:
                    return 1e10
                return _objective_support_soft(coeffs, beta=30.0)
            
            res_de = differential_evolution(
                de_obj, bounds,
                seed=42,
                maxiter=40,
                popsize=20,
                tol=1e-6,
                init='latinhypercube',
                polish=False,
                update_interval=1,
                callback=None
            )
            
            if time.time() < deadline - 30.0 and res_de.fun < 1e9 and _is_valid_support(res_de.x):
                cand_pts = _vertices_from_support(res_de.x)
                if cand_pts is not None:
                    cand_score = _score(cand_pts)
                    update_best(cand_pts, cand_score)
                
        except Exception:
            pass
    
    # Phase 2: L-BFGS-B refinement in coefficient space (soft-min)
    if time.time() < deadline - 40.0:
        # Use best coeffs from DE if available, else init
        # We didn't store coeffs, so we restart from init or try to recover?
        # Simpler: Just run L-BFGS from init and from a few random valid starts.
        
        cand_pts, cand_score = _refine_support_lbfgs(init_coeffs, deadline, beta=50.0)
        if cand_pts is not None and cand_score > best_score:
            update_best(cand_pts, cand_score)
            
        # Try a few random valid starts
        rng = np.random.RandomState(123)
        for _ in range(5):
            if time.time() > deadline - 20:
                break
            c0 = rng.uniform(0.8, 1.5)
            c1_6 = rng.uniform(-0.1, 0.1, 6)
            rand_coeffs = np.concatenate([[c0], c1_6])
            if _is_valid_support(rand_coeffs):
                cand_pts, cand_score = _refine_support_lbfgs(rand_coeffs, deadline, beta=50.0)
                if cand_pts is not None and cand_score > best_score:
                    update_best(cand_pts, cand_score)

    # Phase 3: Nelder-Mead refinement in coefficient space (hard objective)
    if time.time() < deadline - 20.0:
        cand_pts, cand_score = _refine_support_nm(init_coeffs, deadline)
        if cand_pts is not None and cand_score > best_score:
            update_best(cand_pts, cand_score)
            
        # Try Nelder-Mead from best found points? 
        # We don't have coeffs for best_points.
        # We can try to fit coeffs to best_points? Too complex.
        # Instead, try NM from a few random valid starts.
        rng = np.random.RandomState(456)
        for _ in range(3):
            if time.time() > deadline - 10:
                break
            c0 = rng.uniform(0.8, 1.5)
            c1_6 = rng.uniform(-0.1, 0.1, 6)
            rand_coeffs = np.concatenate([[c0], c1_6])
            if _is_valid_support(rand_coeffs):
                cand_pts, cand_score = _refine_support_nm(rand_coeffs, deadline)
                if cand_pts is not None and cand_score > best_score:
                    update_best(cand_pts, cand_score)

    if best_points is None:
        # Fallback: regular 14-gon
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Normalize to unit area
    best_points = _normalize_pts(best_points)
    
    return best_points
# EVOLVE-BLOCK-END
