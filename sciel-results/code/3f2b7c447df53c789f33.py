import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution
import time

N = 14

def _min_triangle_area(points):
    """Compute minimum triangle area for N points (N<=14)."""
    n = points.shape[0]
    if n < 3:
        return 0.0
    min_area = np.inf
    for i in range(n):
        for j in range(i+1, n):
            v1 = points[j] - points[i]
            for k in range(j+1, n):
                v2 = points[k] - points[i]
                area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
                if area < min_area:
                    min_area = area
    return min_area

def _score(points):
    """Compute min triangle area / convex hull area."""
    if points.shape != (N, 2):
        return 0.0
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _neg_score(points):
    """Negative score for minimization."""
    return -_score(points)

def _make_angular_ring(n, radius=1.0, phase=0.0):
    """Create n points on a circle of given radius."""
    angles = np.linspace(0, 2*np.pi, n, endpoint=False) + phase
    return np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))

def _construct_two_ring(n_out, r_in, phase_in):
    """Construct two-ring configuration."""
    outer = _make_angular_ring(n_out, 1.0, 0.0)
    if n_out == N:
        return outer
    inner = _make_angular_ring(N - n_out, r_in, phase_in)
    return np.vstack((outer, inner))

def _optimize_two_ring(n_out, time_budget):
    """Optimize two-ring configuration with n_out outer points."""
    n_in = N - n_out
    if n_in == 0:
        pts = _make_angular_ring(N, 1.0, 0.0)
        return _score(pts), pts
    
    def objective(params):
        r_in, phase_in = params
        if r_in < 0.01 or r_in > 1.4:
            return 1e6
        pts = _construct_two_ring(n_out, r_in, phase_in)
        return _neg_score(pts)
    
    bounds = [(0.05, 1.3), (0, 2*np.pi)]
    try:
        result = differential_evolution(
            objective, bounds, 
            maxiter=50, popsize=15,
            tol=1e-6, seed=42,
            polish=True
        )
        best_params = result.x
        best_score = -result.fun
    except Exception:
        best_score = 0
        best_params = [0.5, 0.0]
        for r in np.linspace(0.1, 1.2, 12):
            for ph in np.linspace(0, 2*np.pi, 12, endpoint=False):
                s = -objective([r, ph])
                if s > best_score:
                    best_score = s
                    best_params = [r, ph]
    
    try:
        res = minimize(objective, best_params, method='Nelder-Mead',
                       options={'xatol':1e-8, 'fatol':1e-8, 'maxiter':200})
        if -res.fun > best_score:
            best_score = -res.fun
            best_params = res.x
    except Exception:
        pass
    
    pts = _construct_two_ring(n_out, best_params[0], best_params[1])
    return best_score, pts

def _polish_full(points, time_budget):
    """Polish full 28D configuration."""
    def obj_flat(x):
        p = x.reshape(N, 2)
        return _neg_score(p)
    
    best_score = _score(points)
    best_pts = points.copy()
    
    try:
        res = minimize(obj_flat, points.flatten(), method='Nelder-Mead',
                       options={'xatol':1e-7, 'fatol':1e-7, 'maxiter':300})
        pts = res.x.reshape(N, 2)
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts
    except Exception:
        pass
    
    return best_score, best_pts

def _heptagon_pair_base():
    """Create base heptagon with paired points at each vertex."""
    angles = np.linspace(0, 2*np.pi, 7, endpoint=False)
    base = np.column_stack((np.cos(angles), np.sin(angles)))
    pairs = []
    for i in range(7):
        pairs.append(base[i])
        pairs.append(base[i])
    return np.array(pairs)

def _heptagon_pair_params_to_points(offsets, affine_params):
    """Convert heptagon-pair parameters to actual points."""
    angles = np.linspace(0, 2*np.pi, 7, endpoint=False)
    base = np.column_stack((np.cos(angles), np.sin(angles)))
    
    points = np.zeros((14, 2))
    for i in range(7):
        points[2*i] = base[i]
        dir_vec = base[i] / np.linalg.norm(base[i])
        points[2*i+1] = base[i] + offsets[i] * dir_vec
    
    a, b, c, d = affine_params
    M = np.array([[a, b], [c, d]])
    points = points @ M.T
    
    return points

def _heptagon_pair_score(params):
    """Score for heptagon-pair parameterization."""
    offsets = params[:7]
    affine = params[7:11]
    
    offsets = np.clip(offsets, -0.5, 0.8)
    
    pts = _heptagon_pair_params_to_points(offsets, affine)
    
    if np.any(np.isnan(pts)) or np.any(np.isinf(pts)):
        return -1e6
    
    return -_score(pts)

def _optimize_heptagon_pair(time_budget):
    """Optimize heptagon-pair configuration."""
    init = np.zeros(11)
    init[:7] = 0.1
    init[7:11] = [1.0, 0.0, 0.0, 1.0]
    
    bounds = [(-0.3, 0.6)] * 7 + [(0.3, 1.5), (-0.5, 0.5), (-0.5, 0.5), (0.3, 1.5)]
    
    best_score = 0
    best_params = init.copy()
    
    try:
        result = differential_evolution(
            _heptagon_pair_score, bounds,
            maxiter=30, popsize=12,
            tol=1e-5, seed=99,
            polish=False
        )
        best_params = result.x
        best_score = -result.fun
    except Exception:
        pass
    
    if time.time() < time_budget:
        try:
            res = minimize(_heptagon_pair_score, best_params, method='Nelder-Mead',
                          options={'xatol':1e-8, 'fatol':1e-8, 'maxiter':200})
            if -res.fun > best_score:
                best_score = -res.fun
                best_params = res.x
        except Exception:
            pass
    
    pts = _heptagon_pair_params_to_points(best_params[:7], best_params[7:11])
    return best_score, pts

def _soft_min_score(points, beta=50.0):
    """Soft-min of triangle areas for smoother optimization."""
    n = points.shape[0]
    if n < 3:
        return 0.0
    
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            v1 = points[j] - points[i]
            for k in range(j+1, n):
                v2 = points[k] - points[i]
                area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
                areas.append(area)
    
    areas = np.array(areas)
    if len(areas) == 0:
        return 0.0
    
    min_val = np.min(areas)
    if min_val < 1e-12:
        return 0.0
    
    shifted = areas - min_val
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    soft_min = min_val - (1.0/beta) * log_sum
    
    return soft_min

def _soft_neg_score(points, beta=50.0):
    """Negative soft score for minimization."""
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-9:
            return 1e6
    except Exception:
        return 1e6
    
    soft_min = _soft_min_score(points, beta)
    return -soft_min / hull_area

def _polish_with_soft_min(points, time_budget):
    """Polish using soft-min objective for better gradient flow."""
    best_score = _score(points)
    best_pts = points.copy()
    
    def obj_soft(x):
        p = x.reshape(N, 2)
        return _soft_neg_score(p, beta=30.0)
    
    try:
        res = minimize(obj_soft, points.flatten(), method='Nelder-Mead',
                       options={'xatol':1e-8, 'fatol':1e-8, 'maxiter':300})
        pts = res.x.reshape(N, 2)
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts
    except Exception:
        pass
    
    return best_score, best_pts

# ===== NEW: Support Function Parameterization =====

def _support_functions_to_points(h):
    """Convert support function values to convex polygon vertices.
    
    h: array of N support function values (distances from origin to tangent lines)
    Returns: array of shape (N, 2) with the vertices of the convex polygon.
    
    The polygon is defined by N edges, where edge i is tangent to the circle
    of radius h[i] at angle theta_i = 2*pi*i/N.
    """
    n = len(h)
    thetas = 2 * np.pi * np.arange(n) / n
    
    # Edge normals
    normals = np.column_stack((np.cos(thetas), np.sin(thetas)))
    
    # Each edge is defined by: normal[i] . x = h[i]
    # Vertex i is the intersection of edges i-1 and i (mod n)
    
    vertices = np.zeros((n, 2))
    for i in range(n):
        i_prev = (i - 1) % n
        # Solve: n_prev . v = h_prev, n_curr . v = h_curr
        A = np.array([normals[i_prev], normals[i]])
        b = np.array([h[i_prev], h[i]])
        try:
            v = np.linalg.solve(A, b)
            vertices[i] = v
        except np.linalg.LinAlgError:
            vertices[i] = normals[i] * h[i]  # fallback
    
    return vertices

def _hull_area_from_support(h):
    """Compute convex hull area analytically from support functions.
    
    Area = sum over edges of (0.5 * edge_length * h[i])
    where edge_length is computed from adjacent vertices.
    """
    n = len(h)
    thetas = 2 * np.pi * np.arange(n) / n
    normals = np.column_stack((np.cos(thetas), np.sin(thetas)))
    
    # Compute vertices
    vertices = _support_functions_to_points(h)
    
    # Area via shoelace formula
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i][0] * vertices[j][1] - vertices[j][0] * vertices[i][1]
    area = 0.5 * abs(area)
    
    return area

def _support_score(params, beta=100.0):
    """Score for support function parameterization.
    
    params: [h_0, ..., h_{N-1}] where h_i > 0 are support function values.
    Uses soft-min for smoothness.
    """
    h = np.abs(params) + 0.1  # Ensure positive
    
    # Check for valid convex polygon
    if np.any(h < 0.05) or np.any(h > 5.0):
        return 1e6
    
    vertices = _support_functions_to_points(h)
    
    if np.any(np.isnan(vertices)) or np.any(np.isinf(vertices)):
        return 1e6
    
    # Compute hull area analytically
    hull_area = _hull_area_from_support(h)
    if hull_area < 1e-9:
        return 1e6
    
    # Compute soft-min of triangle areas
    n = len(vertices)
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            v1 = vertices[j] - vertices[i]
            for k in range(j+1, n):
                v2 = vertices[k] - vertices[i]
                area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
                areas.append(area)
    
    areas = np.array(areas)
    if len(areas) == 0:
        return 1e6
    
    min_val = np.min(areas)
    if min_val < 1e-12:
        return 1e6
    
    shifted = areas - min_val
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    soft_min = min_val - (1.0/beta) * log_sum
    
    return -soft_min / hull_area

def _optimize_support_functions(time_budget):
    """Optimize using support function parameterization."""
    # Initial guess: regular N-gon (all h equal)
    init = np.ones(N) * 1.0
    
    bounds = [(0.1, 3.0)] * N
    
    best_score = 0
    best_params = init.copy()
    
    # Global search
    try:
        result = differential_evolution(
            _support_score, bounds,
            maxiter=40, popsize=15,
            tol=1e-5, seed=123,
            polish=False
        )
        best_params = result.x
        best_score = -result.fun
    except Exception:
        pass
    
    # Local polish with Nelder-Mead
    if time.time() < time_budget:
        try:
            res = minimize(_support_score, best_params, method='Nelder-Mead',
                          options={'xatol':1e-8, 'fatol':1e-8, 'maxiter':300})
            if -res.fun > best_score:
                best_score = -res.fun
                best_params = res.x
        except Exception:
            pass
    
    # Convert to points
    h = np.abs(best_params) + 0.1
    pts = _support_functions_to_points(h)
    
    return best_score, pts

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_score = 0
    best_points = None
    
    def try_update(score, points):
        nonlocal best_score, best_points
        if score > best_score:
            best_score = score
            best_points = points.copy()
    
    # Try support function optimization first (new design)
    if time.time() < deadline - 30:
        time_budget = min(25.0, deadline - time.time() - 30)
        score, pts = _optimize_support_functions(time_budget)
        try_update(score, pts)
        
        # Polish with soft-min
        if best_points is not None and time.time() < deadline - 15:
            s2, pts2 = _polish_with_soft_min(best_points, deadline - time.time() - 15)
            try_update(s2, pts2)
    
    # Try heptagon-pair configuration
    if time.time() < deadline - 25:
        time_budget = min(15.0, deadline - time.time() - 25)
        score, pts = _optimize_heptagon_pair(time_budget)
        try_update(score, pts)
    
    # Try various two-ring configurations
    splits = [(14,0), (12,2), (10,4), (8,6), (7,7)]
    
    for n_out, n_in in splits:
        if time.time() > deadline - 20:
            break
        
        remaining = deadline - 20 - time.time()
        if remaining < 5:
            break
        
        time_budget = min(10.0, remaining / 2.0)
        score, pts = _optimize_two_ring(n_out, time_budget)
        try_update(score, pts)
    
    # Final polish on best found
    if best_points is not None and time.time() < deadline - 5:
        s, pts = _polish_full(best_points, deadline - time.time() - 5)
        try_update(s, pts)
        
        if time.time() < deadline - 3:
            s2, pts2 = _polish_with_soft_min(best_points, deadline - time.time() - 3)
            try_update(s2, pts2)
    
    if best_points is None:
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
    
    return best_points
