"""Heilbronn convex regions n=14: maximize min triangle area / convex hull area."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

# Pre-compute triangle indices for vectorization
# C(14, 3) = 364 triangles
TRI_IDX = np.array(list(combinations(range(N), 3)))

def _convex_hull_area(points):
    """Compute convex hull area of points."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        pts = np.array(points)
        pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
        if len(pts) <= 1:
            return 0.0
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        lower = []
        for p in pts:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)
        upper = []
        for p in reversed(pts):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)
        hull = lower[:-1] + upper[:-1]
        if len(hull) < 3:
            return 0.0
        x = hull[:, 0]
        y = hull[:, 1]
        return 0.5 * np.abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))

def _all_triangle_areas(points):
    """Compute all triangle areas among all triples."""
    pi = points[TRI_IDX[:, 0]]
    pj = points[TRI_IDX[:, 1]]
    pk = points[TRI_IDX[:, 2]]
    
    # Vectorized cross product area
    # Area = 0.5 * | (xj-xi)(yk-yi) - (xk-xi)(yj-yi) |
    cross_prod = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pk[:, 0] - pi[:, 0]) * (pj[:, 1] - pi[:, 1])
    return 0.5 * np.abs(cross_prod)

def _score(points):
    """Compute the minimum triangle area divided by the convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-15:
        return 0.0
    min_area = np.min(_all_triangle_areas(points))
    return min_area / hull_area

def _decode_polygon(params):
    """
    Decode polygon parameters to 14 points.
    params: [14 edge lengths, 13 internal angles]
    The 14th internal angle is determined by the sum constraint: (N-2)*pi.
    """
    edge_lengths = params[:N]
    internal_angles = params[N:N+13]
    
    # The sum of internal angles in an N-gon is (N-2)*pi
    sum_internal = (N - 2) * np.pi
    # Calculate the 14th angle
    a14 = sum_internal - np.sum(internal_angles)
    
    # If a14 is not strictly positive or too small, the polygon is invalid or degenerate
    if a14 <= 1e-6:
        return None, 0.0
        
    all_angles = np.concatenate([internal_angles, [a14]])
    
    # To construct the polygon, we need the exterior angles.
    # Exterior angle = pi - internal angle
    # The sum of exterior angles is 2*pi.
    # Let's verify: sum(pi - a_i) = N*pi - sum(a_i) = N*pi - (N-2)*pi = 2*pi. Correct.
    exterior_angles = np.pi - all_angles
    
    # Start at origin
    points = np.zeros((N, 2))
    current_pos = np.array([0.0, 0.0])
    current_angle = 0.0 # Initial direction
    
    for i in range(N):
        if i > 0:
            points[i] = current_pos
            current_angle += exterior_angles[i-1]
            current_pos = current_pos + edge_lengths[i-1] * np.array([np.cos(current_angle), np.sin(current_angle)])
        else:
            points[i] = current_pos
            # The first edge direction is 0
            current_pos = current_pos + edge_lengths[i] * np.array([np.cos(current_angle), np.sin(current_angle)])
            
    # The loop above places N points. 
    # Wait, the logic above is slightly off for the last point.
    # Let's re-implement carefully.
    # p0 = 0
    # p_{i+1} = p_i + L_i * (cos(theta_i), sin(theta_i))
    # theta_0 = 0
    # theta_{i+1} = theta_i + exterior_angle_i
    
    points = np.zeros((N, 2))
    pos = np.array([0.0, 0.0])
    theta = 0.0
    points[0] = pos
    
    for i in range(N):
        # Add edge i
        pos = pos + edge_lengths[i] * np.array([np.cos(theta), np.sin(theta)])
        if i < N - 1:
            points[i+1] = pos
            theta += exterior_angles[i]
            
    # Check if it closed (it should by construction of angles, but floating point might drift)
    # The last point calculated is p_N which should be p_0 (0,0)
    # But we only stored N points: p_0 ... p_{N-1}
    # The loop runs i=0 to N-1.
    # i=0: pos = p_1. points[1] = p_1. theta += ext_0.
    # ...
    # i=N-1: pos = p_N (should be 0). points[N] is not set.
    # So points[0] to points[N-1] are the vertices.
    
    # Ensure convexity? 
    # If all exterior angles are in (0, pi), the polygon is strictly convex.
    # We enforce this in bounds.
    
    return points, _convex_hull_area(points)

def _objective(params):
    """
    Objective function for SLSQP.
    Returns negative score (to maximize).
    """
    points, hull_area = _decode_polygon(params)
    if points is None:
        return 1e10
        
    areas = _all_triangle_areas(points)
    
    # Soft min to smooth the objective
    # Use a moderate beta for SLSQP stability
    beta = 20.0 
    min_val = np.min(areas)
    shifted = areas - min_val
    # Avoid overflow in exp
    max_exp = np.max(-beta * shifted)
    if max_exp > 700:
        # Fallback to hard min if too unstable
        soft_min = min_val
    else:
        log_sum = np.log(np.sum(np.exp(-beta * shifted)))
        soft_min = min_val - log_sum / beta
        
    if hull_area < 1e-15:
        return 1e10
        
    return -soft_min / hull_area

def _bounds():
    """
    Define bounds for SLSQP.
    14 edge lengths: (0.1, 10.0)
    13 internal angles: (1e-3, pi - 1e-3)
    """
    bounds = []
    for _ in range(N):
        bounds.append((0.1, 10.0))
    for _ in range(N - 1):
        bounds.append((1e-3, np.pi - 1e-3))
    return bounds

def _generate_initial_guesses(rng, num_guesses):
    """
    Generate initial guesses for SLSQP.
    Mix of regular polygon perturbations and random valid polygons.
    """
    guesses = []
    bounds = _bounds()
    
    for _ in range(num_guesses):
        # Start with a regular-ish polygon
        # Regular N-gon internal angle: (N-2)*pi / N
        reg_angle = (N - 2) * np.pi / N
        # Regular N-gon edge length if circumradius R=1: 2*sin(pi/N)
        reg_edge = 2 * np.sin(np.pi / N)
        
        # Perturb angles
        angles = np.full(N - 1, reg_angle)
        # Add random noise
        noise_scale = 0.2
        angles += rng.normal(0, noise_scale, N - 1)
        # Clip to valid range to ensure a14 is likely valid
        angles = np.clip(angles, 1e-3, np.pi - 1e-3)
        
        # Check if a14 would be valid
        a14 = (N - 2) * np.pi - np.sum(angles)
        if a14 <= 1e-3 or a14 >= np.pi - 1e-3:
            # Retry or adjust
            # Simple adjustment: scale angles to fit
            target_sum = (N - 2) * np.pi - (np.pi / 2) # Aim for a14 = pi/2
            current_sum = np.sum(angles)
            if current_sum > 0:
                angles = angles * (target_sum / current_sum)
                angles = np.clip(angles, 1e-3, np.pi - 1e-3)
        
        # Perturb edges
        edges = np.full(N, reg_edge)
        edges += rng.normal(0, 0.2, N)
        edges = np.clip(edges, 0.1, 10.0)
        
        params = np.concatenate([edges, angles])
        guesses.append(params)
        
    return guesses

def construct():
    """
    Generate 14 points in a planar convex region normalized to unit area
    to maximize the minimum area over all triples.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 140s limit, keep 10s safety margin
    
    best_points = None
    best_score = -1
    
    rng = np.random.RandomState(42)
    
    # Initial fallback: Regular 14-gon
    angles_reg = np.linspace(0, 2 * np.pi, N, endpoint=False)
    initial_points = np.column_stack([np.cos(angles_reg), np.sin(angles_reg)])
    hull_area = _convex_hull_area(initial_points)
    if hull_area > 1e-15:
        scale = np.sqrt(1.0 / hull_area)
        initial_points = initial_points * scale
    best_points = initial_points.copy()
    best_score = _score(initial_points)
    
    bounds = _bounds()
    
    # Multi-start SLSQP
    num_starts = 20
    guesses = _generate_initial_guesses(rng, num_starts)
    
    for i, init_params in enumerate(guesses):
        if time.time() > deadline - 10:
            break
            
        try:
            result = minimize(
                _objective,
                init_params,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-12}
            )
            
            if result.fun < 0:
                opt_points, hull_area = _decode_polygon(result.x)
                if opt_points is not None and hull_area > 1e-15:
                    # Normalize to unit area
                    scale = np.sqrt(1.0 / hull_area)
                    opt_points = opt_points * scale
                    
                    opt_score = _score(opt_points)
                    if opt_score > best_score:
                        best_score = opt_score
                        best_points = opt_points.copy()
                        
        except Exception:
            pass
            
    # Final check
    if best_points is None:
        angles_reg = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles_reg), np.sin(angles_reg)])
        hull_area = _convex_hull_area(best_points)
        if hull_area > 1e-15:
            scale = np.sqrt(1.0 / hull_area)
            best_points = best_points * scale
            
    return best_points
# EVOLVE-BLOCK-END
