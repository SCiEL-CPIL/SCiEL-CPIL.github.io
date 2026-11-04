import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution
import time

N = 14

# Precompute triple indices for vectorized area computation
TRIPLE_INDICES = np.array([
    (i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)
])

# Precompute which triples involve each point for fast local updates
POINT_TRIPLES = [[] for _ in range(N)]
for idx, (i, j, k) in enumerate(TRIPLE_INDICES):
    POINT_TRIPLES[i].append(idx)
    POINT_TRIPLES[j].append(idx)
    POINT_TRIPLES[k].append(idx)
POINT_TRIPLES = [np.array(pt) for pt in POINT_TRIPLES]

def _calculate_score(points):
    if points.shape != (N, 2):
        return 0.0
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0

    # Vectorized triangle area computation
    p_i = points[TRIPLE_INDICES[:, 0]]
    p_j = points[TRIPLE_INDICES[:, 1]]
    p_k = points[TRIPLE_INDICES[:, 2]]

    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]

    min_area = 0.5 * np.min(np.abs(cross))
    return min_area / hull_area

def _soft_min_score(points, temperature=0.05):
    """Smooth surrogate using log-sum-exp for the minimum triangle area."""
    if points.shape != (N, 2):
        return 0.0
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0

    p_i = points[TRIPLE_INDICES[:, 0]]
    p_j = points[TRIPLE_INDICES[:, 1]]
    p_k = points[TRIPLE_INDICES[:, 2]]

    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)

    # Soft minimum via log-sum-exp
    min_area = np.min(areas)
    if min_area < 1e-12:
        return 0.0
    shifted = areas - min_area
    log_sum = np.log(np.sum(np.exp(-shifted / temperature)))
    soft_min = min_area - temperature * log_sum
    
    return soft_min / hull_area

def _generate_fourier_points(coeffs):
    """
    Generate N points from 7 Fourier coefficients.
    coeffs = [R0, a1, b1, a2, b2, a3, b3]
    r(theta) = R0 + a1*cos(theta) + b1*sin(theta) + ...
    """
    R0, a1, b1, a2, b2, a3, b3 = coeffs
    
    # Ensure radius is positive to avoid degenerate points
    # We will clamp the final radius to a minimum value if it goes too low
    min_r = 0.1
    
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    
    radii = np.zeros(N)
    for i, theta in enumerate(angles):
        r = R0 + a1*np.cos(theta) + b1*np.sin(theta) + \
            a2*np.cos(2*theta) + b2*np.sin(2*theta) + \
            a3*np.cos(3*theta) + b3*np.sin(3*theta)
        radii[i] = max(r, min_r)
        
    points = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
    return points

def _fourier_objective(coeffs):
    """Objective function for Differential Evolution (minimize negative score)."""
    try:
        pts = _generate_fourier_points(coeffs)
        score = _calculate_score(pts)
        return -score
    except Exception:
        return 1e6

def _polish_points(points, time_limit=5.0):
    """High-precision polish using L-BFGS-B on soft-min, then coordinate descent on exact min."""
    start_time = time.time()
    
    best_pts = points.copy()
    best_score = _calculate_score(points)
    
    # 1. L-BFGS-B on soft-min surrogate to navigate the landscape
    temp = 0.05
    def neg_soft_score_flat(x):
        p = x.reshape(N, 2)
        return -_soft_min_score(p, temp)
    
    try:
        res = minimize(neg_soft_score_flat, points.flatten(), method='L-BFGS-B',
                      options={'maxiter':200, 'ftol':1e-12, 'gtol':1e-8, 'maxls':50})
        new_pts = res.x.reshape(N, 2)
        new_score = _calculate_score(new_pts)
        if new_score > best_score:
            best_pts = new_pts
            best_score = new_score
    except Exception:
        pass

    # 2. Coordinate Descent on exact min area (fast local search)
    if time.time() - start_time < time_limit * 0.7:
        cd_time = time_limit - (time.time() - start_time) - 1.0
        if cd_time > 0.5:
            best_pts = _coordinate_descent_polish(best_pts, cd_time)
            best_score = _calculate_score(best_pts)
            
    return best_pts, best_score

def _coordinate_descent_polish(points, time_limit=2.0):
    """
    Iteratively adjust each point to improve the minimum triangle area.
    Uses fast local updates for triples involving the moved point.
    """
    start_time = time.time()
    pts = points.copy()
    
    # Precompute current areas
    p_i = pts[TRIPLE_INDICES[:, 0]]
    p_j = pts[TRIPLE_INDICES[:, 1]]
    p_k = pts[TRIPLE_INDICES[:, 2]]
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    min_area = np.min(areas)
    
    step_sizes = [0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]
    
    for step in step_sizes:
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            for i in range(N):
                # Get indices of triples involving point i
                tri_idx = POINT_TRIPLES[i]
                
                best_move = None
                best_min_area = min_area
                
                # Current position
                p_i_curr = pts[i]
                
                # Try 4 cardinal directions and 4 diagonal directions
                directions = [
                    [step, 0], [-step, 0], [0, step], [0, -step],
                    [step, step], [step, -step], [-step, step], [-step, -step]
                ]
                
                for dx, dy in directions:
                    new_p = p_i_curr + np.array([dx, dy])
                    
                    # Update areas for triples involving i
                    new_areas = areas.copy()
                    
                    for t_idx in tri_idx:
                        i_t, j_t, k_t = TRIPLE_INDICES[t_idx]
                        p_a = pts[i_t] if i_t != i else new_p
                        p_b = pts[j_t] if j_t != i else new_p
                        p_c = pts[k_t] if k_t != i else new_p
                        
                        v1_t = p_b - p_a
                        v2_t = p_c - p_a
                        cross_t = v1_t[0] * v2_t[1] - v1_t[1] * v2_t[0]
                        new_areas[t_idx] = 0.5 * np.abs(cross_t)
                    
                    curr_min = np.min(new_areas)
                    if curr_min > best_min_area:
                        best_min_area = curr_min
                        best_move = new_p
                
                if best_move is not None:
                    pts[i] = best_move
                    min_area = best_min_area
                    improved = True
                    
                    # Recompute all areas to keep consistency for next iterations
                    p_i = pts[TRIPLE_INDICES[:, 0]]
                    p_j = pts[TRIPLE_INDICES[:, 1]]
                    p_k = pts[TRIPLE_INDICES[:, 2]]
                    v1 = p_j - p_i
                    v2 = p_k - p_i
                    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
                    areas = 0.5 * np.abs(cross)
                    min_area = np.min(areas)
    
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_score = 0
    best_points = None
    
    def try_update(score, points):
        nonlocal best_score, best_points
        if score > best_score:
            best_score = score
            best_points = points.copy()
    
    # 1) Regular Polygon (Baseline)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    pts_reg = np.column_stack((np.cos(angles), np.sin(angles)))
    score_reg = _calculate_score(pts_reg)
    try_update(score_reg, pts_reg)
    
    # Polish the regular polygon
    if time.time() < deadline - 40:
        polished_pts, polished_score = _polish_points(pts_reg, time_limit=10.0)
        try_update(polished_score, polished_pts)
        
    # 2) Fourier Series Optimization using Differential Evolution
    # This is the structural change: global search over a low-dimensional manifold of smooth shapes
    if time.time() < deadline - 50:
        # Bounds: R0 in [0.5, 2.0], coefficients in [-0.5, 0.5]
        bounds = [
            (0.5, 2.0), # R0
            (-0.5, 0.5), (-0.5, 0.5), # a1, b1
            (-0.5, 0.5), (-0.5, 0.5), # a2, b2
            (-0.5, 0.5), (-0.5, 0.5)  # a3, b3
        ]
        
        try:
            # Use DE to find good Fourier coefficients
            # popsize=15, maxiter=100 should be fast enough
            res = differential_evolution(
                _fourier_objective, 
                bounds, 
                seed=42, 
                maxiter=100, 
                popsize=15,
                tol=1e-5,
                atol=1e-5
            )
            
            if res.success or -res.fun > best_score:
                # Generate points from the best Fourier coefficients
                pts_fourier = _generate_fourier_points(res.x)
                score_fourier = _calculate_score(pts_fourier)
                
                # Polish these points
                if time.time() < deadline - 30:
                    polished_pts, polished_score = _polish_points(pts_fourier, time_limit=20.0)
                    try_update(polished_score, polished_pts)
                else:
                    try_update(score_fourier, pts_fourier)
                    
        except Exception:
            pass

    # 3) Two-ring configurations (Fallback/Alternative)
    if time.time() < deadline - 20:
        splits = [(7, 7), (8, 6)]
        for n_out, n_in in splits:
            if time.time() > deadline - 10:
                break
            
            # Simple grid search for two-ring parameters to avoid complex optimizer overhead
            best_ring_score = 0
            best_ring_params = None
            for r_in in [0.5, 0.7, 0.9]:
                for phi_rel in [0, np.pi/n_in, np.pi/(2*n_in)]:
                    angles_out = np.linspace(0, 2*np.pi, n_out, endpoint=False)
                    angles_in = np.linspace(0, 2*np.pi, n_in, endpoint=False) + phi_rel
                    pts_out = np.column_stack((np.cos(angles_out), np.sin(angles_out)))
                    pts_in = np.column_stack((r_in * np.cos(angles_in), r_in * np.sin(angles_in)))
                    pts = np.vstack((pts_out, pts_in))
                    score = _calculate_score(pts)
                    if score > best_ring_score:
                        best_ring_score = score
                        best_ring_params = (r_in, phi_rel)
            
            if best_ring_params is not None:
                # Polish the best ring configuration found
                r_in, phi_rel = best_ring_params
                angles_out = np.linspace(0, 2*np.pi, n_out, endpoint=False)
                angles_in = np.linspace(0, 2*np.pi, n_in, endpoint=False) + phi_rel
                pts_out = np.column_stack((np.cos(angles_out), np.sin(angles_out)))
                pts_in = np.column_stack((r_in * np.cos(angles_in), r_in * np.sin(angles_in)))
                pts = np.vstack((pts_out, pts_in))
                
                if time.time() < deadline - 10:
                    polished_pts, polished_score = _polish_points(pts, time_limit=10.0)
                    try_update(polished_score, polished_pts)

    # Final intensive polish on best points
    if best_points is not None and time.time() < deadline - 5:
        remaining = deadline - time.time() - 3
        final_pts, final_score = _polish_points(best_points, time_limit=remaining)
        try_update(final_score, final_pts)

    if best_points is None:
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
        best_score = _calculate_score(best_points)

    return best_points
