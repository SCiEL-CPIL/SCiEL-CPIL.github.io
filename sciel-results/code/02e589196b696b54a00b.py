"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution, minimize_scalar

def _compute_all_triangle_areas(pts):
    """Compute all C(n,3) triangle areas for n points."""
    n = len(pts)
    areas = []
    for i in range(n - 2):
        pi = pts[i]
        for j in range(i + 1, n - 1):
            pj = pts[j]
            pk = pts[j+1:]
            v1x = pj[0] - pi[0]
            v1y = pj[1] - pi[1]
            v2x = pk[:, 0] - pi[0]
            v2y = pk[:, 1] - pi[1]
            crosses = np.abs(v1x * v2y - v1y * v2x)
            areas.extend(0.5 * crosses)
    return np.array(areas)

def _score(pts):
    """
    Compute the score: min triangle area / convex hull area.
    """
    if len(pts) < 3:
        return 0.0
    
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
    except Exception:
        return 0.0
        
    if hull_area < 1e-12:
        return 0.0
        
    areas = _compute_all_triangle_areas(pts)
    min_area = np.min(areas)
    
    return min_area / hull_area

def _soft_min(values, temperature=0.05):
    """Smooth approximation of minimum using log-sum-exp."""
    values = np.asarray(values)
    if len(values) == 0:
        return 0.0
    m = np.min(values)
    if m < 1e-12:
        return m
    log_sum = np.log(np.sum(np.exp((values - m) / temperature)))
    return m - temperature * log_sum

def _generate_heptagon_pair_points(radii):
    """
    Generate 14 points: for each of 7 heptagon vertices, place a pair
    at radii radii[i] and radii[i] + delta (or just use two radii per vertex).
    
    Actually, let's interpret this as: 7 vertices of regular heptagon, each
    replaced by a pair of points. The pair for vertex i is at angles
    (i*2pi/7 - eps) and (i*2pi/7 + eps) with optimized radii.
    
    Simpler interpretation: 7 pairs, each pair consists of two points at
    angle i*2pi/7 ± small_offset, with radii r1_i and r2_i.
    
    For symmetry, let's use: for vertex i, points at angles (i*2pi/7 - delta_angle)
    and (i*2pi/7 + delta_angle), with radii r_inner and r_outer (same for all i).
    
    Even simpler: two concentric heptagons with angular offset.
    This is the two-ring structure. Let me implement the "pair per vertex" more literally.
    
    Interpretation: 14 points where point 2i is at angle i*2pi/7 with radius r_i,
    and point 2i+1 is at angle i*2pi/7 + pi/7 (half-step) with radius s_i.
    This gives 14 points with 7-fold rotational symmetry if r_i and s_i are constant.
    
    Let's use: 7 pairs. Pair i has points at angles (i*2pi/7) and (i*2pi/7 + pi/7),
    with radii a and b respectively. So 14 points total.
    """
    n_hept = 7
    angles1 = np.linspace(0, 2*np.pi, n_hept, endpoint=False)
    angles2 = angles1 + np.pi / n_hept  # half-step offset
    
    r1 = np.full(n_hept, radii[0])  # radius for first point of each pair
    r2 = np.full(n_hept, radii[1])  # radius for second point of each pair
    
    pts1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
    pts2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
    
    # Interleave: p0, q0, p1, q1, ...
    pts = np.empty((2 * n_hept, 2))
    pts[0::2] = pts1
    pts[1::2] = pts2
    
    return pts

def _apply_affine(pts, a, b, c, d):
    """Apply affine transformation: [x'] = [a b; c d] [x; y]"""
    M = np.array([[a, b], [c, d]])
    return pts @ M.T

def _score_heptagon_pair(r1, r2, a, b, c, d):
    """
    Score for heptagon-pair configuration with affine transform.
    r1, r2: radii for the two points in each pair
    a, b, c, d: affine transformation matrix
    """
    pts = _generate_heptagon_pair_points([r1, r2])
    pts = _apply_affine(pts, a, b, c, d)
    
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
    except Exception:
        return 0.0
    
    if hull_area < 1e-12:
        return 0.0
    
    areas = _compute_all_triangle_areas(pts)
    min_area = np.min(areas)
    
    return min_area / hull_area

def _soft_score_heptagon_pair(r1, r2, a, b, c, d, temperature=0.05):
    """Soft-min score for better gradient information."""
    pts = _generate_heptagon_pair_points([r1, r2])
    pts = _apply_affine(pts, a, b, c, d)
    
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
    except Exception:
        return 0.0
    
    if hull_area < 1e-12:
        return 0.0
    
    areas = _compute_all_triangle_areas(pts)
    sm = _soft_min(areas, temperature)
    
    return sm / hull_area

def _optimize_heptagon_pair(deadline):
    """
    Optimize the heptagon-pair configuration with affine transformation.
    6 parameters: r1, r2, a, b, c, d
    """
    start = time.time()
    
    # Initial guess: identity affine, r1=1.0, r2=0.6
    best_score = -float('inf')
    best_params = None
    
    # Multi-start with different initial conditions
    initial_guesses = [
        [1.0, 0.6, 1.0, 0.0, 0.0, 1.0],
        [1.0, 0.5, 1.0, 0.0, 0.0, 1.0],
        [1.0, 0.7, 1.0, 0.0, 0.0, 1.0],
        [1.0, 0.6, 1.0, 0.2, 0.0, 1.0],
        [1.0, 0.6, 0.8, 0.0, 0.0, 1.2],
        [1.0, 0.6, 1.0, 0.0, 0.3, 0.8],
    ]
    
    for guess in initial_guesses:
        if time.time() > deadline - 20.0:
            break
        
        def objective(params):
            r1, r2, a, b, c, d = params
            # Constraints: ensure non-degenerate affine
            det = a * d - b * c
            if abs(det) < 0.1:
                return 1e9
            if r1 < 0.2 or r1 > 2.0 or r2 < 0.2 or r2 > 2.0:
                return 1e9
            return -_soft_score_heptagon_pair(r1, r2, a, b, c, d, temperature=0.03)
        
        try:
            res = minimize(objective, guess, method='Nelder-Mead',
                          options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 300, 'adaptive': True})
            
            if res.fun < 1e8:
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_params = res.x.copy()
        except Exception:
            continue
    
    if best_params is None:
        return None, -float('inf')
    
    # Polish with exact score
    def exact_objective(params):
        r1, r2, a, b, c, d = params
        det = a * d - b * c
        if abs(det) < 0.1:
            return 1e9
        if r1 < 0.2 or r1 > 2.0 or r2 < 0.2 or r2 > 2.0:
            return 1e9
        return -_score_heptagon_pair(r1, r2, a, b, c, d)
    
    try:
        res = minimize(exact_objective, best_params, method='Nelder-Mead',
                      options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 200, 'adaptive': True})
        
        if res.fun < 1e8:
            score = -res.fun
            if score > best_score:
                best_score = score
                best_params = res.x.copy()
    except Exception:
        pass
    
    r1, r2, a, b, c, d = best_params
    pts = _generate_heptagon_pair_points([r1, r2])
    pts = _apply_affine(pts, a, b, c, d)
    
    # Final verification
    final_score = _score(pts)
    if final_score > best_score:
        best_score = final_score
    
    return pts, best_score

def _optimize_two_ring(deadline):
    """
    Optimize two-ring configuration (7 outer, 7 inner) - kept as fallback.
    """
    best_score = -float('inf')
    best_r = 0.6
    best_theta = np.pi / 7
    
    theta_coarse = np.linspace(0, np.pi/7, 10)
    
    for th in theta_coarse:
        if time.time() > deadline - 15.0:
            break
            
        def neg_score_r(r):
            if r < 0.1 or r > 0.99: return 1e9
            pts = np.column_stack([
                np.cos(np.linspace(0, 2*np.pi, 7, endpoint=False)),
                np.sin(np.linspace(0, 2*np.pi, 7, endpoint=False))
            ])
            inner_angles = np.linspace(0, 2*np.pi, 7, endpoint=False) + th
            inner_pts = r * np.column_stack([np.cos(inner_angles), np.sin(inner_angles)])
            all_pts = np.vstack([pts, inner_pts])
            return -_score(all_pts)
            
        try:
            res_r = minimize_scalar(neg_score_r, bounds=(0.3, 0.95), method='bounded', options={'xatol': 1e-5})
            r_opt = res_r.x
            score_opt = -res_r.fun
        except Exception:
            continue
            
        if score_opt > best_score:
            best_score = score_opt
            best_r = r_opt
            best_theta = th
            
    if time.time() < deadline - 5.0:
        def neg_score_2d(params):
            r, th = params
            if r < 0.1 or r > 0.99: return 1e9
            outer_angles = np.linspace(0, 2*np.pi, 7, endpoint=False)
            outer_pts = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
            inner_angles = outer_angles + th
            inner_pts = r * np.column_stack([np.cos(inner_angles), np.sin(inner_angles)])
            all_pts = np.vstack([outer_pts, inner_pts])
            return -_score(all_pts)
        
        x0 = [best_r, best_theta]
        try:
            res = minimize(neg_score_2d, x0, method='Nelder-Mead', 
                           options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 100, 'adaptive': True})
            if res.fun < -best_score:
                best_r, best_theta = res.x
                best_score = -res.fun
        except Exception:
            pass
    
    outer_angles = np.linspace(0, 2*np.pi, 7, endpoint=False)
    outer_pts = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    inner_angles = outer_angles + best_theta
    inner_pts = best_r * np.column_stack([np.cos(inner_angles), np.sin(inner_angles)])
    best_pts = np.vstack([outer_pts, inner_pts])
    
    return best_pts, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin
    
    # Strategy:
    # 1. Try heptagon-pair with affine (new structural family, 6 params)
    # 2. Fall back to two-ring if needed
    # 3. Return the best
    
    pts_hp, score_hp = None, -float('inf')
    pts_2ring, score_2ring = None, -float('inf')
    
    # Try heptagon-pair first (new direction)
    if time.time() < deadline - 20.0:
        pts_hp, score_hp = _optimize_heptagon_pair(deadline)
    
    # Try two-ring as fallback
    if time.time() < deadline - 5.0:
        pts_2ring, score_2ring = _optimize_two_ring(deadline)
    
    # Compare and return best
    if pts_hp is not None and score_hp > score_2ring:
        return pts_hp
    elif pts_2ring is not None:
        return pts_2ring
    elif pts_hp is not None:
        return pts_hp
    else:
        # Absolute fallback: regular 14-gon
        angles = np.linspace(0, 2*np.pi, 14, endpoint=False)
        return np.column_stack([np.cos(angles), np.sin(angles)])

# EVOLVE-BLOCK-END
