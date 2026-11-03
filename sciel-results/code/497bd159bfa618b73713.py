"""Heilbronn convex regions: maximize min triangle area / convex hull area for 13 points."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time
import itertools

def _min_triangle_area(pts):
    """Compute minimum triangle area among all triples, vectorized."""
    n = len(pts)
    if n < 3:
        return 0.0
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            if j + 1 >= n:
                continue
            v1 = pts[j] - pts[i]
            v2 = pts[j+1:] - pts[i]
            cross = np.abs(v1[0] * v2[:, 1] - v1[1] * v2[:, 0])
            areas = 0.5 * cross
            if len(areas) > 0:
                local_min = np.min(areas)
                if local_min < min_area:
                    min_area = local_min
    return min_area if min_area != float('inf') else 0.0

def _all_triangle_areas(pts):
    """Return array of all C(n,3) triangle areas."""
    n = len(pts)
    areas = []
    for i, j, k in itertools.combinations(range(n), 3):
        v1 = pts[j] - pts[i]
        v2 = pts[k] - pts[i]
        area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
        areas.append(area)
    return np.array(areas)

def _worst_triangle(pts):
    """Return (min_area, (i,j,k)) of the smallest triangle."""
    n = len(pts)
    min_area = float('inf')
    worst = (0, 1, 2)
    for i, j, k in itertools.combinations(range(n), 3):
        v1 = pts[j] - pts[i]
        v2 = pts[k] - pts[i]
        area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
        if area < min_area:
            min_area = area
            worst = (i, j, k)
    return min_area, worst

def _soft_min_triangle_area(pts, alpha=50.0):
    """
    Smooth approximation of the minimum triangle area using log-sum-exp.
    """
    n = len(pts)
    if n < 3:
        return 0.0
    areas = _all_triangle_areas(pts)
    if len(areas) == 0:
        return 0.0
    min_a = np.min(areas)
    shifted = -alpha * (areas - min_a)
    log_sum = np.log(np.sum(np.exp(shifted))) - alpha * min_a
    return -log_sum / alpha

def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score(pts):
    """Score = min triangle area / convex hull area."""
    n = len(pts)
    if n < 3:
        return 0.0
    hull_a = _hull_area(pts)
    if hull_a <= 1e-12:
        return 0.0
    return _min_triangle_area(pts) / hull_a

def _neg_score(params):
    pts = params.reshape(-1, 2)
    return -_score(pts)

def _neg_soft_score(params, alpha=50.0):
    pts = params.reshape(-1, 2)
    hull_a = _hull_area(pts)
    if hull_a <= 1e-12:
        return 1e10
    smooth_min = _soft_min_triangle_area(pts, alpha)
    return -(smooth_min / hull_a)

def _neg_soft_score_concentric(params, alpha=50.0):
    """Negative soft score for concentric-ring parameterization."""
    pts = _make_concentric_points(params)
    hull_a = _hull_area(pts)
    if hull_a <= 1e-12:
        return 1e10
    smooth_min = _soft_min_triangle_area(pts, alpha)
    return -(smooth_min / hull_a)

def _neg_score_concentric(params):
    """Negative exact score for concentric-ring parameterization."""
    pts = _make_concentric_points(params)
    return -_score(pts)

def _make_concentric_points(params):
    """
    Construct 13 points from concentric-ring parameters.
    
    params layout (variable length):
    - First 2: center point (cx, cy)
    - Then for each ring: [radius, angle_offset, n_points]
    
    We use a fixed structure: 1 center + 6 outer + 6 inner = 13 points
    params: [cx, cy, r_outer, th_outer, r_inner, th_inner]
    """
    cx, cy, r_out, th_out, r_in, th_in = params
    
    angles_out = np.linspace(0, 2 * np.pi, 6, endpoint=False) + th_out
    angles_in = np.linspace(0, 2 * np.pi, 6, endpoint=False) + th_in
    
    pts_out = np.column_stack([cx + r_out * np.cos(angles_out), cy + r_out * np.sin(angles_out)])
    pts_in = np.column_stack([cx + r_in * np.cos(angles_in), cy + r_in * np.sin(angles_in)])
    pts_c = np.array([[cx, cy]])
    
    return np.vstack([pts_out, pts_in, pts_c])

def _make_general_concentric(params):
    """
    More flexible concentric: 1 center + ring1(4 pts) + ring2(4 pts) + ring3(4 pts) = 13
    params: [cx, cy, r1, th1, r2, th2, r3, th3]
    """
    cx, cy, r1, th1, r2, th2, r3, th3 = params
    
    def ring(r, th, n):
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + th
        return np.column_stack([cx + r * np.cos(angles), cy + r * np.sin(angles)])
    
    pts_c = np.array([[cx, cy]])
    pts1 = ring(r1, th1, 4)
    pts2 = ring(r2, th2, 4)
    pts3 = ring(r3, th3, 4)
    
    return np.vstack([pts_c, pts1, pts2, pts3])

def _neg_soft_score_general(params, alpha=50.0):
    pts = _make_general_concentric(params)
    hull_a = _hull_area(pts)
    if hull_a <= 1e-12:
        return 1e10
    smooth_min = _soft_min_triangle_area(pts, alpha)
    return -(smooth_min / hull_a)

def _neg_score_general(params):
    pts = _make_general_concentric(params)
    return -_score(pts)

def _neg_symmetric_score(params):
    pts = _make_concentric_points(params)
    return -_score(pts)

def _neg_symmetric_smooth_score(params, alpha=50.0):
    pts = _make_concentric_points(params)
    hull_a = _hull_area(pts)
    if hull_a <= 1e-12:
        return 1e10
    smooth_min = _soft_min_triangle_area(pts, alpha)
    return -(smooth_min / hull_a)

def _targeted_worst_triangle_opt(pts, max_iter=50, step=0.01):
    """
    Iteratively push the vertices of the worst (smallest) triangle apart
    to increase its area, while checking that no other triangle becomes smaller.
    """
    current_pts = pts.copy()
    current_min_area, worst_idx = _worst_triangle(current_pts)
    
    for _ in range(max_iter):
        i, j, k = worst_idx
        p_i = current_pts[i]
        p_j = current_pts[j]
        p_k = current_pts[k]
        
        # Compute centroid of the worst triangle
        centroid = (p_i + p_j + p_k) / 3.0
        
        # Push each vertex away from centroid
        new_i = p_i + step * (p_i - centroid)
        new_j = p_j + step * (p_j - centroid)
        new_k = p_k + step * (p_k - centroid)
        
        # Try the update
        test_pts = current_pts.copy()
        test_pts[i] = new_i
        test_pts[j] = new_j
        test_pts[k] = new_k
        
        test_min, _ = _worst_triangle(test_pts)
        
        if test_min > current_min_area * (1 + 1e-12):
            current_pts = test_pts
            current_min_area = test_min
            worst_idx = _worst_triangle(current_pts)[1]
        else:
            # Try smaller step
            step *= 0.5
            if step < 1e-8:
                break
    
    return current_pts

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    best_score = 0.0
    best_pts = np.zeros((N, 2))
    rng = np.random.RandomState(42)
    
    # === PHASE 1: Concentric Ring Optimization (primary strategy) ===
    # The optimal Heilbronn configuration for small n is known to have
    # points on concentric circles with rotational symmetry.
    
    # Concentric 6+6+1 parameterizations
    concentric_inits = []
    # [cx, cy, r_outer, th_outer, r_inner, th_inner]
    for r_out in [1.0, 1.1, 1.2]:
        for r_in in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            for th_off in [0.0, np.pi/6.0, np.pi/3.0]:
                concentric_inits.append([0.0, 0.0, r_out, 0.0, r_in, th_off])
    
    # General 4+4+4+1 parameterizations
    general_inits = []
    # [cx, cy, r1, th1, r2, th2, r3, th3]
    for r1 in [0.4, 0.5, 0.6]:
        for r2 in [0.7, 0.8, 0.9]:
            for r3 in [1.0, 1.1, 1.2]:
                for th_off in [0.0, np.pi/4.0]:
                    general_inits.append([0.0, 0.0, r1, 0.0, r2, th_off, r3, 2*th_off])
    
    # Evaluate concentric initializations
    for params in concentric_inits:
        if time.time() > deadline - 60.0:
            break
        pts = _make_concentric_points(params)
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()
    
    # Evaluate general initializations
    for params in general_inits:
        if time.time() > deadline - 50.0:
            break
        pts = _make_general_concentric(params)
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()
    
    # === PHASE 2: Smooth optimization in concentric parameter space ===
    # This is a 6D problem vs 26D, much easier to optimize
    best_conc_params = None
    best_conc_score = 0.0
    
    for params in concentric_inits:
        if time.time() > deadline - 40.0:
            break
        p0 = np.array(params)
        try:
            res = minimize(
                lambda p: _neg_soft_score_concentric(p, alpha=80.0),
                p0,
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5), (-0.5, 0.5), (0.1, 3.0), (-np.pi, np.pi), (0.05, 2.5), (-np.pi, np.pi)],
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            pts = _make_concentric_points(res.x)
            s = _score(pts)
            if s > best_conc_score:
                best_conc_score = s
                best_conc_params = res.x.copy()
                if s > best_score:
                    best_score = s
                    best_pts = pts.copy()
        except Exception:
            pass
    
    # Optimize general concentric
    for params in general_inits:
        if time.time() > deadline - 30.0:
            break
        p0 = np.array(params)
        try:
            res = minimize(
                lambda p: _neg_soft_score_general(p, alpha=80.0),
                p0,
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5), (-0.5, 0.5), 
                        (0.1, 2.0), (-np.pi, np.pi),
                        (0.2, 2.5), (-np.pi, np.pi),
                        (0.5, 3.0), (-np.pi, np.pi)],
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            pts = _make_general_concentric(res.x)
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts.copy()
        except Exception:
            pass
    
    # === PHASE 3: Exact refinement in concentric space ===
    if best_conc_params is not None and time.time() < deadline - 15.0:
        try:
            res = minimize(
                _neg_score_concentric,
                best_conc_params,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            pts = _make_concentric_points(res.x)
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts.copy()
        except Exception:
            pass
    
    # === PHASE 4: Targeted worst-triangle optimization ===
    if time.time() < deadline - 10.0:
        # Apply targeted optimization to best found so far
        improved = _targeted_worst_triangle_opt(best_pts, max_iter=30, step=0.02)
        s = _score(improved)
        if s > best_score:
            best_score = s
            best_pts = improved.copy()
        
        # Also try from concentric-optimal points
        if best_conc_params is not None:
            conc_pts = _make_concentric_points(best_conc_params)
            improved2 = _targeted_worst_triangle_opt(conc_pts, max_iter=30, step=0.02)
            s2 = _score(improved2)
            if s2 > best_score:
                best_score = s2
                best_pts = improved2.copy()
    
    # === PHASE 5: General free-space optimization from best ===
    if time.time() < deadline - 8.0:
        try:
            res = minimize(
                lambda p: _neg_soft_score(p, alpha=100.0),
                best_pts.flatten(),
                method='Powell',
                options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            pts = res.x.reshape(-1, 2)
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts.copy()
        except Exception:
            pass
    
    # === PHASE 6: Final Nelder-Mead polish ===
    if time.time() < deadline - 3.0:
        try:
            res = minimize(
                _neg_score,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            pts = res.x.reshape(-1, 2)
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts.copy()
        except Exception:
            pass
    
    # === PHASE 7: Final targeted optimization ===
    if time.time() < deadline - 1.0:
        improved = _targeted_worst_triangle_opt(best_pts, max_iter=20, step=0.005)
        s = _score(improved)
        if s > best_score:
            best_score = s
            best_pts = improved.copy()
    
    return best_pts
# EVOLVE-BLOCK-END
