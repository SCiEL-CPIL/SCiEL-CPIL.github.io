# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _normalize(pts):
    centroid = pts.mean(axis=0)
    p = pts - centroid
    d = np.linalg.norm(p, axis=1)
    mean_d = d.mean()
    if mean_d <= 0.01:
        return None, None, None
    p = p / mean_d
    return p, d, mean_d

def _check_strictly_convex(p):
    n = len(p)
    if n < 3:
        return False
    # no repeated points
    for i in range(n):
        if np.linalg.norm(p[i] - p[(i + 1) % n]) < 1e-9:
            return False
    # compute cross products of consecutive edges
    signs = []
    for i in range(n):
        p0 = p[i]
        p1 = p[(i + 1) % n]
        p2 = p[(i + 2) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        if abs(cross) < 1e-9:
            return False
        signs.append(1 if cross > 0 else -1)
    # all same sign
    if signs.count(signs[0]) != n:
        return False
    # total turning = +-2*pi
    angles = []
    for i in range(n):
        p0 = p[i]
        p1 = p[(i + 1) % n]
        p2 = p[(i + 2) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_a = np.clip(cos_a, -1, 1)
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        sin_a = cross / (np.linalg.norm(v1) * np.linalg.norm(v2))
        ang = np.arctan2(sin_a, cos_a)
        angles.append(ang)
    total_turn = sum(angles)
    expected = 2 * np.pi if signs[0] > 0 else -2 * np.pi
    if abs(abs(total_turn) - 2 * np.pi) > 1e-4:
        return False
    return True

def _vertex_score(p, v_idx):
    n = len(p)
    v = p[v_idx]
    # distances to all other vertices
    d = np.array([np.linalg.norm(v - p[j]) for j in range(n) if j != v_idx])
    d_sorted = np.sort(d)
    if np.any(d_sorted < 0.05):
        return 0.0
    # find k: first index minimizing d_{k+1} - d_k for k in 0..17
    gaps = d_sorted[1:] - d_sorted[:-1]
    k = int(np.argmin(gaps))  # 0..17
    D = (d_sorted[k] + d_sorted[k + 1]) / 2
    if D < 0.1:
        return 0.0
    # select four distances
    if k == 0:
        selected = d_sorted[0:4]
    elif k == 17:
        selected = d_sorted[15:19]
    else:
        selected = d_sorted[k - 1:k + 3]
    if len(selected) < 4:
        return 0.0
    raw = np.mean([max(di / D, D / di) for di in selected])
    if raw < 1e-12:
        return 0.0
    score = min(1.0, 1.0 / raw)
    return score

def _score_polygon(p):
    """Score a polygon. Returns min vertex score (0 if invalid)."""
    pn, _, _ = _normalize(p)
    if pn is None:
        return 0.0
    if not _check_strictly_convex(pn):
        return 0.0
    min_vs = 1.0
    for i in range(len(pn)):
        vs = _vertex_score(pn, i)
        if vs < min_vs:
            min_vs = vs
        if min_vs < 0.01:  # early exit
            break
    return min_vs

def _coeffs_to_points(coeffs):
    """
    Convert real coefficients c_1..c_20 to 20 points in the complex plane.
    We construct P(z) = z^20 - c_1 z^19 - ... - c_20.
    The roots are the vertices.
    To ensure they are on the unit circle (or close), we could map them, 
    but actually the task allows any convex polygon, not necessarily on a circle.
    However, the "equidistant" hint in the function name and the metric structure
    (distance gaps) suggests a regular polygon is the ideal.
    The regular 20-gon has roots of z^20 - 1 = 0, so c_1=0, c_2=0, ..., c_19=0, c_20=1.
    Wait, P(z) = z^20 - 1. Roots are exp(i * 2*pi*k/20).
    So the regular polygon corresponds to coeffs = [0, 0, ..., 0, 1].
    
    However, if we just use np.roots, the roots might not be in cyclic order.
    We need to sort them by angle to ensure cyclic order for the convexity check.
    """
    # coeffs are c_1..c_20
    # Polynomial: z^20 - c_1 z^19 - ... - c_20
    # np.roots expects [a_n, a_{n-1}, ..., a_0]
    # So [1, -c_1, -c_2, ..., -c_20]
    poly_coeffs = np.concatenate(([1.0], -np.array(coeffs)))
    try:
        roots = np.roots(poly_coeffs)
    except Exception:
        return None
    
    # Convert to 2D points
    pts = np.column_stack([roots.real, roots.imag])
    
    # Sort by angle to ensure cyclic order
    angles = np.arctan2(pts[:, 1], pts[:, 0])
    sorted_indices = np.argsort(angles)
    pts = pts[sorted_indices]
    
    return pts

def _smooth_score(coeffs):
    """
    Smoothed surrogate for the min-vertex-score.
    """
    pts = _coeffs_to_points(coeffs)
    if pts is None:
        return -100.0

    pn, _, _ = _normalize(pts)
    if pn is None:
        return -100.0
        
    # Check convexity quickly
    if not _check_strictly_convex(pn):
        return -100.0 # Heavy penalty

    # Calculate vertex scores
    scores = []
    for i in range(len(pn)):
        vs = _vertex_score(pn, i)
        scores.append(vs)
    
    scores = np.array(scores)
    
    # Use a soft-min to create a smooth landscape.
    tau = 0.05
    
    neg_scores_over_tau = -scores / tau
    max_neg = np.max(neg_scores_over_tau)
    log_sum_exp = max_neg + np.log(np.sum(np.exp(neg_scores_over_tau - max_neg)))
    soft_min = -tau * log_sum_exp
    
    return -soft_min # We are minimizing the negative of the soft-min score

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580.0  # 50s safety margin
    
    n = 20
    
    best_score = 0.0
    best_pts = None
    
    # Regular 20-gon coefficients: z^20 - 1 = 0
    # c_1=0, c_2=0, ..., c_19=0, c_20=1
    regular_coeffs = np.zeros(n)
    regular_coeffs[-1] = 1.0
    
    def try_candidate(coeffs):
        nonlocal best_score, best_pts
        pts = _coeffs_to_points(coeffs)
        if pts is None:
            return
        s = _score_polygon(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()

    # Initial seeds for coefficients
    seeds = []
    seeds.append(regular_coeffs.copy())
    
    # Add some perturbations to the coefficients
    # Small random perturbations
    rng = np.random.default_rng(42)
    for _ in range(10):
        pert = rng.normal(0, 0.05, n)
        seeds.append(regular_coeffs + pert)
    
    # Larger perturbations
    for _ in range(5):
        pert = rng.normal(0, 0.2, n)
        seeds.append(regular_coeffs + pert)

    # Run Nelder-Mead from multiple seeds
    max_iter_per_seed = 1000
    tol = 1e-6

    for i, seed in enumerate(seeds):
        if time.time() > deadline:
            break
            
        # Update best with seed
        try_candidate(seed)
        
        # Run optimization
        try:
            res = minimize(
                _smooth_score, 
                seed, 
                method='Nelder-Mead',
                options={'maxiter': max_iter_per_seed, 'xatol': tol, 'fatol': tol}
            )
            
            # Evaluate the result
            try_candidate(res.x)
                
        except Exception:
            pass
            
    # If we didn't find anything good, fall back to regular polygon
    if best_pts is None:
        best_pts = _coeffs_to_points(regular_coeffs)
    
    # Final polish: Do a few random restarts around the best found so far?
    # Or just a fine-grained local search on the best found?
    # Let's do a simple local search with small steps on the best coeffs.
    # We need to recover coeffs from best_pts? No, we stored best_pts.
    # But the search was on coeffs. We should store best_coeffs too.
    
    # Let's re-run the logic to keep track of best_coeffs
    best_score = 0.0
    best_pts = None
    best_coeffs = None

    def try_candidate_v2(coeffs):
        nonlocal best_score, best_pts, best_coeffs
        pts = _coeffs_to_points(coeffs)
        if pts is None:
            return
        s = _score_polygon(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()
            best_coeffs = coeffs.copy()

    for i, seed in enumerate(seeds):
        if time.time() > deadline:
            break
            
        try_candidate_v2(seed)
        
        try:
            res = minimize(
                _smooth_score, 
                seed, 
                method='Nelder-Mead',
                options={'maxiter': max_iter_per_seed, 'xatol': tol, 'fatol': tol}
            )
            try_candidate_v2(res.x)
        except Exception:
            pass

    if best_pts is None:
        best_pts = _coeffs_to_points(regular_coeffs)
    else:
        best_pts = _coeffs_to_points(best_coeffs)

    return best_pts

# EVOLVE-BLOCK-END
