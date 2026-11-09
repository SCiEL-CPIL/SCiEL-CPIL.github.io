# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution

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

def _generate_polygon_from_params(params):
    """
    params: array of length 40.
    First 20 are radii r_i.
    Next 20 are angle offsets delta_theta_i.
    Base angles are regular.
    """
    n = 20
    r = params[:n]
    d_theta = params[n:]
    base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    angles = base_angles + d_theta
    
    # Sort angles to ensure cyclic order? 
    # If d_theta is small, order is preserved. 
    # If not, we might need to sort, but that breaks the 1-to-1 mapping with r_i.
    # We assume small perturbations so order is preserved.
    
    x = r * np.cos(angles)
    y = r * np.sin(angles)
    return np.column_stack([x, y])

def _objective(params):
    try:
        pts = _generate_polygon_from_params(params)
        score = _score_polygon(pts)
        # We want to maximize score, so minimize negative score
        if score == 0.0:
            # Penalize invalid polygons heavily
            return 1000.0
        return -score
    except Exception:
        return 1000.0

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580.0
    
    n = 20
    # Bounds: radii [0.5, 1.5], angle offsets [-0.2, 0.2]
    bounds_r = [(0.5, 1.5)] * n
    bounds_dtheta = [(-0.2, 0.2)] * n
    bounds = bounds_r + bounds_dtheta
    
    best_score = 0.0
    best_params = np.concatenate([np.ones(n), np.zeros(n)])
    
    # Run Differential Evolution
    # Multiple runs with different seeds to increase robustness
    seeds = [1, 2, 3]
    
    for seed in seeds:
        if time.time() > deadline - 60: # Leave 60s for final processing
            break
            
        try:
            result = differential_evolution(
                _objective,
                bounds,
                seed=seed,
                maxiter=200,
                popsize=20,
                tol=1e-5,
                polish=True,
                init='latinhypercube',
                update_interval=1,
                callback=None
            )
            
            if result.success and result.fun < 1000.0: # 1000 is penalty
                current_score = -result.fun
                if current_score > best_score:
                    best_score = current_score
                    best_params = result.x
        except Exception:
            pass

    # Local refinement around the best found parameters using coordinate ascent or similar
    if time.time() < deadline - 10:
        current_params = best_params.copy()
        step_sizes = [0.05, 0.02, 0.01, 0.005]
        
        for step in step_sizes:
            if time.time() > deadline - 5:
                break
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                for i in range(40):
                    for delta in [step, -step]:
                        new_params = current_params.copy()
                        new_params[i] += delta
                        
                        # Check bounds
                        if i < n:
                            if new_params[i] < 0.5 or new_params[i] > 1.5:
                                continue
                        else:
                            if new_params[i] < -0.2 or new_params[i] > 0.2:
                                continue
                        
                        try:
                            pts = _generate_polygon_from_params(new_params)
                            score = _score_polygon(pts)
                            if score > best_score:
                                best_score = score
                                best_params = new_params
                                current_params = new_params
                                improved = True
                        except Exception:
                            pass

    # Generate final points
    pts = _generate_polygon_from_params(best_params)
    return pts

# EVOLVE-BLOCK-END
