import numpy as np
from scipy.optimize import minimize
import time
import itertools
import scipy.spatial as spat

N = 13
DEADLINE = 140.0  # Safety margin for 150s limit

# Precompute triangle indices for vectorized area calculation
TRI_INDICES = np.array(list(itertools.combinations(range(N), 3)))

def _compute_triangle_areas(pts):
    """
    Vectorized computation of areas for all triples of points.
    """
    p1 = pts[TRI_INDICES[:, 0]]
    p2 = pts[TRI_INDICES[:, 1]]
    p3 = pts[TRI_INDICES[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def _hull_area(points):
    """Calculate convex hull area."""
    try:
        hull = spat.ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score_hard(pts):
    """
    Exact score calculation for final evaluation.
    """
    try:
        hull_area = _hull_area(pts)
        if hull_area < 1e-12:
            return 0.0
    except Exception:
        return 0.0
    areas = _compute_triangle_areas(pts)
    min_area = np.min(areas)
    return min_area / hull_area

def _score_soft(pts, beta):
    """
    Score using soft-min area and convex hull area.
    """
    try:
        hull_area = _hull_area(pts)
        if hull_area < 1e-12:
            return 0.0
    except Exception:
        return 0.0
    
    areas = _compute_triangle_areas(pts)
    
    # Soft-min computation
    # We want sum(exp(-beta * areas)).
    # Let A_min = min(areas).
    # sum(exp(-beta * A)) = exp(-beta * A_min) * sum(exp(-beta * (A - A_min)))
    # log(sum(...)) = -beta * A_min + log(sum(exp(-beta * (A - A_min))))
    # soft_min = -1/beta * log(sum(...)) = A_min - 1/beta * log(sum(exp(-beta * (A - A_min))))
    
    a_min = np.min(areas)
    shifted = -beta * (areas - a_min)
    # max shifted is 0, so exp(shifted) <= 1. No overflow.
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min = a_min - log_sum / beta
    
    return soft_min / hull_area

def _neg_score_soft(params, beta):
    pts = params.reshape(N, 2)
    return -_score_soft(pts, beta)

def _generate_initial_configurations():
    """
    Generate several symmetric and random initial configurations.
    """
    configs = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    configs.append(pts)
    
    # 2. 1 center + 12 ring
    pts = np.zeros((N, 2))
    pts[0] = [0, 0]
    angles = np.linspace(0, 2*np.pi, 12, endpoint=False)
    pts[1:] = np.column_stack([np.cos(angles), np.sin(angles)])
    configs.append(pts)
    
    # 3. 1 center + 3 inner + 4 mid + 5 outer
    pts = np.zeros((N, 2))
    idx = 0
    pts[idx] = [0, 0]; idx += 1
    for r, count in [(0.5, 3), (1.0, 4), (1.5, 5)]:
        angles = np.linspace(0, 2*np.pi, count, endpoint=False)
        offset = 0.5 * 2*np.pi / count if count > 2 else 0
        pts[idx:idx+count] = np.column_stack([r * np.cos(angles + offset), r * np.sin(angles + offset)])
        idx += count
    configs.append(pts)
    
    # 4. 1 center + 4 inner + 4 mid + 4 outer
    pts = np.zeros((N, 2))
    idx = 0
    pts[idx] = [0, 0]; idx += 1
    for r, count in [(0.5, 4), (1.0, 4), (1.5, 4)]:
        angles = np.linspace(0, 2*np.pi, count, endpoint=False)
        offset = 0.5 * 2*np.pi / count if count > 2 else 0
        pts[idx:idx+count] = np.column_stack([r * np.cos(angles + offset), r * np.sin(angles + offset)])
        idx += count
    configs.append(pts)
    
    # Add random inits
    rng = np.random.RandomState(42)
    for _ in range(10):
        pts = rng.uniform(-1, 1, (N, 2))
        configs.append(pts)
        
    # Points on circle with random radii
    for _ in range(5):
        angles = 2 * np.pi * np.arange(N) / N + rng.uniform(-0.1, 0.1, N)
        radii = 1.0 + rng.uniform(-0.2, 0.2, N)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        configs.append(pts)

    return configs

def construct():
    start_time = time.time()
    
    configs = _generate_initial_configurations()
    
    best_score = -np.inf
    best_points = None
    
    candidates = []
    
    # Phase 1: Coarse optimization with smaller beta (smoother)
    # Using Powell's method for robustness, as suggested by inspiration 1
    for init_pts in configs:
        if time.time() - start_time > DEADLINE - 30:
            break
            
        beta = 50.0
        try:
            res = minimize(
                lambda x: _neg_score_soft(x, beta),
                init_pts.flatten(),
                method='Powell',
                options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6, 'disp': False}
            )
            
            cand_pts = res.x.reshape(N, 2)
            hard_score = _score_hard(cand_pts)
            
            if hard_score > 0:
                candidates.append((hard_score, cand_pts))
        except Exception:
            continue
    
    # Sort candidates by score
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Phase 2: Refine the top candidates with higher beta (sharper minimum)
    # Use L-BFGS-B for faster convergence on smooth landscapes
    top_k = min(5, len(candidates))
    for i in range(top_k):
        if time.time() - start_time > DEADLINE - 10:
            break
        score, pts = candidates[i]
        
        # Anneal beta from 200 to 1000
        betas = [200.0, 500.0, 1000.0]
        current_pts = pts.flatten()
        
        for beta in betas:
            if time.time() - start_time > DEADLINE - 5:
                break
            try:
                res = minimize(
                    lambda x: _neg_score_soft(x, beta),
                    current_pts,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                current_pts = res.x
                cand_pts = res.x.reshape(N, 2)
                hard_score = _score_hard(cand_pts)
                
                if hard_score > best_score:
                    best_score = hard_score
                    best_points = cand_pts.copy()
            except Exception:
                continue

    # If we have time left, do a final polish on the best point found
    if best_points is not None and time.time() - start_time < DEADLINE - 5:
        try:
            res = minimize(
                lambda x: _neg_score_soft(x, 2000.0),
                best_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-10}
            )
            candidate_pts = res.x.reshape(N, 2)
            hard_score = _score_hard(candidate_pts)
            if hard_score > best_score:
                best_points = candidate_pts.copy()
        except Exception:
            pass

    if best_points is None:
        # Fallback to best candidate or regular polygon
        if candidates:
            best_points = candidates[0][1]
        else:
            angles = np.linspace(0, 2*np.pi, N, endpoint=False)
            best_points = np.column_stack([np.cos(angles), np.sin(angles)])
            
    return best_points
