import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

# Precompute all triangle index triplets for vectorized area computation
_TRIPLETS = np.array(list(combinations(range(N), 3)))
_N_TRIPLES = _TRIPLETS.shape[0]
_I, _J, _K = _TRIPLETS.T

def _triangle_areas(pts):
    """Compute all C(n,3) triangle areas vectorized."""
    a = pts[_I]
    b = pts[_J]
    c = pts[_K]
    areas = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) -
                          (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))
    return areas

def _convex_hull_area(pts):
    """Compute area of convex hull."""
    try:
        if len(pts) < 3:
            return 0.0
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score(pts):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(pts)
    if hull_area <= 0:
        return 0.0
    areas = _triangle_areas(pts)
    if len(areas) == 0:
        return 0.0
    return float(np.min(areas)) / hull_area

def _score_fast(pts):
    """Fast score computation for internal use (no exception handling overhead)."""
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
        if hull_area <= 1e-10:
            return 0.0
        areas = _triangle_areas(pts)
        if len(areas) == 0:
            return 0.0
        return float(np.min(areas)) / hull_area
    except Exception:
        return 0.0

def _soft_min_objective(x, beta=50.0):
    """
    Soft-min of triangle areas / hull area, negative for minimization.
    x is a flat array of length 2*N.
    """
    pts = x.reshape(-1, 2)
    hull_area = _convex_hull_area(pts)
    if hull_area <= 1e-10:
        return 1e6
    
    areas = _triangle_areas(pts)
    if len(areas) == 0:
        return 1e6
    
    # Soft min using log-sum-exp trick for numerical stability
    # soft_min(x) = -(1/beta) * log(sum(exp(-beta * x_i)))
    neg_beta_areas = -beta * areas
    max_val = np.max(neg_beta_areas)
    log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
    soft_min_area = -log_sum_exp / beta
    
    return -soft_min_area / hull_area

def _exact_objective(x):
    """Exact negative score for Nelder-Mead refinement."""
    pts = x.reshape(-1, 2)
    return -_score(pts)

def _regular_polygon(n, radius=1.0):
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _center_ring(n_outer, n_inner, r_outer=1.0, r_inner=0.5):
    """Points on two concentric rings."""
    angles_outer = np.linspace(0, 2*np.pi, n_outer, endpoint=False)
    angles_inner = np.linspace(0, 2*np.pi, n_inner, endpoint=False) + np.pi/n_inner
    outer = np.column_stack([r_outer * np.cos(angles_outer), r_outer * np.sin(angles_outer)])
    inner = np.column_stack([r_inner * np.cos(angles_inner), r_inner * np.sin(angles_inner)])
    return np.vstack([outer, inner])

def _halton_sequence(n, base):
    """Generate first n Halton sequence values in [0,1)."""
    seq = np.zeros(n)
    f = 1.0
    for i in range(1, n+1):
        f /= base
        seq[i-1] = 0.0
        j = i
        while j > 0:
            seq[i-1] += (j % base) * f
            j //= base
    return seq

def _halton_points(n, base1=2, base2=3):
    """Generate n points using 2D Halton sequence."""
    x = _halton_sequence(n, base1)
    y = _halton_sequence(n, base2)
    return np.column_stack([x, y])

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s limit
    
    best_score = 0.0
    best_points = None
    
    def update_best(pts, score):
        nonlocal best_score, best_points
        if score > best_score:
            best_score = score
            best_points = pts.copy()
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Regular 13-gon
    candidates.append(_regular_polygon(N, 1.0))
    
    # 2. Concentric rings (various splits)
    for n_outer in [7, 8, 9, 10]:
        n_inner = N - n_outer
        if n_inner >= 3:
            for r_inner in [0.4, 0.5, 0.6, 0.7]:
                candidates.append(_center_ring(n_outer, n_inner, 1.0, r_inner))
    
    # 3. Halton sequence points (low-discrepancy)
    halton = _halton_points(N)
    candidates.append(halton)
    candidates.append(halton * 2)
    # Also try scaled to unit disk
    radii = np.sqrt(halton[:, 0]**2 + halton[:, 1]**2)
    radii[radii < 1e-10] = 1.0
    candidates.append(halton / radii[:, np.newaxis])
    
    # 4. Random points on circle (sorted by angle for convex position)
    rng = np.random.default_rng(42)
    for i in range(15):
        angles = np.sort(rng.uniform(0, 2*np.pi, N))
        radii = rng.uniform(0.7, 1.0, N)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        candidates.append(pts)
    
    # 5. Random points in disk
    for i in range(10):
        angles = rng.uniform(0, 2*np.pi, N)
        radii = rng.uniform(0, 1, N)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        candidates.append(pts)
    
    # 6. Perturbed regular polygon
    for i in range(10):
        noise = rng.normal(0, 0.05, (N, 2))
        candidates.append(_regular_polygon(N, 1.0) + noise)
    
    # Evaluate all candidates with exact score (single pass)
    scored_candidates = []
    for pts in candidates:
        if time.time() > deadline:
            break
        s = _score_fast(pts)
        scored_candidates.append((s, pts))
        if s > best_score:
            best_score = s
            best_points = pts.copy()
    
    # Sort candidates by score, optimize top ones
    scored_candidates.sort(key=lambda x: -x[0])
    
    # Use top candidates as starting points for optimization
    top_candidates = [pts for _, pts in scored_candidates[:min(8, len(scored_candidates))]]
    
    # Progressive beta: start smooth, get sharper
    betas = [30.0, 80.0, 150.0, 300.0]
    
    for beta in betas:
        if time.time() > deadline - 15:
            break
        
        for init_pts in top_candidates:
            if time.time() > deadline - 10:
                break
            
            try:
                res = minimize(
                    _soft_min_objective,
                    init_pts.flatten(),
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 800, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
                )
                
                res_pts = res.x.reshape(N, 2)
                res_score = _score_fast(res_pts)
                update_best(res_pts, res_score)
                
            except Exception:
                pass
        
        # Update top candidates with current best for next beta
        if best_points is not None:
            top_candidates = [best_points.copy()]
    
    # Final refinement: use exact score with Nelder-Mead on the best point
    if best_points is not None and time.time() < deadline - 3:
        try:
            res = minimize(
                _exact_objective,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 5000}
            )
            res_pts = res.x.reshape(N, 2)
            res_score = _score_fast(res_pts)
            update_best(res_pts, res_score)
        except Exception:
            pass
    
    # Fallback
    if best_points is None:
        best_points = _regular_polygon(N, 1.0)
        best_score = _score_fast(best_points)
    
    return best_points
