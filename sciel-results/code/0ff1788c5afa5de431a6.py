import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by 3 points."""
    min_area = np.inf
    for i in range(N-2):
        for j in range(i+1, N-1):
            for k in range(j+1, N):
                p1 = pts[i]
                p2 = pts[j]
                p3 = pts[k]
                area = 0.5 * abs(p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1]))
                if area < min_area:
                    min_area = area
    return min_area

def _hull_area(pts):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 1e-10

def _score(pts):
    """Calculate the objective score: min_triangle_area / hull_area."""
    min_area = _min_triangle_area(pts)
    h_area = _hull_area(pts)
    if h_area < 1e-10:
        return 0.0
    return min_area / h_area

def _all_triangle_areas(pts):
    """Compute areas of all C(N,3) triangles as a 1D array."""
    areas = np.zeros(len(list(combinations(range(N), 3))))
    idx = 0
    for i in range(N-2):
        for j in range(i+1, N-1):
            for k in range(j+1, N):
                p1 = pts[i]
                p2 = pts[j]
                p3 = pts[k]
                areas[idx] = 0.5 * abs(p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1]))
                idx += 1
    return areas

def _soft_min_score(pts, beta=50.0):
    """
    Smooth approximation of min_triangle_area / hull_area using log-sum-exp.
    soft_min(x) = -(1/beta) * log(sum(exp(-beta * x_i)))
    This is differentiable and approximates min as beta -> inf.
    """
    areas = _all_triangle_areas(pts)
    # Avoid log(0) issues
    areas = np.maximum(areas, 1e-15)
    # Soft minimum: -(1/beta) * log(sum(exp(-beta * areas)))
    # For numerical stability, subtract max
    neg_beta_areas = -beta * areas
    max_val = np.max(neg_beta_areas)
    log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
    soft_min_area = -log_sum_exp / beta
    
    h_area = _hull_area(pts)
    if h_area < 1e-10:
        return 0.0
    return soft_min_area / h_area

def _neg_soft_min_score(x, beta=50.0):
    """Negative soft-min score for minimization."""
    pts = x.reshape(N, 2)
    return -_soft_min_score(pts, beta)

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
    deadline = start_time + 135  # 15s safety margin
    
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
            candidates.append(_center_ring(n_outer, n_inner, 1.0, 0.5))
            candidates.append(_center_ring(n_outer, n_inner, 1.0, 0.6))
            candidates.append(_center_ring(n_outer, n_inner, 1.0, 0.4))
    
    # 3. Halton sequence points (low-discrepancy)
    halton = _halton_points(N)
    candidates.append(halton)
    # Scaled Halton
    candidates.append(halton * 2)
    
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
    
    # Evaluate all candidates with exact score
    for pts in candidates:
        if time.time() > deadline:
            break
        s = _score(pts)
        update_best(pts, s)
    
    # Optimize from best candidates using L-BFGS-B with soft-min surrogate
    # Sort candidates by score, optimize top ones
    scored_candidates = []
    for pts in candidates:
        s = _score(pts)
        scored_candidates.append((s, pts))
    scored_candidates.sort(key=lambda x: -x[0])
    
    # Use top candidates as starting points for optimization
    top_candidates = [pts for _, pts in scored_candidates[:min(10, len(scored_candidates))]]
    
    # Progressive beta: start smooth, get sharper
    betas = [20.0, 50.0, 100.0, 200.0]
    
    for beta in betas:
        if time.time() > deadline:
            break
        
        for init_pts in top_candidates:
            if time.time() > deadline:
                break
            
            try:
                res = minimize(
                    _neg_soft_min_score,
                    init_pts.flatten(),
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
                )
                
                res_pts = res.x.reshape(N, 2)
                res_score = _score(res_pts)
                update_best(res_pts, res_score)
                
                # Also try the soft-min score as a secondary check
                soft_score = _soft_min_score(res_pts, beta)
                
            except Exception:
                pass
        
        # Update top candidates with current best
        top_candidates = [best_points.copy()] if best_points is not None else top_candidates
    
    # Final refinement: use exact score with Nelder-Mead on the best point
    if best_points is not None and time.time() < deadline - 10:
        try:
            res = minimize(
                lambda x: -_score(x.reshape(N, 2)),
                best_points.flatten(),
                method='Nelder-Mead',
                options={'xatol': 1e-6, 'fatol': 1e-8, 'maxiter': 3000}
            )
            res_pts = res.x.reshape(N, 2)
            res_score = _score(res_pts)
            update_best(res_pts, res_score)
        except Exception:
            pass
    
    # Fallback
    if best_points is None:
        best_points = _regular_polygon(N, 1.0)
        best_score = _score(best_points)
    
    return best_points

# EVOLVE-BLOCK-END
