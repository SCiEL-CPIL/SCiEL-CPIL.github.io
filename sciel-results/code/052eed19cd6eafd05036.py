import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
from itertools import combinations
import time

N = 13
# Pre-compute all triangle indices for speed
TRIS = list(combinations(range(N), 3))
NUM_TRIS = len(TRIS)

def _all_triangle_areas(pts):
    """Compute areas of all C(N,3) triangles as a 1D array."""
    areas = np.zeros(NUM_TRIS)
    for idx, (i, j, k) in enumerate(TRIS):
        p1 = pts[i]
        p2 = pts[j]
        p3 = pts[k]
        areas[idx] = 0.5 * abs(p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1]))
    return areas

def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by 3 points."""
    areas = _all_triangle_areas(pts)
    return np.min(areas)

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

def _soft_min_score(pts, beta=50.0):
    """
    Smooth approximation of min_triangle_area / hull_area using log-sum-exp.
    """
    areas = _all_triangle_areas(pts)
    # Avoid log(0) issues
    areas = np.maximum(areas, 1e-15)
    # Soft minimum: -(1/beta) * log(sum(exp(-beta * areas)))
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

def _fourier_to_points(coeffs):
    """
    Convert Fourier coefficients to points.
    coeffs[0] is the mean radius.
    coeffs[1:3] are (a1, b1) for k=1.
    coeffs[3:5] are (a2, b2) for k=2.
    coeffs[5:7] are (a3, b3) for k=3.
    coeffs[7:9] are (a4, b4) for k=4.
    Total 9 parameters.
    """
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    r = np.zeros(N)
    
    # r(theta) = c0 + sum(ak*cos(k*theta) + bk*sin(k*theta))
    c0 = coeffs[0]
    r += c0
    
    # k=1
    r += coeffs[1] * np.cos(angles) + coeffs[2] * np.sin(angles)
    # k=2
    r += coeffs[3] * np.cos(2 * angles) + coeffs[4] * np.sin(2 * angles)
    # k=3
    r += coeffs[5] * np.cos(3 * angles) + coeffs[6] * np.sin(3 * angles)
    # k=4
    r += coeffs[7] * np.cos(4 * angles) + coeffs[8] * np.sin(4 * angles)
    
    # Ensure convexity by enforcing r > 0 and checking if points are in convex position
    # A simpler check: if the area of the polygon formed by the points in angular order
    # is equal to the convex hull area, they are in convex position.
    # However, for optimization, we just need to avoid degenerate/non-convex shapes.
    # If r becomes negative, the point flips, causing non-convexity.
    # We will clip r to a small positive value to maintain continuity, though this might
    # create flat segments. Better to penalize non-convexity in the objective or just
    # let the optimizer avoid it if the initial guess is good.
    
    # To be safe, if any r <= 0, we return a penalized score (handled in wrapper)
    # or just clip. Let's clip to 0.01 to prevent collapse.
    r = np.maximum(r, 0.01)
    
    x = r * np.cos(angles)
    y = r * np.sin(angles)
    return np.column_stack((x, y))

def _is_convex(pts):
    """Check if points are in strictly convex position (no 3 collinear, hull size == N)."""
    try:
        hull = ConvexHull(pts)
        return len(hull.vertices) == N
    except Exception:
        return False

def _neg_fourier_objective(coeffs):
    """
    Objective for differential evolution on Fourier coefficients.
    Returns negative score. Penalizes non-convex configurations heavily.
    """
    pts = _fourier_to_points(coeffs)
    
    # Check convexity
    if not _is_convex(pts):
        return 100.0 # High penalty
    
    score = _score(pts)
    return -score

def _regular_polygon(n, radius=1.0):
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    best_score = 0.0
    best_points = None
    
    def update_best(pts, score):
        nonlocal best_score, best_points
        if score > best_score:
            best_score = score
            best_points = pts.copy()

    # 1. Initial Candidates
    candidates = []
    
    # Regular polygon
    candidates.append(_regular_polygon(N, 1.0))
    
    # Random perturbations of regular polygon
    rng = np.random.default_rng(42)
    for i in range(20):
        noise = rng.normal(0, 0.1, (N, 2))
        pts = _regular_polygon(N, 1.0) + noise
        candidates.append(pts)

    # Evaluate initial candidates
    for pts in candidates:
        if time.time() > deadline:
            break
        if _is_convex(pts):
            s = _score(pts)
            update_best(pts, s)
        else:
            # If not convex, project to convex hull? Or just skip.
            # For now, skip non-convex as they are invalid for the "convex region" constraint implied by the task 
            # (though the task says "Place 13 points in a planar convex region", it usually implies the points themselves
            # should form a convex set to maximize area, or at least the hull area is the denominator).
            # The task says "normalized by the point set convex hull area".
            # If points are not in convex position, some are inside.
            # The score is min_tri_area / hull_area.
            # It is generally better to have all points on the hull.
            pass

    if best_points is None:
        # Fallback
        best_points = _regular_polygon(N, 1.0)
        best_score = _score(best_points)

    # 2. Fourier-based Differential Evolution
    # Parameters: [c0, a1, b1, a2, b2, a3, b3, a4, b4]
    # Bounds:
    # c0: [0.5, 2.0] (Mean radius)
    # ak, bk: [-0.5, 0.5] (Amplitudes)
    bounds = [
        (0.5, 2.0),
        (-0.5, 0.5), (-0.5, 0.5),
        (-0.5, 0.5), (-0.5, 0.5),
        (-0.5, 0.5), (-0.5, 0.5),
        (-0.5, 0.5), (-0.5, 0.5)
    ]
    
    # Initialize DE with the regular polygon coefficients
    # Regular polygon: c0=1, all others 0
    x0 = [1.0, 0, 0, 0, 0, 0, 0, 0, 0]
    
    try:
        # Differential Evolution is global and robust
        res_de = differential_evolution(
            _neg_fourier_objective,
            bounds=bounds,
            x0=x0,
            seed=42,
            maxiter=100,
            popsize=15,
            tol=1e-4,
            polish=False, # We will polish manually
            update_interval=10
        )
        
        if time.time() < deadline - 10:
            best_coeffs = res_de.x
            pts_de = _fourier_to_points(best_coeffs)
            if _is_convex(pts_de):
                s_de = _score(pts_de)
                update_best(pts_de, s_de)
                
    except Exception as e:
        pass

    # 3. Local Refinement with L-BFGS-B on Soft-Min Surrogate
    # Use the best points from DE (or initial) as starting point
    if best_points is not None:
        # Try a few different betas for progressive sharpening
        betas = [20.0, 50.0, 100.0, 200.0]
        current_best = best_points.copy()
        
        for beta in betas:
            if time.time() > deadline - 5:
                break
            
            try:
                res_lbfgs = minimize(
                    _neg_soft_min_score,
                    current_best.flatten(),
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                res_pts = res_lbfgs.x.reshape(N, 2)
                if _is_convex(res_pts):
                    s_res = _score(res_pts)
                    update_best(res_pts, s_res)
                    if s_res > _score(current_best):
                        current_best = res_pts.copy()
                        
            except Exception:
                pass

    # 4. Final Nelder-Mead Polish on Exact Score
    if best_points is not None and time.time() < deadline - 2:
        try:
            def neg_exact(x):
                pts = x.reshape(N, 2)
                if not _is_convex(pts):
                    return 100.0
                return -_score(pts)
            
            res_nm = minimize(
                neg_exact,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 1000}
            )
            
            res_pts = res_nm.x.reshape(N, 2)
            if _is_convex(res_pts):
                s_nm = _score(res_pts)
                update_best(res_pts, s_nm)
                
        except Exception:
            pass

    return best_points
