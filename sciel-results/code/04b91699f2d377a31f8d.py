import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

N = 13

# Pre-compute triangle indices for N=13
_TRI_INDICES = np.array(list(combinations(range(N), 3)))
_I_IDX = _TRI_INDICES[:, 0]
_J_IDX = _TRI_INDICES[:, 1]
_K_IDX = _TRI_INDICES[:, 2]
_NUM_TRI = len(_TRI_INDICES)

def _get_triangle_areas(points):
    """
    Vectorized calculation of areas for all triples.
    points: (N, 2) array
    returns: array of shape (num_triangles,)
    """
    pi = points[_I_IDX]
    pj = points[_J_IDX]
    pk = points[_K_IDX]
    
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return areas

def _hull_area(points):
    """Calculate convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Calculate the raw score: min_triangle_area / hull_area."""
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    areas = _get_triangle_areas(points)
    if len(areas) == 0:
        return 0.0
    min_area = np.min(areas)
    return min_area / hull_area

def _softmin_score(points, alpha):
    """
    Smooth approximation of the score using softmin of triangle areas.
    """
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    
    areas = _get_triangle_areas(points)
    if len(areas) == 0:
        return 0.0
    
    # Softmin: -1/alpha * log(sum(exp(-alpha * areas)))
    a_min = np.min(areas)
    exp_terms = np.exp(-alpha * (areas - a_min))
    log_sum = np.log(np.sum(exp_terms))
    softmin_val = a_min - np.log(np.sum(exp_terms)) / alpha
    
    return softmin_val / hull_area

def _neg_softmin_score(params, alpha):
    """Objective for softmin-based optimization."""
    points = params.reshape(N, 2)
    s = _softmin_score(points, alpha)
    return -s

def _neg_score(params):
    """Objective function for minimization (maximizing score)."""
    points = params.reshape(N, 2)
    s = _score(points)
    return -s

def _fourier_params_to_points(params):
    """
    Convert Fourier parameters to points.
    params: [r0, c1, s1, c2, s2, ..., cK, sK]
    We use a base radius r0 and small perturbations.
    To ensure convexity and reasonable shape, we limit the perturbation amplitude.
    """
    r0 = params[0]
    # Number of harmonics to use. Let's use up to 3 or 4 for flexibility.
    # params[1:] are coefficients for cos(k*theta) and sin(k*theta)
    # Let's say we support K harmonics. 
    # For N=13, frequencies 1, 2, 3 are enough to capture asymmetry.
    K = 3 
    # Total params expected: 1 + 2*K = 7. If more are passed, ignore or error.
    # We will design the bounds to match this.
    
    theta = 2 * np.pi * np.arange(N) / N
    radii = np.ones(N) * r0
    
    for k in range(1, K+1):
        ck = params[1 + 2*(k-1)]
        sk = params[2 + 2*(k-1)]
        radii += ck * np.cos(k * theta) + sk * np.sin(k * theta)
        
    # Ensure radii are positive (convexity towards center)
    # If radii become <= 0, the shape is non-convex or self-intersecting, 
    # but ConvexHull handles it. However, for "regular-like" shapes, we want positive radii.
    # We can clamp or just let the optimizer avoid it. 
    # Given the bounds, r0 ~ 1 and coeffs ~ 0.1, radii should stay positive.
    
    points = np.column_stack([radii * np.cos(theta), radii * np.sin(theta)])
    return points

def _neg_fourier_score(params):
    """Objective for Fourier-based optimization."""
    points = _fourier_params_to_points(params)
    s = _score(points)
    return -s

def _polish_with_local_search(points, deadline, rng):
    """
    Refine a given point set using multi-stage local search.
    1. Nelder-Mead on softmin (smooth)
    2. Nelder-Mead on hard score
    3. Coordinate search on hard score
    """
    best_pts = points.copy()
    best_score = _score(best_pts)
    
    # Stage 1: Softmin polish to smooth out the landscape
    if time.time() < deadline - 10:
        try:
            # Use a moderate alpha for smoothing
            alpha = 20.0
            res = minimize(
                _neg_softmin_score,
                best_pts.flatten(),
                args=(alpha,),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            cand_pts = res.x.reshape(N, 2)
            score = _score(cand_pts)
            if score > best_score:
                best_score = score
                best_pts = cand_pts
        except Exception:
            pass

    # Stage 2: Hard score Nelder-Mead
    if time.time() < deadline - 5:
        try:
            res = minimize(
                _neg_score,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            cand_pts = res.x.reshape(N, 2)
            score = _score(cand_pts)
            if score > best_score:
                best_score = score
                best_pts = cand_pts
        except Exception:
            pass
            
    # Stage 3: Coordinate search for fine tuning
    if time.time() < deadline - 1:
        try:
            # Simple coordinate search
            step = 0.01
            for _ in range(50):
                if time.time() > deadline - 1:
                    break
                improved = False
                for i in range(N):
                    if time.time() > deadline - 1:
                        break
                    for dim in range(2):
                        if time.time() > deadline - 1:
                            break
                        for direction in [1, -1]:
                            trial = best_pts.copy()
                            trial[i, dim] += direction * step
                            trial_score = _score(trial)
                            if trial_score > best_score:
                                best_pts = trial
                                best_score = trial_score
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if not improved:
                    step *= 0.5
                    if step < 1e-8:
                        break
        except Exception:
            pass

    return best_pts, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s kill
    
    best_score = 0.0
    best_points = None
    rng = np.random.RandomState(42)
    
    # 1. Global Search using Differential Evolution on Fourier Parameters
    # This exploits the symmetry and structure of the problem.
    # Parameters: [r0, c1, s1, c2, s2, c3, s3]
    # r0 ~ 1.0, coeffs small.
    bounds = [(0.5, 1.5), 
              (-0.2, 0.2), (-0.2, 0.2), 
              (-0.2, 0.2), (-0.2, 0.2), 
              (-0.2, 0.2), (-0.2, 0.2)]
    
    # DE is stochastic and global.
    # We give it a generous budget.
    if time.time() < deadline - 60:
        try:
            res = differential_evolution(
                _neg_fourier_score,
                bounds=bounds,
                seed=42,
                maxiter=50,
                popsize=20,
                tol=1e-5,
                polish=False, # We will do our own polish
                update_interval=1
            )
            best_params = res.x
            best_pts = _fourier_params_to_points(best_params)
            best_score = _score(best_pts)
            best_points = best_pts
        except Exception:
            pass
    
    # 2. If DE failed or score is low, try a few manual Fourier starts
    if best_score < 50.0:
        # Regular polygon
        angles = 2 * np.pi * np.arange(N) / N
        pts = np.column_stack([np.cos(angles), np.sin(angles)])
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_points = pts
            
        # Slightly perturbed
        for _ in range(5):
            params = np.array([1.0] + rng.uniform(-0.1, 0.1, 6))
            pts = _fourier_params_to_points(params)
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_points = pts

    # 3. Local Polish from the best candidate
    if best_points is not None and time.time() < deadline - 10:
        cand_pts, score = _polish_with_local_search(best_points, deadline, rng)
        if score > best_score:
            best_score = score
            best_points = cand_pts
            
    # 4. Try a few more diverse starts if time permits
    # Restart from random points near the best solution
    if best_points is not None and time.time() < deadline - 20:
        for _ in range(3):
            if time.time() > deadline - 10:
                break
            noise_scale = 0.05
            perturbed = best_points + noise_scale * rng.normal(0, 1, (N, 2))
            cand_pts, score = _polish_with_local_search(perturbed, deadline, rng)
            if score > best_score:
                best_score = score
                best_points = cand_pts
                
    # 5. Final ultra-fine polish
    if best_points is not None and time.time() < deadline - 2:
        try:
            res = minimize(
                _neg_score,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-15, 'fatol': 1e-15, 'adaptive': True}
            )
            cand_pts = res.x.reshape(N, 2)
            score = _score(cand_pts)
            if score > best_score:
                best_score = score
                best_points = cand_pts
        except Exception:
            pass

    # Fallback
    if best_points is None:
        angles = 2 * np.pi * np.arange(N) / N
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return best_points
