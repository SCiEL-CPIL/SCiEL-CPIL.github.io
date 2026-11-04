"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, minimize_scalar, differential_evolution

# Pre-compute indices for N=14 for fast triangle area calculation
_IDX_CACHE = {}
def _get_indices(n):
    if n not in _IDX_CACHE:
        i_idx, j_idx, k_idx = [], [], []
        for i in range(n-2):
            for j in range(i+1, n-1):
                for k in range(j+1, n):
                    i_idx.append(i)
                    j_idx.append(j)
                    k_idx.append(k)
        _IDX_CACHE[n] = (np.array(i_idx), np.array(j_idx), np.array(k_idx))
    return _IDX_CACHE[n]

def _min_triangle_area(pts):
    n = len(pts)
    if n < 3:
        return 0.0
    i_idx, j_idx, k_idx = _get_indices(n)
    
    pi = pts[i_idx]
    pj = pts[j_idx]
    pk = pts[k_idx]
    
    v1 = pj - pi
    v2 = pk - pi
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _convex_hull_area(pts):
    try:
        # Check for degenerate hull
        if np.allclose(pts[:, 0], pts[0, 0]) or np.allclose(pts[:, 1], pts[0, 1]):
            return 1e-10
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 1e-10

def _score(pts):
    hull_area = _convex_hull_area(pts)
    if hull_area < 1e-9:
        return 0.0
    min_area = _min_triangle_area(pts)
    return min_area / hull_area

def _soft_min_area(pts, beta):
    n = len(pts)
    if n < 3:
        return 0.0
    i_idx, j_idx, k_idx = _get_indices(n)
    
    pi = pts[i_idx]
    pj = pts[j_idx]
    pk = pts[k_idx]
    
    v1 = pj - pi
    v2 = pk - pi
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_a = np.min(areas)
    shifted = areas - min_a
    # Avoid overflow in exp
    max_shift = np.max(shifted)
    if max_shift > 50:
        # If beta is very large, just return min_a
        return min_a
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    return min_a - (1.0 / beta) * log_sum

def _soft_score(pts, beta):
    hull_area = _convex_hull_area(pts)
    if hull_area < 1e-9:
        return 0.0
    soft_min = _soft_min_area(pts, beta)
    return soft_min / hull_area

def _two_ring_points(r, alpha):
    """
    Generate points for a two-ring configuration.
    Outer ring: 7 points, radius 1.0, angles 0, 2pi/7, ...
    Inner ring: 7 points, radius r, angles alpha, alpha + 2pi/7, ...
    """
    angles_outer = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    angles_inner = angles_outer + alpha
    
    outer_pts = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    inner_pts = np.column_stack([r * np.cos(angles_inner), r * np.sin(angles_inner)])
    
    return np.vstack([outer_pts, inner_pts])

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    best_score = -np.inf
    best_pts = None
    
    def update_best(pts):
        nonlocal best_score, best_pts
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()
            return True
        return False

    # 1. Baseline: Regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    reg_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    update_best(reg_pts)

    # 2. Two-Ring Optimization
    # We optimize r (inner radius) and alpha (angular offset)
    # Initial guess: r=0.5, alpha=pi/7 (staggered)
    
    def neg_two_ring_score(params):
        r, alpha = params
        pts = _two_ring_points(r, alpha)
        return -_score(pts)

    if time.time() < deadline - 60.0:
        try:
            # Use differential evolution for the 2D parameter space
            # This is fast and robust for 2 parameters
            bounds = [(0.2, 0.9), (0.0, 2 * np.pi / 7)]
            res = differential_evolution(
                neg_two_ring_score, 
                bounds, 
                seed=42, 
                maxiter=100, 
                tol=1e-8,
                polish=True
            )
            if res.fun < 0:
                r_opt, alpha_opt = res.x
                pts = _two_ring_points(r_opt, alpha_opt)
                update_best(pts)
        except Exception:
            pass

    # 3. Local Refinement using Soft-Min Surrogate and L-BFGS-B
    # This allows individual points to move off the rings if beneficial.
    
    if best_pts is not None and time.time() < deadline - 20.0:
        def neg_soft_score_full(x, beta):
            pts = x.reshape(-1, 2)
            return -_soft_score(pts, beta)
        
        current_pts = best_pts.copy()
        current_score = best_score
        
        # Continuation method on beta
        # Start with smooth beta, increase to sharpen
        betas = [5.0, 20.0, 100.0, 500.0]
        
        for beta in betas:
            if time.time() > deadline - 10.0:
                break
            
            try:
                # Use L-BFGS-B which uses gradients (approximated or analytical if possible, 
                # but scipy uses finite differences by default for this interface unless jac provided)
                # Providing a Jacobian would be ideal but complex. Finite diff is okay for 28 dims.
                res = minimize(
                    neg_soft_score_full,
                    current_pts.flatten(),
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                # Check if improvement
                new_score = _score(res.x.reshape(-1, 2))
                if new_score > current_score:
                    current_pts = res.x.reshape(-1, 2)
                    current_score = new_score
                    update_best(current_pts)
            except Exception:
                pass

        # Final polish with hard score using Powell (derivative-free, robust for non-smooth)
        if time.time() < deadline - 5.0:
            try:
                def neg_hard_score_full(x):
                    pts = x.reshape(-1, 2)
                    return -_score(pts)
                    
                res = minimize(
                    neg_hard_score_full,
                    current_pts.flatten(),
                    method='Powell',
                    options={'maxiter': 100, 'xtol': 1e-9, 'ftol': 1e-12}
                )
                
                if res.fun < -current_score:
                    current_pts = res.x.reshape(-1, 2)
                    current_score = -res.fun
                    update_best(current_pts)
            except Exception:
                pass

    # 4. Random Restarts with Local Refinement
    # If time remains, try perturbing the best solution and re-optimizing
    
    if time.time() < deadline - 15.0:
        rng = np.random.RandomState(123)
        for restart in range(5):
            if time.time() > deadline - 10.0:
                break
            
            # Perturb the best known solution
            noise = rng.normal(0, 0.05, size=(N, 2))
            init_pts = best_pts + noise
            update_best(init_pts)
            
            # Quick local refinement with soft min
            if time.time() > deadline - 8.0:
                continue
                
            try:
                def neg_soft_score_restart(x, beta=50.0):
                    pts = x.reshape(-1, 2)
                    return -_soft_score(pts, beta)
                    
                res = minimize(
                    neg_soft_score_restart,
                    init_pts.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-6}
                )
                
                if res.fun < -best_score:
                    new_pts = res.x.reshape(-1, 2)
                    update_best(new_pts)
            except Exception:
                pass

    if best_pts is None:
        return reg_pts
        
    return best_pts
# EVOLVE-BLOCK-END
