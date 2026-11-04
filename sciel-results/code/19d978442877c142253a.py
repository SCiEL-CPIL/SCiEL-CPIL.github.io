"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution, minimize_scalar

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

def _two_ring_points(theta_inner, theta_outer, r_inner):
    """Construct 14 points: 7 inner at radius r_inner, 7 outer at radius 1.0"""
    inner_pts = np.column_stack([r_inner * np.cos(theta_inner), r_inner * np.sin(theta_inner)])
    outer_pts = np.column_stack([np.cos(theta_outer), np.sin(theta_outer)])
    return np.vstack([inner_pts, outer_pts])

def _two_ring_score(params):
    """params = [theta_inner(7), theta_outer(7), r_inner]"""
    theta_inner = params[:7]
    theta_outer = params[7:14]
    r_inner = params[14]
    pts = _two_ring_points(theta_inner, theta_outer, r_inner)
    return -_score(pts)

def _three_ring_points(theta1, theta2, theta3, r1, r2):
    """Construct 14 points in three rings: 5, 5, 4 points"""
    pts1 = np.column_stack([r1 * np.cos(theta1), r1 * np.sin(theta1)])
    pts2 = np.column_stack([r2 * np.cos(theta2), r2 * np.sin(theta2)])
    pts3 = np.column_stack([np.cos(theta3), np.sin(theta3)])
    return np.vstack([pts1, pts2, pts3])

def _three_ring_score(params):
    """params = [theta1(5), theta2(5), theta3(4), r1, r2]"""
    theta1 = params[:5]
    theta2 = params[5:10]
    theta3 = params[10:14]
    r1 = params[14]
    r2 = params[15]
    pts = _three_ring_points(theta1, theta2, theta3, r1, r2)
    return -_score(pts)

def _local_refine(pts, deadline):
    """Local refinement using Powell on the true objective"""
    def objective(x):
        p = x.reshape(-1, 2)
        return -_score(p)
    
    try:
        res = minimize(
            objective, pts.flatten(),
            method='Powell',
            options={'maxiter': 300, 'xtol': 1e-10, 'ftol': 1e-12}
        )
        new_pts = res.x.reshape(-1, 2)
        return new_pts, -res.fun
    except Exception:
        return pts, _score(pts)

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
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    return min_a - (1.0 / beta) * log_sum

def _soft_score(pts, beta):
    hull_area = _convex_hull_area(pts)
    if hull_area < 1e-9:
        return -1e10
    soft_min = _soft_min_area(pts, beta)
    return soft_min / hull_area

def _soft_objective(x, beta):
    pts = x.reshape(-1, 2)
    return -_soft_score(pts, beta)

def _continuation_optimize(pts, deadline):
    """Continuation method with soft-min, then local refinement"""
    best_pts = pts.copy()
    best_score = _score(pts)
    
    betas = [10.0, 50.0, 200.0, 1000.0]
    current_pts = pts
    
    for beta in betas:
        if time.time() > deadline - 8.0:
            break
        try:
            res = minimize(
                _soft_objective, current_pts.flatten(),
                args=(beta,),
                method='L-BFGS-B',
                options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            new_pts = res.x.reshape(-1, 2)
            new_score = _score(new_pts)
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts.copy()
            current_pts = new_pts
        except Exception:
            pass
    
    # Final local refinement
    if time.time() < deadline - 5.0:
        refined_pts, refined_score = _local_refine(best_pts, deadline)
        if refined_score > best_score:
            best_score = refined_score
            best_pts = refined_pts
    
    return best_score, best_pts

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_score = -np.inf
    best_pts = None
    
    # Initializations
    inits = []
    
    # 1. Regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    inits.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 2. Two rings: 7 inner, 7 outer with optimal angular offset
    for r_inner in [0.3, 0.5, 0.7]:
        inner_angles = np.linspace(0, 2 * np.pi, 7, endpoint=False)
        outer_angles = inner_angles + (2 * np.pi / 14)
        inner_pts = np.column_stack([r_inner * np.cos(inner_angles), r_inner * np.sin(inner_angles)])
        outer_pts = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
        inits.append(np.vstack([inner_pts, outer_pts]))
    
    # 3. Three rings: 5, 5, 4
    for r1, r2 in [(0.3, 0.65), (0.4, 0.75), (0.5, 0.8)]:
        t1 = np.linspace(0, 2 * np.pi, 5, endpoint=False)
        t2 = t1 + (2 * np.pi / 10)
        t3 = np.linspace(0, 2 * np.pi, 4, endpoint=False) + (2 * np.pi / 8)
        pts1 = np.column_stack([r1 * np.cos(t1), r1 * np.sin(t1)])
        pts2 = np.column_stack([r2 * np.cos(t2), r2 * np.sin(t2)])
        pts3 = np.column_stack([np.cos(t3), np.sin(t3)])
        inits.append(np.vstack([pts1, pts2, pts3]))
    
    # 4. Perturbed regular polygon
    rng = np.random.RandomState(42)
    for noise_scale in [0.03, 0.1, 0.2]:
        perturbed = inits[0] + rng.normal(0, noise_scale, size=(N, 2))
        inits.append(perturbed)
    
    # 5. Random points
    rng2 = np.random.RandomState(123)
    for seed_offset in range(3):
        rand_angles = rng2.uniform(0, 2 * np.pi, N)
        rand_radii = rng2.uniform(0.4, 1.0, N)
        rand_pts = np.column_stack([rand_radii * np.cos(rand_angles), rand_radii * np.sin(rand_angles)])
        inits.append(rand_pts)
    
    # Phase 1: Evaluate all initializations and pick top candidates
    scored_inits = []
    for init_pts in inits:
        s = _score(init_pts)
        scored_inits.append((s, init_pts))
    scored_inits.sort(key=lambda x: -x[0])
    
    # Phase 2: Optimize top 8 initializations with continuation
    top_k = 8
    for idx in range(min(top_k, len(scored_inits))):
        if time.time() > deadline - 20.0:
            break
        
        init_pts = scored_inits[idx][1]
        local_score, local_pts = _continuation_optimize(init_pts, deadline)
        
        if local_score > best_score:
            best_score = local_score
            best_pts = local_pts.copy()
    
    # Phase 3: Differential evolution on two-ring model
    if time.time() < deadline - 30.0:
        try:
            # Bounds: angles in [0, 2pi), r_inner in [0.2, 0.9]
            bounds = [(0, 2*np.pi)] * 14 + [(0.2, 0.9)]
            de_result = differential_evolution(
                _two_ring_score, bounds,
                maxiter=50, popsize=15,
                seed=42, tol=1e-8,
                polish=True
            )
            if de_result.success or de_result.fun < -best_score:
                de_params = de_result.x
                de_pts = _two_ring_points(de_params[:7], de_params[7:14], de_params[14])
                de_score = _score(de_pts)
                if de_score > best_score:
                    best_score = de_score
                    best_pts = de_pts.copy()
                
                # Refine the DE result
                if time.time() < deadline - 10.0:
                    refined_pts, refined_score = _continuation_optimize(de_pts, deadline)
                    if refined_score > best_score:
                        best_score = refined_score
                        best_pts = refined_pts
        except Exception:
            pass
    
    # Phase 4: Differential evolution on three-ring model
    if time.time() < deadline - 25.0:
        try:
            bounds = [(0, 2*np.pi)] * 14 + [(0.15, 0.6), (0.5, 0.9)]
            de_result3 = differential_evolution(
                _three_ring_score, bounds,
                maxiter=40, popsize=12,
                seed=77, tol=1e-8,
                polish=True
            )
            if de_result3.success or de_result3.fun < -best_score:
                de3_params = de_result3.x
                de3_pts = _three_ring_points(de3_params[:5], de3_params[5:10], de3_params[10:14], de3_params[14], de3_params[15])
                de3_score = _score(de3_pts)
                if de3_score > best_score:
                    best_score = de3_score
                    best_pts = de3_pts.copy()
                
                if time.time() < deadline - 8.0:
                    refined_pts, refined_score = _continuation_optimize(de3_pts, deadline)
                    if refined_score > best_score:
                        best_score = refined_score
                        best_pts = refined_pts
        except Exception:
            pass
    
    # Phase 5: Final local refinement of best result
    if best_pts is not None and time.time() < deadline - 3.0:
        refined_pts, refined_score = _local_refine(best_pts, deadline)
        if refined_score > best_score:
            best_score = refined_score
            best_pts = refined_pts
    
    if best_pts is None:
        return inits[0]
    
    return best_pts
# EVOLVE-BLOCK-END
