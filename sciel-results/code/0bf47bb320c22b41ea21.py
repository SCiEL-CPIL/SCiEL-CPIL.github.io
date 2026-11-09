# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def _score(points):
    """Compute (d_min / d_max)^2 for a set of points."""
    n = len(points)
    if n < 2:
        return 0.0
    dists = pdist(points)
    if len(dists) == 0:
        return 0.0
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max == 0 or not np.isfinite(d_min) or not np.isfinite(d_max):
        return 0.0
    return (d_min / d_max) ** 2

def _smooth_objective(x, n=14, d=3, beta=15.0, alpha=15.0):
    """
    Smooth surrogate: minimize -(log_softmin - log_softmax)
    which is equivalent to maximizing log(softmin/softmax).
    """
    points = x.reshape(n, d)
    dists = pdist(points)
    
    if np.any(dists < 1e-10):
        return 1e6
    
    neg_beta_d = -beta * dists
    max_nb = np.max(neg_beta_d)
    log_sum_exp_neg = max_nb + np.log(np.sum(np.exp(neg_beta_d - max_nb)))
    log_softmin = -1.0 / beta * log_sum_exp_neg
    
    alpha_d = alpha * dists
    max_ad = np.max(alpha_d)
    log_sum_exp_pos = max_ad + np.log(np.sum(np.exp(alpha_d - max_ad)))
    log_softmax = 1.0 / alpha * log_sum_exp_pos
    
    obj = -(log_softmin - log_softmax)
    
    if not np.isfinite(obj):
        return 1e6
    return obj

def _make_icosahedron_plus_2():
    """Icosahedron (12 verts) + 2 face-center points."""
    phi = (1 + np.sqrt(5)) / 2
    verts = []
    for s1 in [1, -1]:
        for s2 in [1, -1]:
            verts.append([0, s1, s2 * phi])
            verts.append([s1, s2 * phi, 0])
            verts.append([s2 * phi, 0, s1])
    points = np.array(verts, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    # Place 2 additional points at opposite face centers
    # For icosahedron, face centers are at distance ~0.951 from origin
    # Two opposite faces: use (0, 0, 1) and (0, 0, -1) as approximations
    p1 = np.array([0.0, 0.0, 1.0])
    p2 = np.array([0.0, 0.0, -1.0])
    points = np.vstack([points, p1, p2])
    return points

def _make_cuboctahedron_plus_2():
    """Cuboctahedron (12 verts) + 2 points."""
    verts = []
    for x in [1, -1]:
        for y in [1, -1]:
            verts.append([x, y, 0])
            verts.append([x, 0, y])
            verts.append([0, x, y])
    points = np.array(verts, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    p1 = np.array([1.0, 0.0, 0.0])
    p2 = np.array([-1.0, 0.0, 0.0])
    points = np.vstack([points, p1, p2])
    return points

def _make_random_unit(n=14, d=3, seed=None):
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    points = rng.randn(n, d)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    points = points / norms
    return points

def _repulsion_refine(points, n_iter=500, deadline=None):
    """
    Greedy repulsion-based local search:
    For each point, try to move it to maximize the minimum distance to all other points.
    Uses a simple coordinate-perturbation approach.
    """
    n, d = points.shape
    best = points.copy()
    best_score = _score(best)
    
    step = 0.02
    for it in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(n):
            for j in range(d):
                orig = best[i, j]
                found = False
                for sign in [1, -1]:
                    for s in [step, step * 0.5, step * 0.25]:
                        cand = best.copy()
                        cand[i, j] = orig + sign * s
                        sc = _score(cand)
                        if sc > best_score + 1e-14:
                            best = cand
                            best_score = sc
                            improved = True
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                break
        if not improved:
            step *= 0.5
            if step < 1e-9:
                break
    return best

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize (d_min/d_max)^2.
    """
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 600  # Keep 30s margin for safety
    
    best_score = -1.0
    best_points = None
    
    # Generate diverse initial guesses
    guesses = []
    guesses.append(_make_icosahedron_plus_2())
    guesses.append(_make_cuboctahedron_plus_2())
    
    # Add random unit sphere points with various seeds for diversity
    for i in range(15):
        guesses.append(_make_random_unit(n, d, seed=42 + i * 7919))
    
    # Optimize each guess
    for i, guess in enumerate(guesses):
        if time.time() > deadline - 60:
            break
            
        x0 = guess.flatten()
        
        # Check initial score
        current_score = _score(guess)
        if current_score > best_score:
            best_score = current_score
            best_points = guess.copy()
        
        # Phase 1: Smooth L-BFGS-B optimization
        try:
            res1 = minimize(_smooth_objective, x0, method='L-BFGS-B',
                           options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50})
            if not np.any(np.isnan(res1.x)):
                pts = res1.x.reshape(n, d)
                sc = _score(pts)
                if sc > best_score:
                    best_score = sc
                    best_points = pts
        except Exception:
            pass
        
        # Phase 2: Higher-precision smooth optimization from best so far
        if time.time() < deadline - 30:
            try:
                x_start = (best_points if best_points is not None else guess).flatten()
                res2 = minimize(_smooth_objective, x_start, method='L-BFGS-B',
                               options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50,
                                        'beta': 25.0})
                # Re-call with adjusted beta via closure is complex; use fixed beta
                # Instead, do another pass with default
                res2 = minimize(_smooth_objective, x_start, method='L-BFGS-B',
                               options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50})
                if not np.any(np.isnan(res2.x)):
                    pts2 = res2.x.reshape(n, d)
                    sc2 = _score(pts2)
                    if sc2 > best_score:
                        best_score = sc2
                        best_points = pts2
            except Exception:
                pass
        
        # Phase 3: Nelder-Mead polish on true objective
        if time.time() < deadline - 15:
            try:
                x_start = (best_points if best_points is not None else guess).flatten()
                res_nm = minimize(lambda x: -_score(x.reshape(n, d)), x_start,
                                 method='Nelder-Mead',
                                 options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
                if not np.any(np.isnan(res_nm.x)):
                    pts_nm = res_nm.x.reshape(n, d)
                    sc_nm = _score(pts_nm)
                    if sc_nm > best_score:
                        best_score = sc_nm
                        best_points = pts_nm
            except Exception:
                pass
    
    # Final polish from the best point found
    if best_points is not None and time.time() < deadline - 10:
        # Coordinate descent polish
        try:
            polished = _repulsion_refine(best_points, n_iter=500, deadline=deadline - 5)
            sc_pol = _score(polished)
            if sc_pol > best_score:
                best_score = sc_pol
                best_points = polished
        except Exception:
            pass
        
        # One more Nelder-Mead from the polished result
        if time.time() < deadline - 5:
            try:
                res_final = minimize(lambda x: -_score(x.reshape(n, d)), best_points.flatten(),
                                    method='Nelder-Mead',
                                    options={'maxiter': 5000, 'xatol': 1e-11, 'fatol': 1e-13, 'adaptive': True})
                if not np.any(np.isnan(res_final.x)):
                    pts_final = res_final.x.reshape(n, d)
                    sc_final = _score(pts_final)
                    if sc_final > best_score:
                        best_score = sc_final
                        best_points = pts_final
            except Exception:
                pass
    
    # Fallback
    if best_points is None:
        best_points = _make_icosahedron_plus_2()
    
    # Ensure finite and correct shape
    if not np.all(np.isfinite(best_points)):
        best_points = _make_icosahedron_plus_2()
    
    return best_points

# EVOLVE-BLOCK-END
