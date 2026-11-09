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

def _smooth_objective(x, n=14, d=3, beta=10.0, alpha=10.0):
    """
    Smooth surrogate for the objective.
    Minimizes - (log(softmin(d)) - log(softmax(d)))
    which is equivalent to maximizing log(softmin(d)/softmax(d))
    """
    points = x.reshape(n, d)
    dists = pdist(points)
    
    # Handle zero distances
    if np.any(dists < 1e-12):
        return 1e6 # Large penalty
    
    # Stable log-sum-exp
    # softmin(d) = -1/beta * log(sum(exp(-beta * d)))
    # softmax(d) = 1/alpha * log(sum(exp(alpha * d)))
    
    # We want to maximize log(softmin) - log(softmax)
    # log(softmin) = -1/beta * logsumexp(-beta * d)
    # log(softmax) = 1/alpha * logsumexp(alpha * d)
    
    # To avoid overflow in exp, we subtract max
    neg_beta_d = -beta * dists
    max_neg_beta_d = np.max(neg_beta_d)
    log_sum_exp_neg = max_neg_beta_d + np.log(np.sum(np.exp(neg_beta_d - max_neg_beta_d)))
    log_softmin = -1.0 / beta * log_sum_exp_neg
    
    alpha_d = alpha * dists
    max_alpha_d = np.max(alpha_d)
    log_sum_exp_pos = max_alpha_d + np.log(np.sum(np.exp(alpha_d - max_alpha_d)))
    log_softmax = 1.0 / alpha * log_sum_exp_pos
    
    objective_value = -(log_softmin - log_softmax)
    
    if not np.isfinite(objective_value):
        return 1e6
        
    return objective_value

def _make_icosahedron_plus_2():
    """Icosahedron (12 verts) + 2 points at poles of a face."""
    phi = (1 + np.sqrt(5)) / 2
    verts = []
    for s1 in [1, -1]:
        for s2 in [1, -1]:
            verts.append([0, s1, s2 * phi])
            verts.append([s1, s2 * phi, 0])
            verts.append([s2 * phi, 0, s1])
    points = np.array(verts, dtype=float)
    # Normalize to unit sphere
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    # Add 2 points: place them at the centers of two opposite faces
    # Using (0,0,1) and (0,0,-1) as initial guesses for the optimizer to move
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

def _coordinate_descent_polish(points, deadline, max_rounds=5000):
    """
    Coordinate descent polish on the true objective.
    This is a local refinement method that moves one coordinate at a time.
    """
    n, d = points.shape
    best_points = points.copy()
    best_score = _score(best_points)
    step_sizes = np.full((n, d), 0.01)
    start_time = time.time()
    
    for round_num in range(max_rounds):
        if time.time() > deadline:
            break
        
        improved = False
        for i in range(n):
            for j in range(d):
                if time.time() > deadline:
                    return best_points
                
                cur_score = best_score
                
                # Try positive step
                pts_plus = best_points.copy()
                pts_plus[i, j] += step_sizes[i, j]
                score_plus = _score(pts_plus)
                
                # Try negative step
                pts_minus = best_points.copy()
                pts_minus[i, j] -= step_sizes[i, j]
                score_minus = _score(pts_minus)
                
                # Try zero step (reset)
                if score_plus > cur_score + 1e-12:
                    best_points = pts_plus
                    best_score = score_plus
                    improved = True
                    step_sizes[i, j] *= 1.1
                elif score_minus > cur_score + 1e-12:
                    best_points = pts_minus
                    best_score = score_minus
                    improved = True
                    step_sizes[i, j] *= 1.1
                else:
                    step_sizes[i, j] *= 0.8
                    if step_sizes[i, j] < 1e-8:
                        step_sizes[i, j] = 0.01
        
        if not improved:
            break
    
    return best_points

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    """
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 590  # 630s limit, keep 40s margin
    
    best_score = -1.0
    best_points = None
    
    # Generate diverse initial guesses
    guesses = []
    guesses.append(_make_icosahedron_plus_2())
    guesses.append(_make_cuboctahedron_plus_2())
    
    # Add random unit sphere points with various seeds
    for i in range(10):
        guesses.append(_make_random_unit(n, d, seed=1000 + i * 137))
    
    # Optimize each guess
    for i, guess in enumerate(guesses):
        if time.time() > deadline - 30:
            break
            
        x0 = guess.flatten()
        
        # Check initial score
        current_score = _score(guess)
        if current_score > best_score:
            best_score = current_score
            best_points = guess.copy()
        
        # Phase 1: Smooth optimization with L-BFGS-B
        try:
            # Use a moderate beta for smoothness
            res_smooth = minimize(_smooth_objective, x0, method='L-BFGS-B', 
                                  options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50})
            
            if not np.any(np.isnan(res_smooth.x)):
                points_smooth = res_smooth.x.reshape(n, d)
                score_smooth = _score(points_smooth)
                if score_smooth > best_score:
                    best_score = score_smooth
                    best_points = points_smooth
        except Exception:
            pass
        
        # Phase 2: Polish with Nelder-Mead on the true objective
        if time.time() > deadline - 15:
            continue
            
        try:
            x_start = best_points.flatten() if best_points is not None else x0
            res_polish = minimize(lambda x: -_score(x.reshape(n, d)), x_start, method='Nelder-Mead', 
                                  options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            
            if not np.any(np.isnan(res_polish.x)):
                points_polish = res_polish.x.reshape(n, d)
                score_polish = _score(points_polish)
                if score_polish > best_score:
                    best_score = score_polish
                    best_points = points_polish
        except Exception:
            pass
    
    # If we still have time, do a final high-precision refinement from the best point
    if best_points is not None and time.time() < deadline - 10:
        x0 = best_points.flatten()
        try:
            # One more smooth pass with higher precision?
            res_final = minimize(_smooth_objective, x0, method='L-BFGS-B', 
                                  options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10})
            if not np.any(np.isnan(res_final.x)):
                points_final = res_final.x.reshape(n, d)
                score_final = _score(points_final)
                if score_final > best_score:
                    best_score = score_final
                    best_points = points_final
        except Exception:
            pass
        
        # Final Nelder-Mead polish
        if time.time() < deadline - 5:
            try:
                res_final_nm = minimize(lambda x: -_score(x.reshape(n, d)), best_points.flatten(), method='Nelder-Mead', 
                                        options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
                if not np.any(np.isnan(res_final_nm.x)):
                    points_final_nm = res_final_nm.x.reshape(n, d)
                    score_final_nm = _score(points_final_nm)
                    if score_final_nm > best_score:
                        best_score = score_final_nm
                        best_points = points_final_nm
            except Exception:
                pass
    
    # Final coordinate descent polish if time permits
    if best_points is not None and time.time() < deadline - 2:
        try:
            best_points = _coordinate_descent_polish(best_points, deadline)
            final_score = _score(best_points)
            if final_score > best_score:
                best_score = final_score
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
