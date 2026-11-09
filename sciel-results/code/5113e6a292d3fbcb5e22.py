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

def _simulated_annealing(points, deadline, max_iter=50000, T0=0.5, T_end=1e-6, cooling=0.9999):
    """Simulated Annealing with projection onto unit sphere."""
    n, d = points.shape
    best_points = points.copy()
    best_score = _score(best_points)
    current_points = points.copy()
    current_score = best_score
    
    T = T0
    rng = np.random.RandomState(42)
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Generate a random perturbation
        idx = rng.randint(n)
        direction = rng.randn(d)
        direction /= np.linalg.norm(direction)
        step = T * 0.1
        candidate = current_points.copy()
        candidate[idx] += step * direction
        
        # Project onto unit sphere
        norms = np.linalg.norm(candidate, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        candidate = candidate / norms
        
        candidate_score = _score(candidate)
        
        # Acceptance criterion
        if candidate_score > current_score:
            current_points = candidate
            current_score = candidate_score
            if current_score > best_score:
                best_score = current_score
                best_points = candidate.copy()
        else:
            delta = candidate_score - current_score
            if delta < 0 and rng.random() < np.exp(delta / T):
                current_points = candidate
                current_score = candidate_score
        
        # Cool down
        T = max(T * cooling, T_end)
    
    return best_points

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
    p1 = np.array([0.0, 0.0, 1.0])
    p2 = np.array([0.0, 0.0, -1.0])
    points = np.vstack([points, p1, p2])
    return points

def _make_snub_cube_plus_2():
    """Snub cube (24 verts) - take 12 from it + 2 more."""
    # Use a subset of snub cube vertices or dodecahedron
    # Dodecahedron has 20 vertices, take 12 + 2
    phi = (1 + np.sqrt(5)) / 2
    verts = []
    # Dodecahedron vertices
    for x in [1, -1]:
        for y in [1, -1]:
            for z in [1, -1]:
                verts.append([x, y, z])
    for x in [1, -1]:
        for y in [1, -1]:
            verts.append([0, x/phi, y*phi])
            verts.append([x/phi, y*phi, 0])
            verts.append([y*phi, 0, x/phi])
    points = np.array(verts, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    # Take first 12
    points = points[:12]
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

def _coordinate_descent_polish(points, deadline, max_rounds=10000):
    """
    Coordinate descent polish on the true objective with adaptive step sizes.
    """
    n, d = points.shape
    best_points = points.copy()
    best_score = _score(best_points)
    step_sizes = np.full((n, d), 0.005)
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
                
                # Try multiple step sizes
                best_step = 0
                best_new_score = cur_score
                for factor in [1.0, 0.5, 0.2, 2.0, 0.1]:
                    step = step_sizes[i, j] * factor
                    for sign in [1, -1]:
                        pts_try = best_points.copy()
                        pts_try[i, j] += sign * step
                        score_try = _score(pts_try)
                        if score_try > best_new_score + 1e-14:
                            best_new_score = score_try
                            best_step = sign * step
                
                if best_step != 0:
                    best_points = best_points.copy()
                    best_points[i, j] += best_step
                    best_score = best_new_score
                    improved = True
                    # Adjust step size based on direction
                    step_sizes[i, j] *= 1.05
                else:
                    step_sizes[i, j] *= 0.7
                    if step_sizes[i, j] < 1e-10:
                        step_sizes[i, j] = 0.005
        
        if not improved:
            # Try a global restart of step sizes
            step_sizes = np.full((n, d), 0.005)
            if round_num > 100:
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
