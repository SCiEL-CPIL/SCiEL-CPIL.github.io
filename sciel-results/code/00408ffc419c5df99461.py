import numpy as np
import time
import scipy.spatial as spat
from itertools import combinations
import scipy.optimize as opt

N = 13

# Pre-compute triangle indices for N=13
_triple_idx = list(combinations(range(N), 3))
_i_arr = np.array([t[0] for t in _triple_idx])
_j_arr = np.array([t[1] for t in _triple_idx])
_k_arr = np.array([t[2] for t in _triple_idx])
_n_triples = len(_triple_idx)

def _triangle_areas(points):
    """Compute areas of all C(N,3) triangles."""
    pts = points
    v1 = pts[_j_arr] - pts[_i_arr]
    v2 = pts[_k_arr] - pts[_i_arr]
    det = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(det)

def _hull_area(points):
    """Compute convex hull area."""
    try:
        hull = spat.ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def calculate_score(points):
    """
    Calculate the minimum triangle area divided by the convex hull area.
    Returns 0 if points are collinear or hull area is 0.
    """
    if len(points) < 3:
        return 0.0
    
    hull_area = _hull_area(points)
    if hull_area <= 1e-9:
        return 0.0

    areas = _triangle_areas(points)
    min_area = np.min(areas)
    
    return min_area / hull_area

def _soft_min_score(points, beta=50.0):
    """
    Smooth surrogate: soft-min of triangle areas / hull area.
    """
    hull_area = _hull_area(points)
    if hull_area <= 1e-9:
        return -1e10
    
    areas = _triangle_areas(points)
    max_a = np.max(areas)
    exp_neg = np.exp(-beta * (areas - max_a))
    log_sum = np.log(np.sum(exp_neg)) + beta * max_a
    soft_min_val = log_sum / beta
    
    return soft_min_val / hull_area

def _soft_min_gradient(points, beta=50.0):
    """
    Gradient of the soft-min score w.r.t. point positions.
    """
    hull_area = _hull_area(points)
    if hull_area <= 1e-9:
        return np.zeros((N, 2))
    
    areas = _triangle_areas(points)
    max_a = np.max(areas)
    exp_neg = np.exp(-beta * (areas - max_a))
    weights = exp_neg / np.sum(exp_neg)
    
    pts = points
    v1 = pts[_j_arr] - pts[_i_arr]
    v2 = pts[_k_arr] - pts[_i_arr]
    det_signed = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    signs = np.sign(det_signed)
    signs[signs == 0] = 1.0
    
    grad = np.zeros((N, 2))
    
    w = weights * 0.5 * signs
    
    grad_i_x = w * (pts[_j_arr, 1] - pts[_k_arr, 1])
    grad_i_y = w * (pts[_j_arr, 0] - pts[_k_arr, 0])
    
    grad_j_x = w * v2[:, 1]
    grad_j_y = w * (-v2[:, 0])
    
    grad_k_x = w * (-v1[:, 1])
    grad_k_y = w * v1[:, 0]
    
    np.add.at(grad, (_i_arr, 0), grad_i_x)
    np.add.at(grad, (_i_arr, 1), grad_i_y)
    np.add.at(grad, (_j_arr, 0), grad_j_x)
    np.add.at(grad, (_j_arr, 1), grad_j_y)
    np.add.at(grad, (_k_arr, 0), grad_k_x)
    np.add.at(grad, (_k_arr, 1), grad_k_y)
    
    return grad / hull_area

def _polish(points, max_iter=5000, tol=1e-14):
    """
    High-precision polish using gradient-based optimization on the soft-min surrogate.
    """
    pts = points.copy().astype(np.float64)
    best_score = calculate_score(pts)
    best_pts = pts.copy()
    
    beta = 100.0
    lr = 0.005
    momentum = np.zeros_like(pts)
    mu = 0.9
    
    for it in range(max_iter):
        score = _soft_min_score(pts, beta)
        if score < -1e9:
            break
        
        grad = _soft_min_gradient(pts, beta)
        grad_norm = np.linalg.norm(grad)
        
        if grad_norm < tol:
            break
        
        momentum = mu * momentum + (1 - mu) * grad
        trial = pts + lr * momentum
        
        trial_score = calculate_score(trial)
        if trial_score > best_score:
            best_score = trial_score
            best_pts = trial.copy()
            pts = trial
            lr *= 0.9995
        else:
            lr *= 0.98
            momentum *= 0.5
            if lr < 1e-10:
                break
    
    return best_pts, best_score

def _generate_ring_configuration(radii, angles):
    """
    Generate points from ring parameters.
    radii: list of radii for each ring
    angles: list of angles for each point in each ring
    """
    points = []
    for r, angs in zip(radii, angles):
        for a in angs:
            points.append([r * np.cos(a), r * np.sin(a)])
    return np.array(points)

def _score_ring_params(params, ring_config):
    """
    Score a specific ring configuration given parameters.
    params: [r1, r2, ..., r_k, offset1, offset2, ..., offset_k]
    ring_config: list of counts per ring, e.g., [7, 6]
    """
    k = len(ring_config)
    radii = params[:k]
    offsets = params[k:]
    
    # Ensure radii are positive and ordered (inner to outer)
    # We will sort radii and assign points to rings based on sorted radii
    # But wait, the points are fixed to rings by index. 
    # If we swap radii, we swap the rings. 
    # To avoid symmetry breaking issues, let's just evaluate as is.
    # However, for global search, we might want to allow any order.
    # For now, assume params correspond to ring 0, ring 1, etc.
    
    angles_list = []
    for i, count in enumerate(ring_config):
        # Regular polygon angles with offset
        base_angles = np.linspace(0, 2*np.pi, count, endpoint=False)
        angles_list.append(base_angles + offsets[i])
    
    points = _generate_ring_configuration(radii, angles_list)
    
    # Normalize? The score is invariant to scaling, but the hull area calculation 
    # depends on the scale. The ratio min_area/hull_area is scale invariant.
    # However, numerical stability might be affected. 
    # Let's not normalize inside the function to keep it fast, 
    # but ensure radii are not zero.
    
    score = calculate_score(points)
    return score

def _optimize_ring_config(ring_config, time_budget, initial_guesses=None):
    """
    Optimize a specific ring configuration using Differential Evolution 
    or a grid search + polish.
    """
    k = len(ring_config)
    # Parameters: k radii, k angular offsets
    bounds = [(0.01, 2.0)] * k + [(-np.pi, np.pi)] * k
    
    # Objective function to maximize
    def objective(params):
        return -_score_ring_params(params, ring_config)
    
    best_score = -1.0
    best_params = None
    
    # Strategy: 
    # 1. Try a few structured initial guesses
    # 2. Use Differential Evolution for global search (if time permits)
    # 3. Polish the best found
    
    # Structured guesses
    guesses = []
    
    # 1. Equal spacing
    if k == 1:
        guesses.append([1.0, 0.0])
    elif k == 2:
        # 7 inner, 6 outer or 6 inner, 7 outer?
        # ring_config defines the counts.
        guesses.append([0.5, 1.0, 0.0, np.pi/ring_config[1]]) # Offset to interleave
        guesses.append([0.6, 1.0, 0.0, 0.0])
        guesses.append([0.7, 1.0, 0.0, np.pi/ring_config[1]])
    elif k == 3:
        guesses.append([0.3, 0.6, 1.0, 0.0, 0.0, 0.0])
        guesses.append([0.4, 0.7, 1.0, 0.0, np.pi/ring_config[1], np.pi/ring_config[2]])
    
    if initial_guesses:
        guesses.extend(initial_guesses)
    
    # Evaluate guesses
    for g in guesses:
        if len(g) != 2*k: continue
        # Clip to bounds
        g_clipped = np.clip(g, [b[0] for b in bounds], [b[1] for b in bounds])
        s = _score_ring_params(g_clipped, ring_config)
        if s > best_score:
            best_score = s
            best_params = g_clipped
    
    # Differential Evolution
    # DE is stochastic and can be slow. We use a small population and few generations.
    # Time budget check
    start = time.time()
    remaining = time_budget - (time.time() - start)
    
    if remaining > 5.0:
        try:
            # maxiter is tricky to bound by time. We'll do a quick run.
            # popsize=10, maxiter=50 is usually fast for 4-6 dims.
            res = opt.differential_evolution(
                objective, 
                bounds, 
                seed=42, 
                maxiter=30, 
                popsize=10, 
                tol=1e-4,
                polish=False # We do our own polish
            )
            if res.fun < -best_score: # Minimizing -score
                best_score = -res.fun
                best_params = res.x
        except Exception as e:
            pass # Ignore DE errors, fall back to guesses
    
    # Polish the best parameters using gradient ascent on the full 2D points
    if best_params is not None:
        points = _generate_ring_configuration(best_params[:k], 
                                              [np.linspace(0, 2*np.pi, c, endpoint=False) + best_params[k+i] 
                                               for i, c in enumerate(ring_config)])
        
        # Final polish
        polished_pts, polished_score = _polish(points, max_iter=2000, tol=1e-12)
        if polished_score > best_score:
            best_score = polished_score
            best_params = points # Store points directly if improved? 
            # Actually, we want to return points.
            # Let's just return the polished points.
            return polished_pts, polished_score
    
    # If no improvement from polish, return the points from best_params
    points = _generate_ring_configuration(best_params[:k], 
                                          [np.linspace(0, 2*np.pi, c, endpoint=False) + best_params[k+i] 
                                           for i, c in enumerate(ring_config)])
    return points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_score = -1.0
    best_points = None
    
    # Define candidate ring topologies
    # Sum of counts must be 13
    topologies = [
        [13],       # Regular 13-gon
        [1, 12],    # Center + 12-gon
        [2, 11],    # 2 inner + 11 outer
        [3, 10],    # 3 inner + 10 outer
        [4, 9],     # 4 inner + 9 outer
        [5, 8],     # 5 inner + 8 outer
        [6, 7],     # 6 inner + 7 outer
        [7, 6],     # 7 inner + 6 outer
        [1, 6, 6],  # Center + 6 + 6
        [2, 5, 6],  # 2 + 5 + 6
        [3, 4, 6],  # 3 + 4 + 6
        [4, 4, 5],  # 4 + 4 + 5
        [5, 4, 4],  # 5 + 4 + 4
        [1, 4, 8],  # Center + 4 + 8
        [2, 4, 7],  # 2 + 4 + 7
        [3, 5, 5],  # 3 + 5 + 5
    ]
    
    # Sort topologies by "complexity" or just iterate
    # We will allocate time based on remaining budget.
    
    for topo in topologies:
        if time.time() > deadline - 10.0:
            break
        
        # Allocate time for this topology
        # More complex topologies (more rings) might need more search? 
        # Or maybe simpler ones are better? 
        # Let's give each a fair share, say 5-8 seconds.
        time_budget = min(8.0, deadline - time.time() - 10.0)
        if time_budget < 1.0:
            break
            
        try:
            pts, score = _optimize_ring_config(topo, time_budget)
            if score > best_score:
                best_score = score
                best_points = pts
        except Exception as e:
            # If optimization fails, try a simple regular polygon for this topo
            # Just in case
            pass
    
    # If we didn't find anything good, fallback to regular 13-gon
    if best_score < 0.1:
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
        best_score = calculate_score(best_points)
        
    # Final global polish on the best points found
    if best_points is not None and time.time() < deadline - 2.0:
        final_polished, final_score = _polish(best_points, max_iter=2000, tol=1e-15)
        if final_score > best_score:
            best_score = final_score
            best_points = final_polished
    
    if best_points is None:
        return np.random.rand(N, 2)
        
    return best_points
