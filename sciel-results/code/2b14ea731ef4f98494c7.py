import numpy as np
import time
import scipy.spatial as spat
from itertools import combinations
from scipy.optimize import differential_evolution

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

def _decode(params):
    """
    Decode parameters to Cartesian points.
    params[0:13] are angular increments (positive, sum to 2*pi).
    params[13:26] are radii (positive).
    """
    # Ensure positivity and sum constraint for angles
    # Softplus for angles to ensure > 0
    angles = np.exp(params[:13])
    # Normalize to sum to 2*pi
    angles = 2 * np.pi * angles / np.sum(angles)
    
    # Cumulative sum to get absolute angles
    abs_angles = np.cumsum(angles)
    abs_angles[0] = 0.0 # Start at 0 for rotational symmetry breaking (optional, but helps)
    # Actually, cumulative sum of increments gives absolute angles if we start at 0.
    # Wait, if angles are increments, then:
    # theta_0 = 0
    # theta_1 = angles[0]
    # theta_2 = angles[0] + angles[1]
    # ...
    # This ensures strictly increasing angles, hence convex order.
    
    # Radii
    radii = np.exp(params[13:26])
    
    # Cartesian coordinates
    x = radii * np.cos(abs_angles)
    y = radii * np.sin(abs_angles)
    
    return np.column_stack((x, y))

def _objective(params):
    """
    Objective function for minimization (scipy.optimize.differential_evolution minimizes).
    We want to maximize min_triangle_area / hull_area.
    So we minimize -score.
    """
    points = _decode(params)
    score = calculate_score(points)
    if score < 1e-9:
        return 1e10
    return -score

def _polish(points, max_iter=5000, tol=1e-14):
    """
    High-precision polish using gradient-based optimization on the soft-min surrogate.
    Uses adaptive learning rate and momentum.
    """
    pts = points.copy().astype(np.float64)
    best_score = calculate_score(pts)
    best_pts = pts.copy()
    
    # Start with higher beta for sharper approximation of min
    beta = 100.0
    lr = 0.005
    momentum = np.zeros_like(pts)
    mu = 0.9
    
    for it in range(max_iter):
        # Compute soft min score and gradient
        hull_area = _hull_area(pts)
        if hull_area <= 1e-9:
            break
        
        areas = _triangle_areas(pts)
        max_a = np.max(areas)
        exp_neg = np.exp(-beta * (areas - max_a))
        weights = exp_neg / np.sum(exp_neg)
        
        # Gradient of soft min area
        v1 = pts[_j_arr] - pts[_i_arr]
        v2 = pts[_k_arr] - pts[_i_arr]
        det_signed = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        signs = np.sign(det_signed)
        signs[signs == 0] = 1.0
        
        grad_area = np.zeros((N, 2))
        w = weights * 0.5 * signs
        
        grad_i_x = w * (pts[_j_arr, 1] - pts[_k_arr, 1])
        grad_i_y = w * (pts[_j_arr, 0] - pts[_k_arr, 0])
        
        grad_j_x = w * v2[:, 1]
        grad_j_y = w * (-v2[:, 0])
        
        grad_k_x = w * (-v1[:, 1])
        grad_k_y = w * v1[:, 0]
        
        np.add.at(grad_area, (_i_arr, 0), grad_i_x)
        np.add.at(grad_area, (_i_arr, 1), grad_i_y)
        np.add.at(grad_area, (_j_arr, 0), grad_j_x)
        np.add.at(grad_area, (_j_arr, 1), grad_j_y)
        np.add.at(grad_area, (_k_arr, 0), grad_k_x)
        np.add.at(grad_area, (_k_arr, 1), grad_k_y)
        
        # Gradient of score = min_area / hull_area
        # d(score)/d(pts) = (grad_area * hull_area - min_area * grad_hull) / hull_area^2
        # Approximation: ignore grad_hull for stability in polish, or use finite diff?
        # Let's just use grad_area / hull_area as a proxy for maximizing min area,
        # assuming hull area is relatively stable or we just want to expand the bottleneck.
        # Actually, maximizing min_area/hull_area is tricky.
        # Let's stick to the previous successful gradient approach from the seed if possible,
        # but the seed used a soft-min surrogate that divided by hull area.
        # Let's compute grad of (soft_min_area / hull_area).
        # This is complex. Let's use a simpler local search polish instead to avoid bugs.
        
        # Simple local search polish
        break 
        
    # Fallback: Simple local search polish
    current_pts = pts
    current_score = calculate_score(current_pts)
    best_pts = current_pts.copy()
    best_score = current_score
    
    step_size = 0.01
    no_improve = 0
    max_no_improve = 100
    
    for it in range(max_iter):
        if no_improve > max_no_improve:
            break
        
        # Find bottleneck triangle
        areas = _triangle_areas(current_pts)
        min_idx = np.argmin(areas)
        i, j, k = _i_arr[min_idx], _j_arr[min_idx], _k_arr[min_idx]
        
        p_i = current_pts[i]
        p_j = current_pts[j]
        p_k = current_pts[k]
        
        centroid_jk = (p_j + p_k) / 2.0
        centroid_ik = (p_i + p_k) / 2.0
        centroid_ij = (p_i + p_j) / 2.0
        
        dir_i = p_i - centroid_jk
        dir_j = p_j - centroid_ik
        dir_k = p_k - centroid_ij
        
        norm_i = np.linalg.norm(dir_i)
        norm_j = np.linalg.norm(dir_j)
        norm_k = np.linalg.norm(dir_k)
        
        if norm_i > 1e-6: dir_i /= norm_i
        if norm_j > 1e-6: dir_j /= norm_j
        if norm_k > 1e-6: dir_k /= norm_k
        
        improved = False
        for factor in [1.0, 0.5, 0.2, 0.1]:
            trial_pts = current_pts.copy()
            trial_pts[i] += step_size * factor * dir_i
            trial_pts[j] += step_size * factor * dir_j
            trial_pts[k] += step_size * factor * dir_k
            
            trial_score = calculate_score(trial_pts)
            
            if trial_score > current_score:
                current_pts = trial_pts
                current_score = trial_score
                no_improve = 0
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_pts = current_pts.copy()
                break
        else:
            no_improve += 1
            step_size *= 0.95
            
            if no_improve % 20 == 0:
                # Random perturbation
                np.random.seed(int(time.time() * 1000) % 100000)
                perturb = np.random.normal(0, step_size / 2, (N, 2))
                trial_pts = current_pts + perturb
                trial_score = calculate_score(trial_pts)
                if trial_score > current_score:
                    current_pts = trial_pts
                    current_score = trial_score
                    no_improve = 0
                    if trial_score > best_score:
                        best_score = trial_score
                        best_pts = current_pts.copy()
                
                if step_size < 1e-6:
                    break
                    
    return best_pts, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_score = -1.0
    best_points = None
    
    # Bounds for Differential Evolution
    # Angles (increments) are in exp space. 
    # If we want increments to be roughly 2*pi/13 ~ 0.48, exp(x) ~ 0.48 => x ~ -0.73
    # Range: -2 to 2 covers exp(-2)=0.13 to exp(2)=7.3. 
    # Radii: exp(y). We want radii roughly 1. y ~ 0.
    # Range: -2 to 2 covers 0.13 to 7.3.
    bounds = [(-2, 2)] * (2 * N)
    
    # Initial population for DE
    # 1. Regular polygon
    angles_reg = np.full(N, 2 * np.pi / N)
    radii_reg = np.ones(N)
    params_reg = np.concatenate((np.log(angles_reg), np.log(radii_reg)))
    
    # 2. Slightly perturbed regular polygon
    np.random.seed(42)
    angles_pert = angles_reg + np.random.normal(0, 0.05, N)
    radii_pert = radii_reg + np.random.normal(0, 0.05, N)
    params_pert = np.concatenate((np.log(angles_pert), np.log(radii_pert)))
    
    # 3. Random within bounds
    params_rand = np.random.uniform(-1, 1, 2 * N)
    
    # Create initial population
    pop_size = 20
    initial_pop = [params_reg, params_pert, params_rand]
    while len(initial_pop) < pop_size:
        initial_pop.append(np.random.uniform(-1, 1, 2 * N))
    
    # Run Differential Evolution
    try:
        # maxiter is limited by time, but DE will stop if it converges or hits maxiter
        # We'll run DE for a fixed number of iterations and check time
        # DE is slow per iteration, so we need to be careful.
        # Let's run DE with a small population and limited iterations first.
        
        # To respect time limits, we'll run DE in chunks or just rely on its internal stopping.
        # However, scipy DE doesn't have a direct time limit.
        # We can wrap the objective to check time, but that's risky.
        # Instead, we'll run DE with a reasonable maxiter (e.g., 100) and see if it finishes.
        # If it's too slow, we'll fallback to local search.
        
        # Let's estimate cost. N=13, pop=20, maxiter=100. 2000 evaluations.
        # Each eval is fast (vectorized). Should be < 1s.
        
        res = differential_evolution(
            _objective, 
            bounds, 
            seed=42, 
            maxiter=100, 
            popsize=pop_size, 
            init=initial_pop, 
            tol=1e-8, 
            disp=False
        )
        
        if res.success or res.fun < 1e9:
            best_params = res.x
            best_points_de = _decode(best_params)
            best_score_de = calculate_score(best_points_de)
            
            if best_score_de > best_score:
                best_score = best_score_de
                best_points = best_points_de.copy()
                
    except Exception as e:
        # Fallback if DE fails
        pass
    
    # If DE didn't find anything good, or to improve further, run local polish
    if best_points is not None:
        if time.time() < deadline - 5.0:
            polished, polish_score = _polish(best_points, max_iter=2000, tol=1e-14)
            if polish_score > best_score:
                best_score = polish_score
                best_points = polished.copy()
    else:
        # Fallback to regular polygon if DE failed completely
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
        best_score = calculate_score(best_points)

    return best_points
