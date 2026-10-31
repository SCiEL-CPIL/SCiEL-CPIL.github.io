import numpy as np
import time
import scipy.spatial as spat
from itertools import combinations

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
    Uses log-sum-exp for numerical stability.
    """
    hull_area = _hull_area(points)
    if hull_area <= 1e-9:
        return -1e10
    
    areas = _triangle_areas(points)
    # Soft min: -log(sum(exp(-beta * a))) / beta
    # For stability, subtract max
    max_a = np.max(areas)
    exp_neg = np.exp(-beta * (areas - max_a))
    log_sum = np.log(np.sum(exp_neg)) + beta * max_a
    soft_min_val = log_sum / beta
    
    return soft_min_val / hull_area

def _soft_min_gradient(points, beta=50.0):
    """
    Gradient of the soft-min score w.r.t. point positions.
    Returns gradient array of shape (N, 2).
    """
    hull_area = _hull_area(points)
    if hull_area <= 1e-9:
        return np.zeros((N, 2))
    
    areas = _triangle_areas(points)
    max_a = np.max(areas)
    exp_neg = np.exp(-beta * (areas - max_a))
    weights = exp_neg / np.sum(exp_neg)  # softmax weights (sum to 1)
    
    pts = points
    v1 = pts[_j_arr] - pts[_i_arr]
    v2 = pts[_k_arr] - pts[_i_arr]
    det_signed = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    signs = np.sign(det_signed)
    signs[signs == 0] = 1.0
    
    grad = np.zeros((N, 2))
    
    # Vectorized gradient computation
    # For point i: d(area)/dp_i = 0.5 * sign * (p_jy - p_ky, p_jx - p_kx)
    # For point j: d(area)/dp_j = 0.5 * sign * (p_ky - p_iy, -(p_kx - p_ix))
    # For point k: d(area)/dp_k = 0.5 * sign * (-(p_jy - p_iy), (p_jx - p_ix))
    
    # Build arrays for vectorized accumulation
    w = weights * 0.5 * signs
    
    # Contributions to each point
    # For i:
    grad_i_x = w * (pts[_j_arr, 1] - pts[_k_arr, 1])
    grad_i_y = w * (pts[_j_arr, 0] - pts[_k_arr, 0])
    
    # For j:
    grad_j_x = w * v2[:, 1]
    grad_j_y = w * (-v2[:, 0])
    
    # For k:
    grad_k_x = w * (-v1[:, 1])
    grad_k_y = w * v1[:, 0]
    
    # Accumulate using np.add.at
    np.add.at(grad, (_i_arr, 0), grad_i_x)
    np.add.at(grad, (_i_arr, 1), grad_i_y)
    np.add.at(grad, (_j_arr, 0), grad_j_x)
    np.add.at(grad, (_j_arr, 1), grad_j_y)
    np.add.at(grad, (_k_arr, 0), grad_k_x)
    np.add.at(grad, (_k_arr, 1), grad_k_y)
    
    return grad / hull_area

def find_bottleneck(points):
    """
    Returns the indices (i, j, k) of the triangle with the minimum area,
    and the area itself.
    """
    areas = _triangle_areas(points)
    min_idx = np.argmin(areas)
    return _i_arr[min_idx], _j_arr[min_idx], _k_arr[min_idx], areas[min_idx]

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
        score = _soft_min_score(pts, beta)
        if score < -1e9:
            break
        
        grad = _soft_min_gradient(pts, beta)
        grad_norm = np.linalg.norm(grad)
        
        if grad_norm < tol:
            break
        
        # Momentum-based update
        momentum = mu * momentum + (1 - mu) * grad
        trial = pts + lr * momentum
        
        trial_score = calculate_score(trial)
        if trial_score > best_score:
            best_score = trial_score
            best_pts = trial.copy()
            pts = trial
            # Decay learning rate slowly
            lr *= 0.9995
        else:
            lr *= 0.98
            # Reduce momentum if not improving
            momentum *= 0.5
            if lr < 1e-10:
                break
    
    return best_pts, best_score

def _bottleneck_directed_search(points, time_budget):
    """
    Directed local search that moves bottleneck triangle vertices apart.
    Returns improved points and score.
    """
    current_pts = points.copy()
    current_score = calculate_score(current_pts)
    best_pts = current_pts.copy()
    best_score = current_score
    
    step_size = 0.02
    no_improve = 0
    max_no_improve = 300
    start = time.time()
    
    while time.time() - start < time_budget and no_improve < max_no_improve:
        i, j, k, min_area = find_bottleneck(current_pts)
        
        p_i = current_pts[i]
        p_j = current_pts[j]
        p_k = current_pts[k]
        
        # Direction away from the opposite side
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
        
        # Try multiple step sizes
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
                
                if current_score > best_score:
                    best_score = current_score
                    best_pts = current_pts.copy()
                break
        else:
            no_improve += 1
            step_size *= 0.95
            
            if no_improve % 20 == 0:
                # Random perturbation to escape local minimum
                np.random.seed(int(time.time() * 1000) % 100000)
                perturb = np.random.normal(0, step_size / 2, (N, 2))
                trial_pts = current_pts + perturb
                trial_score = calculate_score(trial_pts)
                if trial_score > current_score:
                    current_pts = trial_pts
                    current_score = trial_score
                    no_improve = 0
                    if current_score > best_score:
                        best_score = current_score
                        best_pts = current_pts.copy()
                
                if step_size < 1e-6:
                    break
    
    return best_pts, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_score = -1.0
    best_points = None
    
    # Initialize with multiple structured seeds
    # 1. Regular 13-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    initial_pts = np.column_stack((np.cos(angles), np.sin(angles)))
    
    # 2. Perturbed regular 13-gon
    np.random.seed(42)
    noise = np.random.normal(0, 0.1, (N, 2))
    initial_pts2 = initial_pts + noise
    
    # 3. Fibonacci lattice
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    fib_pts = np.zeros((N, 2))
    for i in range(N):
        r = np.sqrt((i + 0.5) / N)
        theta = golden_angle * i
        fib_pts[i] = [r * np.cos(theta), r * np.sin(theta)]
    fib_pts /= np.max(np.linalg.norm(fib_pts, axis=1))
    
    # 4. Another perturbation
    np.random.seed(123)
    noise3 = np.random.normal(0, 0.05, (N, 2))
    initial_pts3 = initial_pts + noise3
    
    # 5. Two-ring structure: inner hexagon + outer heptagon
    inner_angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
    outer_angles = np.linspace(0, 2*np.pi, 7, endpoint=False) + np.pi/7
    two_ring = np.vstack([
        np.column_stack((0.5 * np.cos(inner_angles), 0.5 * np.sin(inner_angles))),
        np.column_stack((np.cos(outer_angles), np.sin(outer_angles)))
    ])
    
    # 6. Three-ring: center + 4 inner + 8 outer
    center = np.array([[0.0, 0.0]])
    inner4_angles = np.linspace(0, 2*np.pi, 4, endpoint=False)
    outer8_angles = np.linspace(0, 2*np.pi, 8, endpoint=False) + np.pi/8
    three_ring = np.vstack([
        center,
        np.column_stack((0.4 * np.cos(inner4_angles), 0.4 * np.sin(inner4_angles))),
        np.column_stack((np.cos(outer8_angles), np.sin(outer8_angles)))
    ])
    
    candidates = [initial_pts, initial_pts2, fib_pts, initial_pts3, two_ring, three_ring]
    
    for init in candidates:
        if time.time() > deadline - 20.0:
            break
        
        current_pts = init.copy()
        current_score = calculate_score(current_pts)
        
        if current_score > best_score:
            best_score = current_score
            best_points = current_pts.copy()
        
        # Phase 1: Gradient-based optimization on soft-min surrogate
        # Use the smooth objective for faster convergence
        beta = 30.0
        lr = 0.02
        no_improve = 0
        max_no_improve = 400
        momentum = np.zeros_like(current_pts)
        mu = 0.85
        phase_start = time.time()
        
        while time.time() < deadline - 25.0 and no_improve < max_no_improve:
            score = _soft_min_score(current_pts, beta)
            if score < -1e9:
                break
            
            grad = _soft_min_gradient(current_pts, beta)
            grad_norm = np.linalg.norm(grad)
            
            if grad_norm < 1e-12:
                break
            
            # Momentum-based gradient ascent
            momentum = mu * momentum + (1 - mu) * grad
            trial = current_pts + lr * momentum
            
            trial_score = calculate_score(trial)
            
            if trial_score > current_score:
                current_pts = trial
                current_score = trial_score
                no_improve = 0
                lr *= 0.999
                
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_pts.copy()
            else:
                no_improve += 1
                lr *= 0.97
                momentum *= 0.5
                
                # Periodic random restart to escape local minima
                if no_improve % 25 == 0:
                    np.random.seed(int(time.time() * 1000) % 100000)
                    perturb = np.random.normal(0, 0.01, (N, 2))
                    trial = current_pts + perturb
                    trial_score = calculate_score(trial)
                    if trial_score > current_score:
                        current_pts = trial
                        current_score = trial_score
                        no_improve = 0
                        momentum = np.zeros_like(current_pts)
                        if trial_score > best_score:
                            best_score = trial_score
                            best_points = current_pts.copy()
                    
                    if lr < 1e-7:
                        break
        
        # Phase 2: Bottleneck-directed local search
        if time.time() < deadline - 15.0:
            time_budget = min(10.0, deadline - time.time() - 15.0)
            if time_budget > 2.0:
                improved_pts, improved_score = _bottleneck_directed_search(current_pts, time_budget)
                current_pts = improved_pts
                current_score = improved_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_pts.copy()
        
        # Phase 3: High-precision polish
        if time.time() < deadline - 5.0:
            polished, polish_score = _polish(current_pts, max_iter=3000, tol=1e-14)
            if polish_score > best_score:
                best_score = polish_score
                best_points = polished.copy()
    
    # Final global polish on the best points found
    if best_points is not None and time.time() < deadline - 2.0:
        final_polished, final_score = _polish(best_points, max_iter=2000, tol=1e-15)
        if final_score > best_score:
            best_score = final_score
            best_points = final_polished.copy()
    
    if best_points is None:
        return np.random.rand(N, 2)
        
    return best_points
