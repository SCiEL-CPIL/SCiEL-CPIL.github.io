import numpy as np
import time
import scipy.spatial as spat
import itertools

N = 13

def calculate_score(points):
    """
    Calculate the minimum triangle area divided by the convex hull area.
    Returns 0 if points are collinear or hull area is 0.
    """
    if len(points) < 3:
        return 0.0
    
    # Compute convex hull area
    try:
        hull = spat.ConvexHull(points)
        hull_area = hull.volume # In 2D, volume is area
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0

    pts = points
    n = len(pts)
    
    # Efficient triangle area calculation
    # Pre-computed indices for C(13,3) = 286 triangles
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_flat = i_idx[mask]
    j_flat = j_idx[mask]
    k_flat = k_idx[mask]
    
    v1 = pts[j_flat] - pts[i_flat] 
    v2 = pts[k_flat] - pts[i_flat]
    
    # Determinant in 2D
    det = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(det)
    
    min_area = np.min(areas)
    
    return min_area / hull_area

def find_bottleneck(points):
    """
    Returns the indices (i, j, k) of the triangle with the minimum area,
    and the area itself.
    """
    pts = points
    n = len(pts)
    
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_flat = i_idx[mask]
    j_flat = j_idx[mask]
    k_flat = k_idx[mask]
    
    v1 = pts[j_flat] - pts[i_flat] 
    v2 = pts[k_flat] - pts[i_flat]
    
    det = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(det)
    
    min_idx = np.argmin(areas)
    return i_flat[min_idx], j_flat[min_idx], k_flat[min_idx], areas[min_idx]

def _local_polish(points, max_iter=50, deadline=None):
    """
    Local refinement using a simple hill-climbing step on the bottleneck triangle.
    """
    current_pts = points.copy()
    current_score = calculate_score(current_pts)
    step_size = 0.05
    
    for _ in range(max_iter):
        if deadline and time.time() > deadline:
            break
        
        i, j, k, _ = find_bottleneck(current_pts)
        
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
        
        trial_pts = current_pts.copy()
        trial_pts[i] += step_size * dir_i
        trial_pts[j] += step_size * dir_j
        trial_pts[k] += step_size * dir_k
        
        trial_score = calculate_score(trial_pts)
        
        if trial_score > current_score:
            current_pts = trial_pts
            current_score = trial_score
        else:
            step_size *= 0.9
            if step_size < 1e-4:
                break
                
    return current_pts, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s safety margin
    
    best_score = -1.0
    best_points = None
    
    # Differential Evolution parameters
    pop_size = 30
    F = 0.8  # Scaling factor
    CR = 0.9  # Crossover rate
    
    # Initialize population with diverse structures
    # 1. Regular polygon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    base_poly = np.column_stack((np.cos(angles), np.sin(angles)))
    
    population = []
    # Seed with regular polygon and perturbations
    for i in range(pop_size):
        if i == 0:
            pop = base_poly.copy()
        elif i == 1:
            # Fibonacci lattice
            golden_angle = np.pi * (3.0 - np.sqrt(5.0))
            fib_pts = np.zeros((N, 2))
            for j in range(N):
                r = np.sqrt((j + 0.5) / N)
                theta = golden_angle * j
                fib_pts[j] = [r * np.cos(theta), r * np.sin(theta)]
            pop = fib_pts
        else:
            # Random perturbation of regular polygon
            noise = np.random.normal(0, 0.2, (N, 2))
            pop = base_poly + noise
            
        # Normalize scale (score is scale invariant, but helps numerical stability)
        # Center at origin
        pop -= np.mean(pop, axis=0)
        
        score = calculate_score(pop)
        population.append((pop, score))
        
        if score > best_score:
            best_score = score
            best_points = pop.copy()

    # Differential Evolution Loop
    while time.time() < deadline - 30.0: # Keep 30s for final polish and safety
        
        for i in range(pop_size):
            if time.time() > deadline - 30.0:
                break
                
            # Select 3 distinct individuals other than i
            idxs = [j for j in range(pop_size) if j != i]
            r1, r2, r3 = np.random.choice(idxs, 3, replace=False)
            
            x1 = population[r1][0]
            x2 = population[r2][0]
            x3 = population[r3][0]
            xi = population[i][0]
            
            # Generate mutant vector
            v = x1 + F * (x2 - x3)
            
            # Recombination (Crossover)
            trial = np.zeros_like(xi)
            j_rand = np.random.randint(0, N)
            for j in range(N):
                if np.random.rand() < CR or j == j_rand:
                    trial[j] = v[j]
                else:
                    trial[j] = xi[j]
            
            # Normalize trial (centering)
            trial -= np.mean(trial, axis=0)
            
            # Evaluate
            trial_score = calculate_score(trial)
            
            # Selection
            if trial_score > population[i][1]:
                population[i] = (trial, trial_score)
                
                if trial_score > best_score:
                    best_score = trial_score
                    best_points = trial.copy()
                    
    # Final Polish on the best point found
    if best_points is not None and time.time() < deadline - 5.0:
        polished_pts, polished_score = _local_polish(best_points, max_iter=100, deadline=deadline - 5.0)
        if polished_score > best_score:
            best_score = polished_score
            best_points = polished_pts
            
    if best_points is None:
        return np.random.rand(N, 2)
        
    return best_points
