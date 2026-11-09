import numpy as np
from scipy.spatial import ConvexHull
import time
from scipy.optimize import differential_evolution, minimize
from itertools import combinations

N = 14

def get_score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    points: np.array of shape (N, 2)
    """
    # 1. Convex Hull Area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-10:
            return 0.0
    except Exception:
        return 0.0

    # 2. Min Triangle Area (Vectorized)
    # Generate all unique triples (i, j, k) with i < j < k
    i_idx, j_idx, k_idx = np.indices((N, N, N))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_flat = i_idx[mask]
    j_flat = j_idx[mask]
    k_flat = k_idx[mask]
    
    p1 = points[i_flat]
    p2 = points[j_flat]
    p3 = points[k_flat]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    if min_area < 1e-10:
        return 0.0
        
    return min_area / hull_area

def fast_score(points):
    """
    Fast score calculation for inner optimization loops.
    Uses the same logic but with minimal overhead.
    """
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-10:
            return 0.0
    except Exception:
        return 0.0

    i_idx, j_idx, k_idx = np.indices((N, N, N))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_flat = i_idx[mask]
    j_flat = j_idx[mask]
    k_flat = k_idx[mask]
    
    p1 = points[i_flat]
    p2 = points[j_flat]
    p3 = points[k_flat]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    if min_area < 1e-10:
        return 0.0
        
    return min_area / hull_area

def local_refine(points, deadline, max_iter=2000):
    """
    Local refinement using coordinate descent with adaptive step sizes.
    This is more effective than Nelder-Mead for this type of problem.
    """
    current_points = points.copy()
    current_score = get_score(current_points)
    best_points = current_points.copy()
    best_score = current_score
    
    step_sizes = np.ones(N) * 0.05
    no_improve_count = 0
    
    for iteration in range(max_iter):
        if time.time() > deadline - 2.0:
            break
        
        improved = False
        
        # Try moving each point
        for i in range(N):
            if time.time() > deadline - 2.0:
                break
            
            best_local_point = current_points[i].copy()
            best_local_score = current_score
            
            # Try different step sizes and directions
            for step_mult in [1.0, 0.5, 0.25, 2.0]:
                step = step_sizes[i] * step_mult
                for _ in range(8):
                    angle = np.random.uniform(0, 2 * np.pi)
                    dx = step * np.cos(angle)
                    dy = step * np.sin(angle)
                    
                    trial_points = current_points.copy()
                    trial_points[i] = current_points[i] + np.array([dx, dy])
                    trial_score = fast_score(trial_points)
                    
                    if trial_score > best_local_score:
                        best_local_score = trial_score
                        best_local_point = trial_points[i].copy()
                        improved = True
            
            if improved:
                current_points[i] = best_local_point
                current_score = best_local_score
                step_sizes[i] *= 1.1  # Increase step size slightly
                no_improve_count = 0
                
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_points.copy()
            else:
                step_sizes[i] *= 0.7  # Decrease step size
                no_improve_count += 1
        
        # If no improvement in a full cycle, reduce all step sizes
        if not improved:
            step_sizes *= 0.5
            no_improve_count += 1
            
            if no_improve_count > 20:
                break
    
    return best_points, best_score

def construct():
    start_time = time.time()
    # The evaluator kills after 150s. We keep a safety margin.
    deadline = start_time + 145.0
    
    # Initial guess: Regular N-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)
    initial_points = np.column_stack((x, y))
    
    best_points = initial_points.copy()
    best_score = get_score(best_points)
    
    # Objective function for minimizers (maximize score -> minimize -score)
    def objective(flat_points):
        pts = flat_points.reshape(N, 2)
        return -fast_score(pts)
        
    # 1. Global Search using Differential Evolution
    # Bounds: [-2, 2] is sufficient for normalized convex hull area ~1
    bounds = [(-2, 2)] * (2 * N)
    
    try:
        # Use DE for global exploration
        res = differential_evolution(
            objective, 
            bounds, 
            seed=42,
            maxiter=30, # Reduced to leave more time for local refinement
            popsize=12,
            tol=0.01,
            polish=False
        )
        
        if res.fun < -best_score:
            best_score = -res.fun
            best_points = res.x.reshape(N, 2)
            
    except Exception:
        pass
        
    # Check time
    if time.time() > deadline - 30.0:
        # Not enough time for local refinement, return what we have
        return best_points

    # 2. Local Refinement using coordinate descent
    # This is more effective for this problem
    best_points, best_score = local_refine(best_points, deadline)
    
    # Check time
    if time.time() > deadline - 10.0:
        return best_points

    # 3. Second round of DE from the refined point (if time allows)
    if time.time() < deadline - 20.0:
        try:
            # Create tighter bounds around the current best
            center = best_points.mean(axis=0)
            spread = np.std(best_points, axis=0)
            tight_bounds = []
            for i in range(2 * N):
                idx = i // 2
                coord = i % 2
                lower = best_points[idx, coord] - 2 * spread[coord] - 0.5
                upper = best_points[idx, coord] + 2 * spread[coord] + 0.5
                tight_bounds.append((lower, upper))
            
            res2 = differential_evolution(
                objective, 
                tight_bounds, 
                seed=123,
                maxiter=20,
                popsize=10,
                tol=0.01,
                polish=False
            )
            
            if res2.fun < -best_score:
                best_score = -res2.fun
                best_points = res2.x.reshape(N, 2)
                
        except Exception:
            pass
    
    # Final local refinement
    if time.time() < deadline - 5.0:
        best_points, best_score = local_refine(best_points, deadline)

    return best_points
