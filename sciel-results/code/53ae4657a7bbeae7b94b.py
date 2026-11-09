import numpy as np
from scipy.spatial import ConvexHull
import time
from scipy.optimize import differential_evolution, minimize

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
        return -get_score(pts)
        
    # 1. Global Search using Differential Evolution
    # Bounds: [-2, 2] is sufficient for normalized convex hull area ~1
    bounds = [(-2, 2)] * (2 * N)
    
    try:
        # Use DE for global exploration
        # maxiter is limited by time, but we check time explicitly in the loop if we were doing it manually.
        # DE has a timeout option? No, but we can wrap it or just let it run and check later.
        # Better to use a moderate popsize and maxiter, then check time.
        
        res = differential_evolution(
            objective, 
            bounds, 
            seed=42,
            maxiter=50, # Keep low to avoid timeout, rely on local refine
            popsize=15,
            tol=0.01,
            polish=False # We will do our own polish
        )
        
        if res.fun < -best_score:
            best_score = -res.fun
            best_points = res.x.reshape(N, 2)
            
    except Exception:
        pass
        
    # Check time
    if time.time() > deadline - 10.0:
        return best_points

    # 2. Local Refinement using Nelder-Mead
    # Start from the best point found by DE (or initial if DE failed)
    current_points = best_points.copy()
    
    try:
        res_nm = minimize(
            objective, 
            current_points.flatten(), 
            method='Nelder-Mead', 
            options={'xatol': 1e-5, 'fatol': 1e-7, 'maxiter': 5000}
        )
        if res_nm.fun < -best_score:
            best_score = -res_nm.fun
            best_points = res_nm.x.reshape(N, 2)
    except Exception:
        pass

    # Check time
    if time.time() > deadline - 5.0:
        return best_points

    # 3. Final Polish with Random Walk (Simulated Annealing-like)
    # This helps escape tiny local minima or refine further
    current_points = best_points.copy()
    current_score = best_score
    step_size = 0.01
    T = 0.01 # Initial temperature for acceptance
    
    while time.time() < deadline - 1.0:
        improved = False
        # Try moving a random point
        i = np.random.randint(N)
        new_point = current_points[i] + np.random.normal(0, step_size, 2)
        trial_points = current_points.copy()
        trial_points[i] = new_point
        trial_score = get_score(trial_points)
        
        # Accept if better, or with probability exp(-delta/T) if worse (SA)
        if trial_score > current_score:
            current_points = trial_points
            current_score = trial_score
            improved = True
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
        else:
            delta = current_score - trial_score
            if delta > 0:
                prob = np.exp(-delta / T)
                if np.random.random() < prob:
                    current_points = trial_points
                    current_score = trial_score
        
        # Cool down
        T *= 0.999
        if T < 1e-6:
            T = 1e-6
            step_size *= 0.95
            if step_size < 1e-5:
                break

    return best_points
