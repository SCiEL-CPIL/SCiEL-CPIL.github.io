import numpy as np
from scipy.spatial import ConvexHull
import random
import time
import math

N = 13

# EVOLVE-BLOCK-START
def _triangle_areas(pts):
    """Compute areas of all triangles formed by triples of points."""
    n = len(pts)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Area = 0.5 * |cross(p_j - p_i, p_k - p_i)|
                v1 = pts[j] - pts[i]
                v2 = pts[k] - pts[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                areas.append(area)
    return np.array(areas)

def _convex_hull_area(pts):
    """Compute area of convex hull of points."""
    if len(pts) < 3:
        return 0.0
    try:
        hull = ConvexHull(pts)
        return hull.volume # In 2D, volume is area
    except Exception:
        # Fallback if points are collinear or degenerate
        return 0.0

def _get_score(pts):
    """Calculate the raw score: min triangle area / convex hull area."""
    areas = _triangle_areas(pts)
    hull_area = _convex_hull_area(pts)
    if hull_area < 1e-9:
        return 0.0
    min_area = np.min(areas)
    return min_area / hull_area

def _smoothed_score(pts, beta):
    """
    Calculate a smoothed version of the score to allow local optimization.
    Uses log-sum-exp to approximate min area.
    Score approx = (1/beta * log(sum(exp(-beta * areas)))) / hull_area
    We want to maximize this, so we minimize the negative.
    """
    areas = _triangle_areas(pts)
    hull_area = _convex_hull_area(pts)
    if hull_area < 1e-9:
        return 0.0
    
    # Soft min: min(x) ~ (1/beta) * log(sum(exp(-beta * x)))
    # To avoid overflow, shift by max
    max_val = np.max(areas)
    # exp(-beta * (areas - max_val)) is safe
    shifted = -beta * (areas - max_val)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min_area = (max_val - log_sum / beta)
    
    return soft_min_area / hull_area

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    # 1. Initialization: Regular 13-gon
    # This is a strong candidate for the global optimum structure
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    # Add a small perturbation to break symmetry slightly and help SA
    rng = np.random.default_rng(42)
    initial_pts = np.column_stack((np.cos(angles), np.sin(angles)))
    initial_pts += rng.normal(0, 0.05, initial_pts.shape)
    
    best_pts = initial_pts.copy()
    best_score = _get_score(best_pts)
    
    # Precompute triangle indices for faster area calculation if needed, 
    # but N=13 is small, direct calculation is fine.
    
    # 2. Simulated Annealing
    current_pts = initial_pts.copy()
    current_score = best_score
    T = 0.1 # Initial temperature
    T_min = 1e-6
    cooling_rate = 0.9995
    step_size = 0.1
    
    # Track time for local polish frequency
    last_polish_time = start_time
    polish_interval = 5.0 # seconds
    
    iterations = 0
    max_iterations = 200000
    
    while time.time() < deadline and iterations < max_iterations:
        iterations += 1
        
        # Check if we should do a local polish
        if time.time() - last_polish_time > polish_interval:
            # Local Polish using Smoothed Objective
            # We minimize -smoothed_score
            def objective_flat(flat_pts):
                pts = flat_pts.reshape(-1, 2)
                # Beta annealing: start low, increase
                # Using a fixed beta for the polish phase is simpler
                beta = 50.0
                return -_smoothed_score(pts, beta)
            
            try:
                res = scipy_optimize_minimize(
                    objective_flat, 
                    current_pts.flatten(), 
                    method='Nelder-Mead', 
                    options={'xatol': 1e-4, 'fatol': 1e-6, 'maxiter': 200}
                )
                new_pts = res.x.reshape(-1, 2)
                new_score = _get_score(new_pts)
                if new_score > current_score:
                    current_pts = new_pts
                    current_score = new_score
                if new_score > best_score:
                    best_pts = new_pts.copy()
                    best_score = new_score
            except Exception:
                pass
            
            last_polish_time = time.time()
        
        # SA Step
        # Perturb a random point
        idx = rng.integers(0, N)
        move = rng.normal(0, step_size, 2)
        new_pts = current_pts.copy()
        new_pts[idx] += move
        
        new_score = _get_score(new_pts)
        
        # Acceptance criterion
        delta = new_score - current_score
        if delta > 0 or (T > T_min and rng.random() < math.exp(delta / T)):
            current_pts = new_pts
            current_score = new_score
            if new_score > best_score:
                best_pts = new_pts.copy()
                best_score = new_score
        
        # Cooling
        T *= cooling_rate
        # Decay step size slowly
        step_size *= 0.9999
        
        # If T is too low, reset for a few kicks? 
        # Or just let it run. The polish handles fine tuning.
        
    # Final Polish on the best points found
    if time.time() < deadline - 10:
        def final_objective_flat(flat_pts):
            pts = flat_pts.reshape(-1, 2)
            beta = 100.0 # Higher beta for sharper min approximation
            return -_smoothed_score(pts, beta)
        
        try:
            res = scipy_optimize_minimize(
                final_objective_flat, 
                best_pts.flatten(), 
                method='Powell', # Powell is often better for this
                options={'xatol': 1e-5, 'fatol': 1e-7, 'maxiter': 500}
            )
            final_pts = res.x.reshape(-1, 2)
            final_score = _get_score(final_pts)
            if final_score > best_score:
                best_pts = final_pts
                best_score = final_score
        except Exception:
            pass
            
    return best_pts

# Import scipy optimize inside function or at top? 
# Top level is better for speed, but task says "standard library, NumPy, SciPy available".
# Let's import at top to be safe.
import scipy.optimize as scipy_optimize

# EVOLVE-BLOCK-END
