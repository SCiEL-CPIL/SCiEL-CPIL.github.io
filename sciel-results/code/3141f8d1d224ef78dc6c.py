import numpy as np
import itertools
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 13
DEADLINE = 110.0  # Safety margin from 120s

def get_triangle_areas(points):
    """Calculate areas of all triangles formed by triples of points."""
    n = len(points)
    areas = []
    # Vectorized approach for speed
    # We can use broadcasting, but 13 is small enough that a loop or itertools is fine.
    # Let's use a slightly vectorized inner loop or just itertools for clarity and safety.
    for i, j, k in itertools.combinations(range(n), 3):
        p1 = points[i]
        p2 = points[j]
        p3 = points[k]
        # Area = 0.5 * | (p2-p1) x (p3-p1) |
        v1 = p2 - p1
        v2 = p3 - p1
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        areas.append(0.5 * abs(cross))
    return np.array(areas)

def get_hull_area(points):
    """Calculate the area of the convex hull of the points."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 1e-10

def exact_score(points):
    """Calculate the exact score: min triangle area / hull area."""
    areas = get_triangle_areas(points)
    min_area = np.min(areas)
    hull_area = get_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    return min_area / hull_area

def soft_min_objective(x, beta):
    """
    Objective function for optimization.
    Minimizes the negative of the soft-min score.
    x is a flattened array of 2*N coordinates.
    """
    points = x.reshape(-1, 2)
    areas = get_triangle_areas(points)
    
    # Soft min approximation: -1/beta * log(sum(exp(-beta * areas)))
    # To avoid overflow/underflow, we can shift by min area
    min_a = np.min(areas)
    # If min_a is very large, exp(-beta*(a-min_a)) is safe.
    # If min_a is 0, it's fine.
    exp_terms = np.exp(-beta * (areas - min_a))
    sum_exp = np.sum(exp_terms)
    soft_min_val = min_a - (1.0 / beta) * np.log(sum_exp)
    
    hull_area = get_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    
    score = soft_min_val / hull_area
    return -score

def construct():
    start_time = time.time()
    best_score = 0.0
    best_points = np.random.rand(N, 2) * 2 - 1  # Initialize with random points
    
    # Pre-compute combinations for efficiency if needed, but get_triangle_areas handles it.
    
    restarts = 0
    while time.time() - start_time < DEADLINE:
        restarts += 1
        
        # 1. Generate initial points
        # Strategy A: Uniform in square
        # Strategy B: Concentric rings (often better for this problem)
        if restarts % 2 == 0:
            # Concentric rings
            # e.g., 5 points on outer ring, 5 on middle, 3 on inner
            # Or just random radii and angles
            angles = np.linspace(0, 2*np.pi, N, endpoint=False) + np.random.uniform(0, 0.5)
            # Random radii to break symmetry
            radii = np.random.uniform(0.2, 1.0, N)
            # Sort by angle to avoid crossing? No, just place them.
            # Actually, random angles are better to avoid grid-like artifacts
            angles = np.sort(np.random.uniform(0, 2*np.pi, N))
            points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        else:
            # Uniform in square
            points = np.random.uniform(-1, 1, (N, 2))
        
        x0 = points.flatten()
        
        # 2. Optimization Phase 1: Soft-min with L-BFGS-B
        # Start with low beta for broad basin, then increase
        betas = [20, 50, 100, 200]
        current_x = x0
        
        for beta in betas:
            # Check time
            if time.time() - start_time > DEADLINE - 10:
                break
                
            try:
                res = minimize(
                    soft_min_objective, 
                    current_x, 
                    args=(beta,), 
                    method='L-BFGS-B', 
                    options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-8}
                )
                current_x = res.x
            except Exception:
                pass
        
        # 3. Optimization Phase 2: Polish with Nelder-Mead on exact objective
        # Nelder-Mead doesn't use gradients, good for non-smooth min
        if time.time() - start_time < DEADLINE - 5:
            try:
                res_nm = minimize(
                    lambda x: -exact_score(x.reshape(-1, 2)), 
                    current_x, 
                    method='Nelder-Mead', 
                    options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
                )
                current_x = res_nm.x
            except Exception:
                pass
        
        # 4. Evaluate and update best
        final_points = current_x.reshape(-1, 2)
        current_score = exact_score(final_points)
        
        if current_score > best_score:
            best_score = current_score
            best_points = final_points.copy()
            
        # Optional: Small random perturbations around best point for a few iterations
        # This can help escape local minima if Nelder-Mead got stuck.
        # But given time constraints, we rely on restarts.
        
    return best_points

# EVOLVE-BLOCK-START
# The construct function above is the implementation.
# We need to ensure the interface matches.
# The prompt says "Keep the function names and signatures of the seed program."
# Seed program:
# def construct():
#     ...
#     return np.array(...)
#
# My construct() returns best_points which is an np.array of shape (N, 2).
# This matches.

# EVOLVE-BLOCK-END
