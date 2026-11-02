import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize
import time

N = 13
# Pre-compute indices for all triangles
indices = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            indices.append((i, j, k))
indices = np.array(indices)
I, J, K = indices[:, 0], indices[:, 1], indices[:, 2]

def _score_points(pts):
    """
    Compute the score: min triangle area / convex hull area.
    pts: (N, 2) array
    """
    if len(pts) < 3:
        return 0.0
    
    # Calculate Convex Hull Area
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
        if hull_area <= 1e-10:
            return 0.0
    except Exception:
        return 0.0

    # Calculate Triangle Areas Vectorized
    # Area = 0.5 * | (Pj - Pi) x (Pk - Pi) |
    # In 2D, cross product magnitude is x1*y2 - y1*x2
    Pi = pts[I]
    Pj = pts[J]
    Pk = pts[K]
    
    v1 = Pj - Pi
    v2 = Pk - Pi
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    return min_area / hull_area

def _neg_score(pts):
    """
    Negative score for minimization algorithms.
    Reshape flat array to (N, 2).
    """
    pts_2d = pts.reshape(-1, 2)
    score = _score_points(pts_2d)
    # Penalize if score is 0 (degenerate)
    if score == 0.0:
        return 1.0
    return -score

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Initial guess: Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    init_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    init_flat = init_pts.flatten()
    
    best_score = _score_points(init_pts)
    best_pts = init_pts.copy()
    
    # Phase 1: Differential Evolution (Global Search)
    # Bounds: Since the problem is scale invariant, we can constrain points to [-1, 1] or similar.
    # However, DE might find a better scale/shape.
    # Let's constrain to [-2, 2] to allow some flexibility, though normalization makes scale irrelevant.
    # Actually, fixing the scale helps convergence. Let's fix the max norm or just let it be.
    # Using bounds [-1, 1] is a good heuristic for normalized problems.
    bounds = [(-1, 1)] * (2 * N)
    
    # DE parameters
    # We need to be careful with the number of evaluations.
    # N=13 -> 26 variables.
    # Popsize 15 -> 390 individuals.
    # Maxiter 20 -> ~8000 evaluations.
    # Each evaluation is ~286 cross products + ConvexHull.
    # ConvexHull is the bottleneck.
    
    try:
        result_de = differential_evolution(
            _neg_score,
            bounds,
            x0=init_flat,
            seed=42,
            maxiter=50,
            popsize=15,
            tol=1e-8,
            polish=False, # We will polish manually
            init='latinhypercube'
        )
        
        if result_de.fun < -best_score:
            best_pts = result_de.x.reshape(N, 2)
            best_score = -result_de.fun
            
    except Exception:
        pass

    # Check time
    if time.time() > deadline - 20:
        return best_pts

    # Phase 2: Local Refinement
    # Use Powell or Nelder-Mead
    # Powell is often better for non-smooth functions
    try:
        result_lm = minimize(
            _neg_score,
            best_pts.flatten(),
            method='Powell',
            options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10, 'disp': False}
        )
        
        if result_lm.fun < -best_score:
            best_pts = result_lm.x.reshape(N, 2)
            best_score = -result_lm.fun
            
    except Exception:
        pass

    # Check time
    if time.time() > deadline - 5:
        return best_pts

    # Phase 3: High-precision polish with Nelder-Mead
    try:
        result_nm = minimize(
            _neg_score,
            best_pts.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12, 'disp': False}
        )
        
        if result_nm.fun < -best_score:
            best_pts = result_nm.x.reshape(N, 2)
            best_score = -result_nm.fun
            
    except Exception:
        pass

    return best_pts

# EVOLVE-BLOCK-START
# (The code above is the implementation. 
# The prompt asks to keep function names and signatures.
# The current program has `construct()` inside EVOLVE-BLOCK.
# I will place the helper functions and logic inside the block or before it?
# The instruction says: "Keep the function names and signatures of the seed program."
# Seed program:
# N = 13
# # EVOLVE-BLOCK-START
# import random
# import numpy as np
# def construct():
#     ...
# # EVOLVE-BLOCK-END
#
# I should output the complete program.
# I will put imports at the top, and the logic in the block.
# )
# EVOLVE-BLOCK-END
