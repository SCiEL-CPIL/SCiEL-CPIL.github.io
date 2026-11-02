"""Heilbronn convex regions: maximize min triangle area over convex hull area for 13 points."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull
from itertools import combinations

_DEADLINE = 150.0
_START = time.time()
_SAFETY = 10.0  # seconds before deadline to stop

# Pre-compute triangle indices
TRI_INDICES = np.array(list(combinations(range(N), 3)))
NUM_TRI = len(TRI_INDICES)

def _min_triangle_area_exact(pts):
    """Minimum area of any triangle formed by 3 of the points. Vectorized."""
    p_i = pts[TRI_INDICES[:, 0]]
    p_j = pts[TRI_INDICES[:, 1]]
    p_k = pts[TRI_INDICES[:, 2]]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    return np.min(areas)

def _convex_hull_area(pts):
    """Area of the convex hull of the points."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score(pts):
    """min triangle area / convex hull area."""
    hull_area = _convex_hull_area(pts)
    if hull_area <= 0:
        return 0.0
    return _min_triangle_area_exact(pts) / hull_area

def _neg_score(x):
    """Negative score for minimization."""
    pts = x.reshape(N, 2)
    return -_score(pts)

def _local_polish(x0, max_iter=1000):
    """
    Perform a local search using Nelder-Mead to refine the starting point x0.
    Returns the best x found.
    """
    try:
        res = minimize(_neg_score, x0, method='Nelder-Mead', 
                       options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-8, 'disp': False})
        return res.x
    except Exception:
        return x0

def _generate_initial_points():
    """
    Generate a diverse set of initial points for the population.
    Returns a list of point arrays (N, 2).
    """
    inits = []
    
    # 1. Regular 13-gon
    angles = 2 * np.pi * np.arange(N) / N
    pts_reg = np.column_stack([0.5 + 0.4 * np.cos(angles), 0.5 + 0.4 * np.sin(angles)])
    inits.append(pts_reg)
    
    # 2. Regular 12-gon + Center
    angles_12 = 2 * np.pi * np.arange(12) / 12
    pts_12 = np.column_stack([0.5 + 0.4 * np.cos(angles_12), 0.5 + 0.4 * np.sin(angles_12)])
    pts_12_c = np.vstack([pts_12, np.array([0.5, 0.5])])
    inits.append(pts_12_c)
    
    # 3. Jittered Grid (4x4 grid minus 3 points, scaled)
    # A grid often provides good separation.
    # 4x4 = 16 points. Remove 3.
    grid_x = np.linspace(0.1, 0.9, 4)
    grid_y = np.linspace(0.1, 0.9, 4)
    gx, gy = np.meshgrid(grid_x, grid_y)
    grid_pts = np.column_stack([gx.ravel(), gy.ravel()])
    # Remove 3 random points to get 13
    rng = np.random.RandomState(123)
    indices_to_keep = list(range(16))
    removed = rng.choice(16, 3, replace=False)
    indices_to_keep = [i for i in indices_to_keep if i not in removed]
    pts_grid = grid_pts[indices_to_keep]
    inits.append(pts_grid)
    
    # 4. Random points in unit square
    rng = np.random.RandomState(42)
    for _ in range(5):
        pts_rand = rng.rand(N, 2) * 0.8 + 0.1
        inits.append(pts_rand)
        
    return inits

def construct():
    global _START, _DEADLINE
    elapsed = time.time() - _START
    remaining = _DEADLINE - _SAFETY - elapsed
    
    if remaining < 10.0:
        # Fallback: return a regular polygon
        angles = 2 * np.pi * np.arange(N) / N
        return np.column_stack([0.5 + 0.4 * np.cos(angles), 0.5 + 0.4 * np.sin(angles)])

    # Define bounds for Differential Evolution
    # Points are in [0, 1] x [0, 1]
    bounds = [(0.0, 1.0)] * (2 * N)
    
    best_x = None
    best_score = -np.inf
    
    # Generate initial points to seed the population or for local polish
    inits = _generate_initial_points()
    
    # Evaluate initial points to have a baseline
    for init in inits:
        s = _score(init)
        if s > best_score:
            best_score = s
            best_x = init.flatten()
            
    # Run Differential Evolution
    # We want to maximize score, so minimize -score
    # DE is robust to non-smoothness.
    # maxiter and popsize are tuned for the 150s budget.
    # 2N = 26 dimensions.
    # popsize=15 means 15*26 = 390 evaluations per generation.
    # maxiter=100 means 39000 evaluations.
    # Each evaluation is vectorized but involves convex hull. 
    # ConvexHull on 13 points is fast (~microseconds).
    # Let's be conservative to avoid timeout.
    
    # We will run DE in a loop, checking time.
    # Instead of one long DE run, we can run shorter runs and keep the best.
    # Or just one run with a tight maxiter.
    
    # Let's try a single DE run first. If it's too slow, we rely on the timeout guard.
    # To be safe, we'll use a wrapper that checks time? No, DE doesn't support callbacks easily for stopping.
    # We must estimate time.
    # Let's run DE with maxiter=50, popsize=10.
    # 50 * 10 * 26 = 13000 evaluations.
    # If each eval is 0.1ms, that's 1.3s. If 1ms, that's 13s.
    # ConvexHull is likely the bottleneck. Let's assume ~0.5ms per eval.
    # 13000 * 0.5ms = 6.5s. This is safe.
    
    try:
        de_result = differential_evolution(
            _neg_score, 
            bounds, 
            seed=42, 
            maxiter=100, 
            popsize=15,
            tol=1e-6,
            polish=False, # We will do our own polish
            init='latinhypercube' # Good spread
        )
        
        if de_result.success or de_result.fun < best_score * -1: # de_result.fun is -score
            candidate_x = de_result.x
            candidate_score = -de_result.fun
            if candidate_score > best_score:
                best_score = candidate_score
                best_x = candidate_x
    except Exception as e:
        # If DE fails or times out internally (though it shouldn't with maxiter), continue
        pass

    # Check time before local polish
    elapsed = time.time() - _START
    if time.time() - _START > _DEADLINE - _SAFETY - 5.0:
        return best_x.reshape(N, 2)

    # Local Polish with Nelder-Mead
    # Nelder-Mead is effective for final refinement of the minimum.
    # It handles the non-differentiability better than gradient methods in local regions.
    
    # We will polish the best_x found so far.
    # We might also polish the best initial points if they were better.
    
    polish_candidates = [best_x]
    # Add a few other good candidates if we have time
    for init in inits[:3]:
        polish_candidates.append(init.flatten())
        
    for x_cand in polish_candidates:
        if time.time() - _START > _DEADLINE - _SAFETY - 2.0:
            break
            
        polished_x = _local_polish(x_cand, max_iter=500)
        polished_score = -_neg_score(polished_x)
        
        if polished_score > best_score:
            best_score = polished_score
            best_x = polished_x

    return best_x.reshape(N, 2)
# EVOLVE-BLOCK-END
