import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull

N = 14
DEADLINE = 140.0  # Safety margin before 150s kill

def _hull_area(pts):
    """Compute convex hull area."""
    try:
        if len(pts) < 3:
            return 0.0
        # Check for collinearity or duplicates which cause errors
        if np.allclose(pts[:, 0], pts[0, 0]) or np.allclose(pts[:, 1], pts[0, 1]):
            return 0.0
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _min_triangle_area_vec(points):
    """Vectorized minimum triangle area for N points."""
    n = len(points)
    if n < 3:
        return 0.0
    min_area = float('inf')
    x = points[:, 0]
    y = points[:, 1]
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            # Vectors from i to j and i to k
            dx_ji = x[j] - x[i]
            dy_ji = y[j] - y[i]
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            # Cross product magnitude / 2
            cross = dx_ji * dy_ki - dy_ji * dx_ki
            local_min = 0.5 * np.min(np.abs(cross))
            if local_min < min_area:
                min_area = local_min
                if min_area < 1e-14:
                    return min_area
    return min_area

def _score(pts):
    """Compute min triangle area / hull area."""
    hull_a = _hull_area(pts)
    if hull_a <= 0:
        return 0.0
    min_area = _min_triangle_area_vec(pts)
    return min_area / hull_a

def _polar_to_cartesian(polar_params):
    """
    Convert sorted polar parameters (r, theta) to Cartesian coordinates.
    Enforces strictly increasing theta by construction if input is sorted.
    """
    r = polar_params[0::2]
    theta = polar_params[1::2]
    # Ensure theta is in [0, 2*pi) and sorted? 
    # The DE bounds will handle theta ranges. 
    # We assume the input theta is already sorted in the search space? 
    # No, DE searches independent variables. 
    # To enforce "strictly increasing angular order", we can define the search space 
    # as delta_thetas > 0 and a base theta, but that's complex. 
    # Alternative: Just search (r_i, theta_i) and sort by theta before scoring? 
    # The problem doesn't require points to be in a specific order, but sorting 
    # them by angle ensures the convex hull vertices are in cyclic order, which 
    # is good for hull calculation stability and might help the optimizer 
    # avoid self-intersecting "bow-tie" configurations that reduce area.
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    pts = np.column_stack([x, y])
    
    # Sort by angle to maintain cyclic order on the convex hull (if points are in convex position)
    # This is a heuristic to keep the geometry "nice".
    # However, if points are not in convex position (some inside), sorting by angle 
    # doesn't guarantee convex hull order, but it's a good initialization for the hull.
    # Actually, for maximizing min triangle area, we want points spread out.
    # Sorting by angle is safe.
    sort_idx = np.argsort(theta)
    return pts[sort_idx]

def _neg_score_polar(polar_params):
    """Objective for Differential Evolution: minimize -score."""
    pts = _polar_to_cartesian(polar_params)
    s = _score(pts)
    return -s

def _initial_polar_regular(n):
    """
    Generate initial polar parameters for a regular n-gon.
    Returns array of shape (2n,) with [r0, t0, r1, t1, ...]
    """
    angles = 2 * np.pi * np.arange(n) / n
    # Regular n-gon inscribed in circle of radius R has area (n/2) * R^2 * sin(2*pi/n)
    # We want area ~ 1. So R^2 = 2 / (n * sin(2*pi/n))
    R = np.sqrt(2.0 / (n * np.sin(2.0 * np.pi / n)))
    r = np.full(n, R)
    # Interleave r and angles
    params = np.zeros(2 * n)
    params[0::2] = r
    params[1::2] = angles
    return params

def _initial_polar_random(n, seed):
    """
    Generate random initial polar parameters.
    """
    rng = np.random.RandomState(seed)
    # Radii between 0.2 and 1.2 (allowing some outside unit circle, DE will adjust)
    r = rng.uniform(0.2, 1.2, size=n)
    # Angles randomly sorted
    theta = np.sort(rng.uniform(0, 2 * np.pi, size=n))
    params = np.zeros(2 * n)
    params[0::2] = r
    params[1::2] = theta
    return params

def _local_search_cartesian(pts, start_time, deadline):
    """
    Perform local optimization using Nelder-Mead on the exact negative score.
    """
    if time.time() > deadline:
        return pts
    
    x0 = pts.flatten()
    best_pts = pts
    best_score = _score(pts)
    
    # Try Nelder-Mead
    try:
        res = minimize(_neg_score, x0, method='Nelder-Mead',
                       options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
        if res.x is not None:
            cand = res.x.reshape(-1, 2)
            s = _score(cand)
            if s > best_score:
                best_pts = cand
                best_score = s
    except Exception:
        pass
        
    return best_pts

def _neg_score(pts_flat):
    """Negative score for minimization."""
    pts = pts_flat.reshape(-1, 2)
    s = _score(pts)
    return -s

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Bounds for Differential Evolution
    # r: [0.1, 2.0]
    # theta: [0, 2*pi]
    # We will interleave them.
    # DE expects a list of (min, max) tuples.
    bounds = []
    for i in range(N):
        bounds.append((0.1, 2.0))  # r_i
        bounds.append((0.0, 2.0 * np.pi))  # theta_i
    
    best_pts = None
    best_score = -1.0
    
    # Phase 1: Differential Evolution
    # DE is a global stochastic search. It is robust to local minima.
    # We use a population size of 10*N and maxiter 50.
    # This might be slow, so we set a tight maxiter and check time.
    
    # Initial population can be seeded with regular polygon to help DE.
    # DE in scipy accepts 'init' as a numpy array of shape (popsize, dim).
    # Default is random.
    
    popsize = 10
    max_iter = 50
    
    # To ensure we have a good starting point, we run DE.
    # We can also run a few local searches on the best DE result.
    
    # Since DE can be slow, we might want to limit it.
    # Let's try running DE.
    
    try:
        # Seed for reproducibility in DE? DE uses random internally.
        # We can pass a seed via 'seed' parameter in newer scipy versions? 
        # Actually, differential_evolution doesn't take a seed directly in older versions, 
        # but we can set np.random.seed.
        # Let's just run it.
        
        # We will run DE.
        # To avoid timeout, we might need to reduce popsize or maxiter.
        # Let's estimate time: 10*14=140 individuals, 50 iterations.
        # Each evaluation is O(N^2) ~ 196 ops. Very fast.
        # 140 * 50 = 7000 evaluations.
        # 7000 * 1us = 7ms. So DE should be fast.
        
        # However, DE convergence might be slow.
        # Let's run it.
        
        res = differential_evolution(_neg_score_polar, bounds, 
                                     maxiter=max_iter, 
                                     popsize=popsize, 
                                     tol=1e-4, 
                                     mutation=(0.5, 1.0), 
                                     crossover=0.9, 
                                     seed=42)
        
        if res.success or res.x is not None:
            best_polar = res.x
            best_pts = _polar_to_cartesian(best_polar)
            best_score = _score(best_pts)
            
    except Exception as e:
        print(f"DE failed: {e}")
    
    # If DE didn't find anything good, or to improve further, use local search.
    if best_pts is None or best_score < 0.1:
        # Fallback to regular polygon
        polar0 = _initial_polar_regular(N)
        best_pts = _polar_to_cartesian(polar0)
        best_score = _score(best_pts)
        
    # Phase 2: Local Search Polish
    # Convert best DE result to Cartesian and run Nelder-Mead.
    # We might run multiple local searches from different points in the DE population?
    # DE only returns the best.
    # Let's run local search on the best DE result.
    
    if time.time() < deadline - 5:
        improved = _local_search_cartesian(best_pts, start_time, deadline)
        s = _score(improved)
        if s > best_score:
            best_pts = improved
            best_score = s
            
    # Phase 3: Try a few random restarts of local search if time permits
    # This helps if DE got stuck in a local minimum.
    
    # We can generate a few random polar configurations, convert to Cartesian, and polish.
    for i in range(5):
        if time.time() > deadline - 10:
            break
        
        polar_rand = _initial_polar_random(N, seed=100 + i)
        pts_rand = _polar_to_cartesian(polar_rand)
        
        # Polish
        improved = _local_search_cartesian(pts_rand, start_time, deadline)
        s = _score(improved)
        
        if s > best_score:
            best_pts = improved
            best_score = s
            
    # Final check
    if best_pts is None:
        # Absolute fallback
        angles = 2 * np.pi * np.arange(N) / N
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return best_pts

# EVOLVE-BLOCK-END
