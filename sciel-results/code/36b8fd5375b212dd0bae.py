import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull

N = 13

def get_triangle_indices(n):
    idx = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                idx.append((i, j, k))
    return np.array(idx)

TRI_IDX = get_triangle_indices(N)
NUM_TRI = len(TRI_IDX)

def min_triangle_area(points):
    """Calculate the minimum triangle area for a set of points."""
    p0 = points[TRI_IDX[:, 0]]
    p1 = points[TRI_IDX[:, 1]]
    p2 = points[TRI_IDX[:, 2]]
    v1 = p1 - p0
    v2 = p2 - p0
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def convex_hull_area(points):
    """Calculate the area of the convex hull."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 1e-10

def score(points):
    """Calculate the score: min triangle area / convex hull area."""
    a_min = min_triangle_area(points)
    a_hull = convex_hull_area(points)
    if a_hull < 1e-10:
        return 0.0
    return a_min / a_hull

def smooth_surrogate(x):
    """
    Objective function for Differential Evolution.
    Minimizes - (min_area / hull_area).
    Uses a log-sum-exp approximation for the minimum to make it smooth.
    We maximize min_area/hull_area, so we minimize -(min_area/hull_area).
    
    To make it smooth:
    min(areas) approx -1/k * log(sum(exp(-k*areas)))
    We want to maximize this. So we minimize - ( -1/k * log(sum(exp(-k*areas))) / hull_area )
    = (1/k * log(sum(exp(-k*areas))) - log(hull_area))
    
    Actually, let's stick to the exact score for DE but it's non-differentiable.
    DE doesn't need gradients. It works well with non-smooth functions too, 
    but a smooth surrogate might help convergence. 
    However, the exact score is the ground truth. 
    Let's use the exact score for DE, as it is robust.
    """
    pts = x.reshape(-1, 2)
    # Add a small penalty if points are degenerate or hull is invalid
    a_hull = convex_hull_area(pts)
    if a_hull < 1e-10:
        return 1e6
    
    a_min = min_triangle_area(pts)
    
    # We want to maximize a_min / a_hull
    # So we minimize - (a_min / a_hull)
    return - (a_min / a_hull)

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    # Initial points: Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    init_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Bounds for Differential Evolution: [-1, 1] x [-1, 1]
    # The optimal configuration will fit within a unit circle roughly.
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    best_score = score(init_pts)
    best_points = init_pts.copy()
    
    # Run Differential Evolution
    # maxiter=1000, tol=1e-8, seed=42
    # The number of function evaluations is (popsize * maxiter) roughly.
    # popsize default is 15 * len(bounds) = 15*26 = 390.
    # 390 * 1000 = 390,000 evals. 
    # Each eval takes ~0.1ms? 390,000 * 0.1ms = 39s. 
    # This is risky. Let's reduce maxiter or popsize.
    
    # Let's try popsize=10, maxiter=500.
    # 10*26=260 pop size. 260*500 = 130,000 evals. ~13s. Safe.
    
    try:
        # DE is stochastic. We run it once with a good seed.
        # If time permits, we can run a second DE with a different seed or polish.
        
        # We use the exact score function for DE.
        # DE handles non-smoothness okay, but it's slower.
        
        result = differential_evolution(
            smooth_surrogate, 
            bounds, 
            seed=42, 
            maxiter=600, 
            tol=1e-8, 
            popsize=12, 
            mutation=(0.5, 1.0), 
            recombination=0.7,
            polish=True, # DE will do a local polish at the end
            init='sobol' # Better initial distribution
        )
        
        if result.success:
            candidate_pts = result.x.reshape(-1, 2)
            candidate_score = score(candidate_pts)
            if candidate_score > best_score:
                best_score = candidate_score
                best_points = candidate_pts.copy()
        
    except Exception as e:
        pass

    # Phase 2: Local Polish with Nelder-Mead from the best DE result
    if time.time() < deadline - 5.0:
        x0 = best_points.flatten()
        
        def objective(x):
            pts = x.reshape(-1, 2)
            return -score(pts)
        
        try:
            res = minimize(
                objective, 
                x0, 
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            pts = res.x.reshape(-1, 2)
            s = score(pts)
            if s > best_score:
                best_score = s
                best_points = pts.copy()
        except Exception:
            pass

    # Phase 3: If time remains, try a second DE run with a different seed or init
    if time.time() < deadline - 20.0:
        try:
            result2 = differential_evolution(
                smooth_surrogate, 
                bounds, 
                seed=123, 
                maxiter=400, 
                tol=1e-8, 
                popsize=10, 
                mutation=(0.5, 1.0), 
                recombination=0.7,
                polish=False, # Skip polish in DE, we'll do it later
                init='latinhypercube'
            )
            
            if result2.success:
                candidate_pts = result2.x.reshape(-1, 2)
                candidate_score = score(candidate_pts)
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_points = candidate_pts.copy()
                    
                    # Polish the new best
                    if time.time() < deadline - 5.0:
                        x0 = best_points.flatten()
                        res = minimize(
                            objective, 
                            x0, 
                            method='Nelder-Mead',
                            options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
                        )
                        pts = res.x.reshape(-1, 2)
                        s = score(pts)
                        if s > best_score:
                            best_score = s
                            best_points = pts.copy()
        except Exception:
            pass

    return best_points

# EVOLVE-BLOCK-END
