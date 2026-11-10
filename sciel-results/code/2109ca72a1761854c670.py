# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution, minimize
import warnings


def _score(points):
    """Compute R = (d_min/d_max)^2 for a set of points."""
    d = pdist(points)
    d_min = np.min(d)
    d_max = np.max(d)
    if d_max == 0:
        return 0.0
    return (d_min / d_max) ** 2


def _objective(points_flat):
    """Objective to minimize: negative log of score for numerical stability."""
    n = 16
    points = points_flat.reshape(n, 2)
    score = _score(points)
    if score <= 0:
        return 1e10
    return -np.log(score)


def _local_refine(points, maxiter=200, tol=1e-12):
    """Local refinement using Nelder-Mead on the full point set."""
    x0 = points.flatten()
    result = minimize(
        _objective,
        x0,
        method='Nelder-Mead',
        options={'maxiter': maxiter, 'xatol': tol, 'fatol': tol, 'adaptive': True}
    )
    refined = result.x.reshape(16, 2)
    return refined


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    n = 16
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s limit

    best_points = None
    best_score = 0.0

    # Initial best guess: random points
    np.random.seed(42)
    init_points = np.random.randn(n, 2)
    best_points = init_points.copy()
    best_score = _score(best_points)

    # Multi-start Differential Evolution
    # Bounds: we can normalize to unit scale, so [-5, 5] is generous
    bounds = [(-5.0, 5.0)] * (n * 2)

    # Strategy 1: Differential Evolution with moderate population
    # Using a smooth surrogate might be faster, but let's try direct first with DE
    # DE is robust for global search
    
    # We'll run DE with a limited number of iterations to keep it fast,
    # then refine the best result locally.
    
    try:
        # Run DE
        de_result = differential_evolution(
            _objective,
            bounds,
            seed=42,
            maxiter=50,
            popsize=20,
            tol=1e-8,
            polish=False,  # We will do our own polish
            init='sobol'
        )
        
        if de_result.x is not None:
            de_points = de_result.x.reshape(n, 2)
            de_score = _score(de_points)
            if de_score > best_score:
                best_score = de_score
                best_points = de_points.copy()
                
    except Exception as e:
        pass

    # Check time
    if time.time() > deadline - 10:
        return best_points

    # Refine the best candidate found by DE
    refined_points = _local_refine(best_points, maxiter=500)
    refined_score = _score(refined_points)
    if refined_score > best_score:
        best_score = refined_score
        best_points = refined_points.copy()

    # Multi-start local refinement with different initializations
    # Use structured initializations: rings, grids, etc.
    
    initializations = []
    
    # 1. Regular hexagonal-like packing
    # 2. Random points on a circle
    # 3. Grid points
    # 4. Fibonacci sphere projected to 2D (or just random)
    
    # Add a few more random starts
    for seed in range(10):
        if time.time() > deadline - 20:
            break
        np.random.seed(seed)
        pts = np.random.randn(n, 2) * 2
        initializations.append(pts)
        
    # Add a ring initialization
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    ring_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    initializations.append(ring_pts)
    
    # Add a grid initialization
    grid_pts = np.array([[i % 4, i // 4] for i in range(n)])
    grid_pts = (grid_pts - 1.5) * 0.5
    initializations.append(grid_pts)
    
    for init_pts in initializations:
        if time.time() > deadline - 30:
            break
        
        # Local refinement from this start
        refined = _local_refine(init_pts, maxiter=300)
        score = _score(refined)
        if score > best_score:
            best_score = score
            best_points = refined.copy()

    # Final polish with tighter tolerance
    if time.time() < deadline - 5:
        final_refined = _local_refine(best_points, maxiter=1000, tol=1e-14)
        final_score = _score(final_refined)
        if final_score > best_score:
            best_points = final_refined.copy()
            best_score = final_score

    return best_points


# EVOLVE-BLOCK-END
