# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the coordinates of the 14 points.
    """
    start_time = time.time()
    deadline = start_time + 550  # 550s limit, leaving 50s margin for safety
    
    n = 14
    d = 3
    
    def get_score(points):
        """Compute the metric (d_min/d_max)^2"""
        if not np.all(np.isfinite(points)):
            return 0.0
        dists = pdist(points)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2
    
    def objective(x):
        """Negative of the score for minimization"""
        points = x.reshape(n, d)
        if not np.all(np.isfinite(points)):
            return 0.0
        dists = pdist(points)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        # We want to maximize (d_min/d_max)^2, so minimize negative
        # Use a smooth approximation to avoid issues with min/max
        return -(d_min / d_max) ** 2
    
    # Initialize with icosahedron + 2 points
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    
    # 12 icosahedron vertices
    ico_vertices = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    
    # Normalize to unit sphere
    ico_vertices = ico_vertices / np.linalg.norm(ico_vertices, axis=1, keepdims=True)
    
    # Add 2 more points - place them at opposite poles or symmetric positions
    # Let's try placing them along an axis where there's symmetry
    # The icosahedron has 5-fold symmetry axes. Let's add points at (0,0,1) and (0,0,-1)
    # But first check if those are already close to existing vertices
    extra_points = np.array([[0, 0, 1], [0, 0, -1]], dtype=float)
    
    # Combine
    points = np.vstack([ico_vertices, extra_points])
    
    # Normalize all points to unit sphere for initial optimization
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    # Flatten for optimization
    x0 = points.flatten()
    
    best_score = get_score(points)
    best_points = points.copy()
    
    # Run Nelder-Mead optimization
    try:
        result = minimize(
            objective,
            x0,
            method='Nelder-Mead',
            options={
                'maxiter': 100000,
                'xatol': 1e-10,
                'fatol': 1e-12,
                'adaptive': True
            }
        )
        
        if result.success or result.fun < 0:
            opt_points = result.x.reshape(n, d)
            opt_score = get_score(opt_points)
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_points.copy()
        
    except Exception as e:
        pass
    
    # Check deadline
    if time.time() > deadline:
        return best_points
    
    # Try a second optimization from a different starting point
    # Perturb the best solution and try again
    np.random.seed(123)
    perturbed = best_points + 0.1 * np.random.randn(n, d)
    perturbed = perturbed / np.linalg.norm(perturbed, axis=1, keepdims=True)
    x1 = perturbed.flatten()
    
    try:
        result2 = minimize(
            objective,
            x1,
            method='Nelder-Mead',
            options={
                'maxiter': 100000,
                'xatol': 1e-10,
                'fatol': 1e-12,
                'adaptive': True
            }
        )
        
        if result2.success or result2.fun < 0:
            opt_points2 = result2.x.reshape(n, d)
            opt_score2 = get_score(opt_points2)
            if opt_score2 > best_score:
                best_score = opt_score2
                best_points = opt_points2.copy()
        
    except Exception as e:
        pass
    
    if time.time() > deadline:
        return best_points
    
    # Try yet another initialization: uniform random on sphere with good seed
    for seed in [42, 7, 99]:
        if time.time() > deadline:
            break
        np.random.seed(seed)
        rand_points = np.random.randn(n, d)
        rand_points = rand_points / np.linalg.norm(rand_points, axis=1, keepdims=True)
        x_rand = rand_points.flatten()
        
        try:
            result_rand = minimize(
                objective,
                x_rand,
                method='Nelder-Mead',
                options={
                    'maxiter': 50000,
                    'xatol': 1e-9,
                    'fatol': 1e-11,
                    'adaptive': True
                }
            )
            
            if result_rand.success or result_rand.fun < 0:
                opt_points_rand = result_rand.x.reshape(n, d)
                opt_score_rand = get_score(opt_points_rand)
                if opt_score_rand > best_score:
                    best_score = opt_score_rand
                    best_points = opt_points_rand.copy()
                    
        except Exception as e:
            pass
    
    return best_points


# EVOLVE-BLOCK-END
