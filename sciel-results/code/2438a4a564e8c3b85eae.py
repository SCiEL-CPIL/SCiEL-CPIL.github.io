# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize, differential_evolution
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    n = 16
    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    
    def time_exceeded():
        return time.time() > deadline

    def objective(params):
        """Objective function to minimize: -R = -(d_min/d_max)^2"""
        points = params.reshape((n, 2))
        dists = pdist(points, 'euclidean')
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-10:
            return 0.0
        ratio = d_min / d_max
        return -(ratio ** 2)
        
    def evaluate(points):
        dists = pdist(points, 'euclidean')
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-10:
            return 0.0
        return (d_min / d_max) ** 2

    best_points = None
    best_score = -1.0

    def update_best(points):
        nonlocal best_score, best_points
        score = evaluate(points)
        if score > best_score:
            best_score = score
            best_points = points

    # 1. Generate diverse initial configurations
    starts = []
    
    # Hexagonal lattice fragment
    # Generate a hex grid with radius R=3 (37 points) and pick the 16 closest to center
    hex_points = []
    R = 3
    for q in range(-R, R+1):
        for r in range(max(-R, -q-R), min(R, -q+R)+1):
            x = 1.5 * q
            y = np.sqrt(3) / 2 * q + np.sqrt(3) * r
            hex_points.append([x, y])
    hex_points = np.array(hex_points)
    hex_points -= np.mean(hex_points, axis=0)
    dists_center = np.linalg.norm(hex_points, axis=1)
    sorted_idx = np.argsort(dists_center)
    hex_start = hex_points[sorted_idx[:16]]
    starts.append(hex_start)

    # Square grid
    x = np.linspace(-1, 1, 4)
    y = np.linspace(-1, 1, 4)
    xx, yy = np.meshgrid(x, y)
    grid_start = np.column_stack((xx.ravel(), yy.ravel()))
    starts.append(grid_start)

    # Circle
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    circle_start = np.column_stack((np.cos(angles), np.sin(angles)))
    starts.append(circle_start)

    # Random starts
    for i in range(5):
        np.random.seed(100+i)
        starts.append(np.random.uniform(-1, 1, (n, 2)))

    # 2. Local Optimization (Nelder-Mead) on each start
    for start in starts:
        if time_exceeded(): break
        
        # Normalize start to unit scale to help NM
        norm = np.max(np.linalg.norm(start, axis=1))
        if norm > 1e-10:
            start_norm = start / norm
        else:
            start_norm = np.random.randn(n, 2)
            
        def callback(xk):
            return time_exceeded()
            
        try:
            # Use tight tolerances for precision
            res = minimize(objective, start_norm.ravel(), method='Nelder-Mead', 
                           options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14},
                           callback=callback)
            if not time_exceeded():
                update_best(res.x.reshape((n, 2)))
        except Exception:
            pass

    # 3. Global Optimization (Differential Evolution)
    # DE is robust and can escape local minima.
    if not time_exceeded():
        bounds = [(-1, 1)] * (n * 2)
        try:
            # Sobol sequence provides good coverage of the search space
            res_de = differential_evolution(
                objective, 
                bounds=bounds, 
                maxiter=2000, 
                popsize=30, 
                tol=1e-6, 
                seed=42,
                polish=False,
                init='sobol'
            )
            if not time_exceeded():
                update_best(res_de.x.reshape((n, 2)))
        except Exception:
            pass

    # 4. Final Polish on the best found
    if not time_exceeded() and best_points is not None:
        # Try a few perturbations of the best point to ensure we are at the local optimum
        for i in range(10):
            if time_exceeded(): break
            np.random.seed(200 + i)
            noise = np.random.randn(n, 2) * 0.01
            start = best_points + noise
            
            # Normalize
            norm = np.max(np.linalg.norm(start, axis=1))
            if norm > 1e-10:
                start = start / norm
                
            def callback2(xk):
                return time_exceeded()
                
            try:
                res = minimize(objective, start.ravel(), method='Nelder-Mead', 
                               options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14},
                               callback=callback2)
                if not time_exceeded():
                    update_best(res.x.reshape((n, 2)))
            except Exception:
                pass

    # Fallback
    if best_points is None:
        np.random.seed(0)
        best_points = np.random.randn(n, 2)
        
    return best_points

# EVOLVE-BLOCK-END
