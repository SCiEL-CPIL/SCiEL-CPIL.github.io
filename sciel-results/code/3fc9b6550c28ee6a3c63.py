# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    n = 16
    d = 2
    
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    def score(points):
        """Calculate R = (d_min / d_max)^2"""
        if len(points) < 2:
            return 0.0
        dists = pdist(points)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def objective(x):
        """Objective function for minimization (we minimize -score)"""
        pts = x.reshape(-1, 2)
        s = score(pts)
        # We want to maximize s, so minimize -s
        # Adding a small penalty for non-finite values
        if not np.isfinite(s):
            return 1e10
        return -s

    # Generate initial configurations
    initial_guesses = []
    
    # 1. 4x4 Grid
    x = np.arange(4)
    y = np.arange(4)
    xx, yy = np.meshgrid(x, y)
    grid_pts = np.column_stack((xx.ravel(), yy.ravel()))
    # Normalize to unit diameter roughly
    grid_pts = grid_pts - np.mean(grid_pts, axis=0)
    d_max_grid = np.max(pdist(grid_pts))
    if d_max_grid > 0:
        grid_pts = grid_pts / d_max_grid
    initial_guesses.append(grid_pts.flatten())

    # 2. Concentric Rings: 1 center, 5 ring, 10 ring
    # Center
    p0 = np.array([0.0, 0.0])
    # Ring 1 (5 points)
    angles1 = np.linspace(0, 2*np.pi, 5, endpoint=False)
    r1 = 0.5
    ring1 = np.column_stack((r1 * np.cos(angles1), r1 * np.sin(angles1)))
    # Ring 2 (10 points)
    angles2 = np.linspace(0, 2*np.pi, 10, endpoint=False)
    r2 = 1.0
    ring2 = np.column_stack((r2 * np.cos(angles2), r2 * np.sin(angles2)))
    
    pts_1_5_10 = np.vstack((p0.reshape(1,2), ring1, ring2))
    # Normalize
    pts_1_5_10 = pts_1_5_10 - np.mean(pts_1_5_10, axis=0)
    d_max_1510 = np.max(pdist(pts_1_5_10))
    if d_max_1510 > 0:
        pts_1_5_10 = pts_1_5_10 / d_max_1510
    initial_guesses.append(pts_1_5_10.flatten())

    # 3. Concentric Rings: 1 center, 6 ring, 9 ring
    angles1_6 = np.linspace(0, 2*np.pi, 6, endpoint=False)
    r1_6 = 0.5
    ring1_6 = np.column_stack((r1_6 * np.cos(angles1_6), r1_6 * np.sin(angles1_6)))
    angles2_9 = np.linspace(0, 2*np.pi, 9, endpoint=False)
    r2_9 = 1.0
    ring2_9 = np.column_stack((r2_9 * np.cos(angles2_9), r2_9 * np.sin(angles2_9)))
    
    pts_1_6_9 = np.vstack((p0.reshape(1,2), ring1_6, ring2_9))
    pts_1_6_9 = pts_1_6_9 - np.mean(pts_1_6_9, axis=0)
    d_max_169 = np.max(pdist(pts_1_6_9))
    if d_max_169 > 0:
        pts_1_6_9 = pts_1_6_9 / d_max_169
    initial_guesses.append(pts_1_6_9.flatten())

    # 4. Random points (seeded for reproducibility in initial guess generation, but we'll vary it)
    np.random.seed(42)
    for i in range(5):
        rand_pts = np.random.randn(n, 2)
        rand_pts = rand_pts - np.mean(rand_pts, axis=0)
        d_max_r = np.max(pdist(rand_pts))
        if d_max_r > 0:
            rand_pts = rand_pts / d_max_r
        initial_guesses.append(rand_pts.flatten())

    best_score = -np.inf
    best_points = initial_guesses[0].reshape(-1, 2) # Fallback to first guess

    # Optimization loop
    # We use Powell as it is derivative-free and robust for non-smooth objectives
    # We limit iterations to avoid timeouts
    
    for x0 in initial_guesses:
        if time.time() > deadline:
            break
            
        try:
            # Use Powell method
            # maxiter is the number of iterations (function evaluations)
            # We allow a generous number but check time frequently
            # Note: Powell can be slow. 1000 iterations might be too many.
            # Let's try 200 iterations first to see speed, then scale if time permits.
            # Actually, for 32 dimensions, Powell is quite slow.
            # Let's try Nelder-Mead with maxiter=1000? Also slow.
            
            # Alternative: Use L-BFGS-B with numerical gradients?
            # The objective is - (min/max)^2.
            # Gradient is undefined at min/max switches.
            # However, scipy's minimize with 'L-BFGS-B' uses numerical gradients by default if jac is None.
            # This is expensive (2*N evaluations per gradient step).
            
            # Let's stick with Powell but with a strict time check inside a wrapper?
            # Scipy's minimize doesn't support a callback for time checking easily in all versions.
            # We can just run it and hope it's fast enough, or use a smaller maxiter.
            
            # Let's try 'Nelder-Mead' with a tight maxiter.
            # Or 'Powell' with maxiter=100.
            
            res = minimize(objective, x0, method='Powell', options={'maxiter': 100, 'maxfev': 1000})
            
            if res.success or res.fun < best_score:
                # Check if result is valid
                pts = res.x.reshape(-1, 2)
                s = score(pts)
                if np.isfinite(s) and s > best_score:
                    best_score = s
                    best_points = pts
                    
        except Exception as e:
            # If optimization fails, try the initial guess
            pts = x0.reshape(-1, 2)
            s = score(pts)
            if np.isfinite(s) and s > best_score:
                best_score = s
                best_points = pts

    # If we have time left, do a second pass with more iterations on the best point found
    if time.time() < deadline - 50:
        try:
            res = minimize(objective, best_points.flatten(), method='Powell', options={'maxiter': 500, 'maxfev': 5000})
            if res.fun < best_score:
                pts = res.x.reshape(-1, 2)
                s = score(pts)
                if np.isfinite(s) and s > best_score:
                    best_score = s
                    best_points = pts
        except Exception:
            pass

    # Final sanity check
    if not np.all(np.isfinite(best_points)):
        # Fallback to grid
        best_points = initial_guesses[0].reshape(-1, 2)

    return best_points

# EVOLVE-BLOCK-END
