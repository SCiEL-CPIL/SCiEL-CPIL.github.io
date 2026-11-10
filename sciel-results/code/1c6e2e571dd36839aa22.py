# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the points.
    """
    n = 16
    deadline = time.time() + 580  # safety margin before 630s kill
    
    def compute_R(points):
        """Compute R = (d_min/d_max)^2"""
        d = pdist(points)
        d_min = d.min()
        d_max = d.max()
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2
    
    def objective(x):
        """Objective to minimize: -log(R) = -2*log(d_min/d_max) = 2*log(d_max) - 2*log(d_min)"""
        points = x.reshape(n, 2)
        d = pdist(points)
        d_min = d.min()
        d_max = d.max()
        if d_min <= 1e-12 or d_max <= 1e-12:
            return 1e10
        return 2.0 * (np.log(d_max) - np.log(d_min))
    
    # Initial guess: 4x4 square grid
    def make_grid(scale=1.0):
        x = np.linspace(0, 3 * scale, 4)
        y = np.linspace(0, 3 * scale, 4)
        X, Y = np.meshgrid(x, y)
        pts = np.column_stack((X.ravel(), Y.ravel()))
        pts -= pts.mean(axis=0)  # center
        return pts
    
    best_points = make_grid()
    best_R = compute_R(best_points)
    
    # Try optimization from grid start
    try:
        res = minimize(objective, best_points.ravel(), method='L-BFGS-B', 
                       options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
        if res.success:
            cand = res.x.reshape(n, 2)
            cand_R = compute_R(cand)
            if cand_R > best_R:
                best_R = cand_R
                best_points = cand
    except Exception:
        pass
    
    if time.time() > deadline - 30:
        return best_points
    
    # Try multiple random restarts
    np.random.seed(42)
    for restart in range(50):
        if time.time() > deadline - 30:
            break
        
        # Different initializations
        if restart % 3 == 0:
            # Perturbed grid
            init = make_grid(scale=1.0 + np.random.randn() * 0.1)
            init += np.random.randn(n, 2) * 0.1
        elif restart % 3 == 1:
            # Random points in unit square, centered
            init = np.random.randn(n, 2) * 0.5
            init -= init.mean(axis=0)
        else:
            # Hexagonal-ish pattern
            init = np.zeros((n, 2))
            for i in range(4):
                for j in range(4):
                    idx = i * 4 + j
                    init[idx, 0] = j + 0.5 * (i % 2)
                    init[idx, 1] = i * np.sqrt(3) / 2
            init -= init.mean(axis=0)
            init += np.random.randn(n, 2) * 0.05
        
        try:
            res = minimize(objective, init.ravel(), method='L-BFGS-B',
                           options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.success:
                cand = res.x.reshape(n, 2)
                cand_R = compute_R(cand)
                if cand_R > best_R:
                    best_R = cand_R
                    best_points = cand
        except Exception:
            pass
    
    # Final refinement on best point
    if time.time() < deadline - 10:
        try:
            res = minimize(objective, best_points.ravel(), method='L-BFGS-B',
                           options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10})
            if res.success:
                cand = res.x.reshape(n, 2)
                cand_R = compute_R(cand)
                if cand_R > best_R:
                    best_R = cand_R
                    best_points = cand
        except Exception:
            pass
    
    # Ensure valid output
    best_points = np.asarray(best_points, dtype=np.float64)
    if best_points.shape != (16, 2):
        # Fallback to grid
        best_points = make_grid()
    
    return best_points


# EVOLVE-BLOCK-END
