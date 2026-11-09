# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the coordinates of the 14 points.
    """
    n = 14
    d = 3
    
    start_time = time.time()
    time_limit = 590.0  # Safety margin for 600s limit

    def sph_to_cart(sph):
        # sph is (n, 2): theta, phi
        theta = sph[:, 0]
        phi = sph[:, 1]
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        return np.column_stack((x, y, z))

    def objective(sph):
        cart = sph_to_cart(sph)
        dists = pdist(cart)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-12:
            return 0.0
        ratio = (d_min / d_max) ** 2
        return -ratio

    best_score = -np.inf
    best_points = None
    
    # Use multiple random starts
    num_starts = 100
    np.random.seed(42)
    
    for i in range(num_starts):
        if time.time() - start_time > time_limit:
            break
            
        # Random initialization on sphere
        u = np.random.rand(n)
        v = np.random.rand(n)
        theta = np.arccos(2 * u - 1)
        phi = 2 * np.pi * v
        
        x0 = np.column_stack((theta, phi))
        
        # Optimization
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
            
            current_score = -res.fun
            if current_score > best_score:
                best_score = current_score
                best_points = sph_to_cart(res.x)
                
        except Exception:
            continue
            
    # Fallback if optimization failed or time ran out before any success
    if best_points is None:
        u = np.random.rand(n)
        v = np.random.rand(n)
        theta = np.arccos(2 * u - 1)
        phi = 2 * np.pi * v
        best_points = sph_to_cart(np.column_stack((theta, phi)))
        
    return best_points

# EVOLVE-BLOCK-END
