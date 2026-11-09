# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    n = 16
    start_time = time.time()
    timeout = 600.0  # Safety margin
    
    def objective(x):
        # x is flattened array of shape (32,)
        points = x.reshape(n, 2)
        dists = pdist(points)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        # We want to maximize d_min / d_max, so minimize -ratio
        return -(d_min / d_max)

    def run_optimization(init_points):
        x0 = init_points.flatten()
        try:
            from scipy.optimize import minimize
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-10})
            if res.success:
                return res.x.reshape(n, 2)
            else:
                return init_points
        except Exception:
            return init_points

    best_points = None
    best_score = -1.0

    # Initialization 1: Regular 16-gon
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    init1 = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Initialization 2: Random points in unit disk
    np.random.seed(42)
    r = np.sqrt(np.random.rand(n))
    theta = np.random.rand(n) * 2 * np.pi
    init2 = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    
    # Initialization 3: Hexagonal lattice points inside unit disk
    init3 = []
    # Generate lattice points
    s = 0.5
    for i in range(-10, 11):
        for j in range(-10, 11):
            x = i * s * np.sqrt(3)
            y = j * s * 3.0
            if i % 2 == 1:
                y += s * 1.5
            if x*x + y*y <= 1.0:
                init3.append([x, y])
    init3 = np.array(init3)
    if len(init3) >= n:
        # Shuffle and take first n, or pick best n? 
        # For init, just take first n after shuffling
        np.random.shuffle(init3)
        init3 = init3[:n]
    else:
        # Fallback to random if lattice has too few points
        init3 = init2.copy()

    initializations = [init1, init2, init3]
    
    for i, init in enumerate(initializations):
        if time.time() - start_time > timeout - 60:
            break
        
        # Run optimization
        opt_points = run_optimization(init)
        
        # Calculate score
        dists = pdist(opt_points)
        if len(dists) > 0:
            d_min = np.min(dists)
            d_max = np.max(dists)
            if d_max > 0:
                score = (d_min / d_max) ** 2
            else:
                score = 0.0
        else:
            score = 0.0
            
        if score > best_score:
            best_score = score
            best_points = opt_points

    # If no optimization succeeded or took too long, return the best initial config
    if best_points is None:
        best_points = init1
        
    return best_points

# EVOLVE-BLOCK-END
