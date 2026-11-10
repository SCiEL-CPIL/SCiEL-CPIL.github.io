import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time

def min_max_dist_dim3_14() -> np.ndarray:
    n = 14
    
    # Objective function to maximize: (d_min / d_max)^2
    # We minimize the negative of this.
    # To handle scale invariance, we normalize the points inside the objective.
    
    def score_func(x):
        p = x.reshape(n, 3)
        # Center the points
        p = p - np.mean(p, axis=0)
        
        # Normalize scale: Fix sum of squared norms to n (avg norm 1)
        norm_sq = np.sum(p**2)
        if norm_sq < 1e-12:
            return 0.0
        p = p * np.sqrt(n / norm_sq)
        
        d = pdist(p)
        if len(d) == 0:
            return 0.0
        d_min = np.min(d)
        d_max = np.max(d)
        
        if d_max < 1e-12:
            return 0.0
            
        ratio = d_min / d_max
        return ratio**2

    def objective(x):
        return -score_func(x)

    # Initial Guess 1: Icosahedron + 2 Poles
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, -1, phi],
        [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0],
        [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1],
        [phi, 0, -1], [-phi, 0, -1]
    ])
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    poles = np.array([[0, 0, 1], [0, 0, -1]])
    init1 = np.vstack([ico, poles]).flatten()

    # Initial Guess 2: Octahedron (6) + Cube (8)
    octa = np.array([
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1]
    ])
    # Cube vertices
    cube = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]
    ])
    # Scale cube to match octahedron radius? 
    # Octa radius 1. Cube radius sqrt(3).
    # Let's just normalize later in the objective.
    init2 = np.vstack([octa, cube]).flatten()

    # Initial Guess 3: Random on Sphere
    np.random.seed(123)
    init3 = np.random.randn(n, 3)
    init3 = init3 / np.linalg.norm(init3, axis=1, keepdims=True)
    init3 = init3.flatten()

    best_score = -np.inf
    best_points = init1.reshape(n, 3)
    
    start_time = time.time()
    time_limit = 300 # Keep well under 630s
    
    inits = [init1, init2, init3]
    
    for i, init in enumerate(inits):
        if time.time() - start_time > time_limit:
            break
            
        try:
            res = minimize(
                objective, 
                init, 
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
            )
            
            # Check time
            if time.time() - start_time > time_limit:
                break
                
            current_score = -res.fun
            if current_score > best_score:
                best_score = current_score
                best_points = res.x.reshape(n, 3)
                
        except Exception as e:
            pass

    # Final normalization of the best points to ensure finite and reasonable values
    # Although the objective function normalizes, the returned points should be clean.
    # The problem doesn't require specific scaling, just the ratio.
    # However, returning points with norm ~1 is standard.
    p = best_points - np.mean(best_points, axis=0)
    norm_sq = np.sum(p**2)
    if norm_sq > 0:
        p = p * np.sqrt(n / norm_sq)
        
    return p
