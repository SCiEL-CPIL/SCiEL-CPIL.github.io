# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time

def _score(points):
    """
    Calculate the score (d_min / d_max)^2 for a set of points.
    """
    if len(points) < 2:
        return 0.0
    
    # Compute all pairwise distances
    dists = pdist(points, metric='euclidean')
    
    if len(dists) == 0:
        return 0.0

    d_min = np.min(dists)
    d_max = np.max(dists)
    
    if d_max == 0 or np.isnan(d_min) or np.isnan(d_max):
        return 0.0
        
    return (d_min / d_max) ** 2

def _objective(x, n=14, d=3):
    """
    Objective function for minimization (negative of score).
    """
    points = x.reshape(n, d)
    return -_score(points)

def min_max_dist_dim3_14() -> np.ndarray:
    n = 14
    d = 3
    start_time = time.time()
    max_time = 590.0  # Safety margin
    
    best_score = -1
    best_points = None
    
    # Generate Inits
    inits = []
    
    # 1. Icosahedron + 2 points
    phi = (1 + np.sqrt(5)) / 2
    icosa = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    icosa_norms = np.linalg.norm(icosa, axis=1, keepdims=True)
    icosa = icosa / icosa_norms
    
    # Add 2 points opposite to each other or random
    # Try a few configurations for the extra 2 points
    for _ in range(10):
        # Random perturbations of icosa
        perturbed = icosa + np.random.randn(12, 3) * 0.1
        # Normalize back to sphere? No, let optimizer handle scale/position
        # But keeping them on sphere is a good prior
        p_norms = np.linalg.norm(perturbed, axis=1, keepdims=True)
        perturbed = perturbed / p_norms
        
        # Add 2 random points on sphere
        p13 = np.random.randn(1, 3)
        p13 = p13 / np.linalg.norm(p13)
        p14 = np.random.randn(1, 3)
        p14 = p14 / np.linalg.norm(p14)
        
        inits.append(np.vstack([perturbed, p13, p14]))
        
    # 2. Cube + Octahedron
    cube = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]
    ])
    octa = np.array([
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1]
    ])
    inits.append(np.vstack([cube, octa]))
    
    # 3. Random Sphere
    np.random.seed(42)
    for _ in range(20):
        z = np.random.randn(n, d)
        norms = np.linalg.norm(z, axis=1, keepdims=True)
        z = z / norms
        inits.append(z)

    # Optimization Loop
    num_restarts = len(inits)
    for i in range(num_restarts):
        if time.time() - start_time > max_time:
            break
            
        x0 = inits[i]
        try:
            # Use Nelder-Mead for robustness
            res = minimize(_objective, x0, method='Nelder-Mead', 
                           options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 2000, 'adaptive': True})
            
            if res.fun < -best_score:
                best_score = -res.fun
                best_points = res.x.reshape(n, d)
                
        except Exception:
            pass

    if best_points is None:
        best_points = inits[0]
        
    return best_points

# EVOLVE-BLOCK-END
