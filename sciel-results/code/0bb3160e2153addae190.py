# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import time
from scipy.optimize import minimize

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    """
    start_time = time.time()
    deadline = 600  # 600 seconds limit, keep a margin
    
    n = 14
    d = 3
    
    # Helper to calculate the metric (min distance)
    # We maximize d_min. We assume points are normalized to unit sphere or we handle scale.
    # The ratio (d_min/d_max)^2 is scale invariant.
    # We can just maximize d_min subject to points being "tight".
    # A common trick: maximize d_min subject to sum(||p_i||^2) = N.
    # Or simply normalize the result at the end.
    # For optimization, we use the objective: -d_min + penalty for not being on unit sphere.
    
    def objective(x):
        # x is shape (14, 3)
        points = x.reshape(n, d)
        
        # Calculate pairwise distances
        # pdist returns condensed form
        dists = pdist(points)
        if len(dists) == 0:
            return 0
        d_min = np.min(dists)
        
        # Penalty to keep points on the unit sphere
        norms = np.linalg.norm(points, axis=1)
        sphere_penalty = np.sum((norms - 1.0)**2)
        
        # We want to maximize d_min, so minimize -d_min
        # Add a large weight to the sphere penalty to enforce the constraint
        return -d_min + 10.0 * sphere_penalty

    # Initialization 1: Icosahedron + 2 points at face centroids
    phi = (1 + np.sqrt(5)) / 2
    ico_coords = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    # Normalize icosahedron to unit sphere
    ico_norms = np.linalg.norm(ico_coords, axis=1)
    ico_coords = ico_coords / ico_norms[:, np.newaxis]
    
    # Find face centroids? 
    # The icosahedron has 12 vertices. The dual is the dodecahedron with 20 vertices (face centroids).
    # We can compute the centroids of the faces of the icosahedron.
    # Or simpler: The dodecahedron vertices are the face centroids of the icosahedron.
    # Dodecahedron vertices:
    # (±1, ±1, ±1), (0, ±1/φ, ±φ), (±1/φ, ±φ, 0), (±φ, 0, ±1/φ)
    # Let's generate the dodecahedron vertices and pick 2.
    # Actually, we just need 2 points. Let's pick two opposite vertices of the dodecahedron.
    # Dodecahedron vertex (1, 1, 1) normalized.
    v1 = np.array([1, 1, 1])
    v1 = v1 / np.linalg.norm(v1)
    v2 = -v1
    
    init_1 = np.vstack([ico_coords, [v1], [v2]])
    
    # Initialization 2: Random points on sphere
    np.random.seed(42)
    init_2 = np.random.randn(n, d)
    init_2 /= np.linalg.norm(init_2, axis=1)[:, np.newaxis]
    
    # Initialization 3: Another random start
    np.random.seed(123)
    init_3 = np.random.randn(n, d)
    init_3 /= np.linalg.norm(init_3, axis=1)[:, np.newaxis]
    
    best_score = -np.inf
    best_points = init_1
    
    candidates = [init_1, init_2, init_3]
    
    for i, init in enumerate(candidates):
        if time.time() > deadline - 30:
            break
            
        try:
            # Use Powell method which is derivative-free and robust
            res = minimize(objective, init.flatten(), method='Powell', options={'maxiter': 500, 'xtol': 1e-8, 'ftol': 1e-8})
            
            if res.success or res.fun < best_score:
                res_points = res.x.reshape(n, d)
                # Calculate actual metric
                dists = pdist(res_points)
                d_min = np.min(dists)
                d_max = np.max(dists)
                if d_max > 0:
                    score = (d_min / d_max)**2
                else:
                    score = 0
                
                if score > best_score:
                    best_score = score
                    best_points = res_points
                    
        except Exception as e:
            pass

    # If we have time, do a few more random restarts
    while time.time() < deadline - 60:
        np.random.seed(int(time.time() * 1000) % 10000)
        init_rand = np.random.randn(n, d)
        init_rand /= np.linalg.norm(init_rand, axis=1)[:, np.newaxis]
        
        try:
            res = minimize(objective, init_rand.flatten(), method='Powell', options={'maxiter': 300, 'xtol': 1e-8, 'ftol': 1e-8})
            res_points = res.x.reshape(n, d)
            dists = pdist(res_points)
            d_min = np.min(dists)
            d_max = np.max(dists)
            if d_max > 0:
                score = (d_min / d_max)**2
            else:
                score = 0
            
            if score > best_score:
                best_score = score
                best_points = res_points
        except Exception:
            pass

    # Final check and normalization
    if not np.all(np.isfinite(best_points)):
        return np.zeros((14, 3))
        
    return best_points

# EVOLVE-BLOCK-END
