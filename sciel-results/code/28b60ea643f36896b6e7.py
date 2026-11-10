import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform

def min_max_dist_dim2_16() -> np.ndarray:
    n = 16
    # We fix 2 points, so we optimize 14 points (28 variables)
    n_free = 14
    
    def objective(params):
        # params: shape (28,)
        # Fixed points:
        p0 = np.array([0.0, 0.0])
        p1 = np.array([1.0, 0.0])
        
        # Reconstruct full points array
        rest = params.reshape(n_free, 2)
        points = np.vstack((p0, p1, rest))
        
        # Calculate pairwise distances
        dists = pdist(points)
        d_min = np.min(dists)
        d_max = np.max(dists)
        
        if d_max < 1e-10:
            return 1e10
            
        # We want to maximize (d_min / d_max)^2
        # So minimize -(d_min / d_max)^2
        ratio = d_min / d_max
        return -(ratio ** 2)

    def prepare_guess(points):
        # points: shape (16, 2)
        # 1. Center
        points = points - np.mean(points, axis=0)
        
        # 2. Find max distance pair to fix
        # Use squareform for easy indexing
        D = squareform(pdist(points))
        i, j = np.unravel_index(np.argmax(D), D.shape)
        
        p_i = points[i]
        p_j = points[j]
        d_max = np.linalg.norm(p_j - p_i)
        
        if d_max < 1e-10:
             # Fallback if degenerate
             return np.zeros(28)

        # 3. Translate so p_i is at origin
        points = points - p_i
        
        # 4. Rotate so p_j is on positive x-axis
        v = points[j] # This is p_j - p_i
        c = v[0] / d_max
        s = v[1] / d_max
        R = np.array([[c, s], [-s, c]])
        points = points @ R.T
        
        # 5. Scale so distance is 1
        points = points / d_max
        
        # 6. Remove the two fixed points
        keep_indices = [k for k in range(n) if k != i and k != j]
        params = points[keep_indices].flatten()
        
        return params

    # Generate initial guesses
    
    # Guess 1: 4x4 Grid
    x = np.arange(4) - 1.5
    y = np.arange(4) - 1.5
    xx, yy = np.meshgrid(x, y)
    guess1 = np.column_stack((xx.ravel(), yy.ravel()))
    
    # Guess 2: Hexagonal lattice subset (16 closest to center)
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, np.sqrt(3)/2])
    hex_points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            p = i * v1 + j * v2
            hex_points.append(p)
    hex_points = np.array(hex_points)
    # Center
    hex_points = hex_points - np.mean(hex_points, axis=0)
    # Sort by distance from center, take 16 closest
    dists_to_center = np.linalg.norm(hex_points, axis=1)
    idx = np.argsort(dists_to_center)[:16]
    guess2 = hex_points[idx]
    
    # Guess 3: Random
    np.random.seed(42)
    guess3 = np.random.randn(n, 2)
    
    # Guess 4: Circle based
    angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
    radius = 1.0
    guess4 = np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))
    
    guesses = [guess1, guess2, guess3, guess4]
    
    best_score = -1e10
    best_params = None
    
    for g in guesses:
        # Prepare guess to 28-dim vector
        x0 = prepare_guess(g)
        
        # Run Nelder-Mead
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxfev': 5000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
            
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass
            
    if best_params is None:
        # Fallback to prepared guess 1
        best_params = prepare_guess(guess1)
        
    # Reconstruct points
    p0 = np.array([0.0, 0.0])
    p1 = np.array([1.0, 0.0])
    rest = best_params.reshape(n_free, 2)
    final_points = np.vstack((p0, p1, rest))
    
    return final_points
