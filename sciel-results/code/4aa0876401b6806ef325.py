import numpy as np
from scipy.optimize import minimize

N = 50

def construct():
    # Objective: maximize r = min(min_{i<j} |c_i - c_j|/2, min_i min_k (0.5 - |c_ik|))
    # We minimize -r
    
    def objective(flat):
        c = flat.reshape(N, 3)
        # Check bounds
        if np.any(c < -0.5) or np.any(c > 0.5):
            return 1e10
        # Pairwise distances
        from scipy.spatial.distance import cdist
        D = cdist(c, c)
        np.fill_diagonal(D, np.inf)
        min_pair = np.min(D) / 2.0
        # Wall distances
        wall_dists = np.min(np.abs(c), axis=1)  # distance to nearest face
        min_wall = 0.5 - np.max(np.abs(c), axis=1)  # wait, need min over k of (0.5 - |c_ik|)
        # Actually: for each sphere i, min_k (0.5 - |c_ik|) = 0.5 - max_k |c_ik|
        min_wall = np.min(0.5 - np.max(np.abs(c), axis=1))
        r = min(min_pair, min_wall)
        return -r
    
    def initial_guess(seed):
        rng = np.random.RandomState(seed)
        # Start with a 5x5x2 grid
        xs = np.linspace(-0.4, 0.4, 5)
        ys = np.linspace(-0.4, 0.4, 5)
        zs = np.array([-0.2, 0.2])
        grid = []
        for z in zs:
            for y in ys:
                for x in xs:
                    grid.append([x, y, z])
        grid = np.array(grid[:N])
        # Add small random perturbation
        grid += rng.normal(0, 0.01, grid.shape)
        # Clip to bounds
        grid = np.clip(grid, -0.49, 0.49)
        return grid.flatten()
    
    best_r = -1e10
    best_c = None
    
    for seed in range(10):
        x0 = initial_guess(seed)
        res = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10})
        c = res.x.reshape(N, 3)
        # Compute actual r
        from scipy.spatial.distance import cdist
        D = cdist(c, c)
        np.fill_diagonal(D, np.inf)
        min_pair = np.min(D) / 2.0
        min_wall = np.min(0.5 - np.max(np.abs(c), axis=1))
        r = min(min_pair, min_wall)
        if r > best_r:
            best_r = r
            best_c = c
    
    return best_c
