import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

N = 100

# EVOLVE-BLOCK-START
def construct():
    def get_radius(pts):
        """Calculate the maximum valid radius for a set of points."""
        if len(pts) == 0:
            return 0.0
        # Pairwise distances
        dists = pdist(pts)
        min_pair_dist = np.min(dists) / 2.0 if len(dists) > 0 else np.inf
        
        # Wall distances
        # min_k (0.5 - |pts[i, k]|)
        wall_dists = 0.5 - np.abs(pts)
        min_wall_dist = np.min(wall_dists)
        
        return min(min_pair_dist, min_wall_dist)

    def objective(x):
        pts = x.reshape(-1, 3)
        dists = pdist(pts)
        wall_dists = 0.5 - np.abs(pts)
        min_wall_dists = np.min(wall_dists, axis=1)
        
        eps = 1e-6
        # Use a softer potential to avoid numerical instability if points get very close
        # 1/d^2 is stiff. 1/d^4 is stiffer. Let's stick with 1/d^2.
        # To improve conditioning, we can cap the energy if distance is very large? 
        # No, 1/d^2 decays fast.
        
        pairwise_energy = np.sum(1.0 / (dists + eps)**2)
        wall_energy = np.sum(1.0 / (min_wall_dists + eps)**2)
        
        return pairwise_energy + wall_energy

    bounds = [(-0.5, 0.5)] * (N * 3)
    
    best_radius = -np.inf
    best_pts = None
    
    # Try multiple initializations
    for seed in [42, 123, 999, 7]:
        rng = np.random.default_rng(seed)
        
        # Base grid: 4x5x5
        nx, ny, nz = 4, 5, 5
        xs = np.linspace(-0.4, 0.4, nx)
        ys = np.linspace(-0.4, 0.4, ny)
        zs = np.linspace(-0.4, 0.4, nz)
        
        grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
        
        # Jitter
        jitter_scale = 0.02
        grid += rng.uniform(-jitter_scale, jitter_scale, grid.shape)
        
        x0 = grid.flatten()
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-9, 'disp': False})
            
            pts = res.x.reshape(-1, 3)
            # Clip to ensure validity
            pts = np.clip(pts, -0.5 + 1e-9, 0.5 - 1e-9)
            
            r = get_radius(pts)
            
            if r > best_radius:
                best_radius = r
                best_pts = pts
        except Exception:
            continue
            
    if best_pts is None:
        # Fallback to simple grid
        m = 5
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N]
        
    return best_pts
# EVOLVE-BLOCK-END
