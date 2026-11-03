"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def construct():
    """
    Construct 300 equal circles inside the unit circle using a multi-start
    local optimization approach.
    
    Strategy:
    1. Initialize points using a hexagonal lattice (dense packing).
    2. Use scipy.optimize.minimize to maximize the minimum distance (radius)
       by minimizing the negative of the minimum pair distance and boundary
       distance.
    3. Multiple restarts with different initial configurations to escape local minima.
    """
    
    def objective(params, N):
        """
        Objective function to minimize.
        We want to maximize r = min(min_{i<j} |c_i - c_j| / 2, min_i (1 - |c_i|)).
        So we minimize -r.
        """
        pts = params.reshape(-1, 2)
        
        # Boundary constraint: all points must be inside unit circle
        # 1 - |c_i| should be positive
        dist_to_origin = np.linalg.norm(pts, axis=1)
        boundary_margin = 1.0 - dist_to_origin
        
        # If any point is outside, return a large penalty
        if np.any(boundary_margin <= 0):
            return 1e10 - np.min(boundary_margin)
        
        # Pairwise distances
        # Compute all pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        # Set diagonal to infinity so it doesn't affect min
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        r = min(min_pair_dist / 2.0, np.min(boundary_margin))
        
        return -r
    
    def make_hexagon_init(n_points, scale=1.0):
        """Generate initial points in a hexagonal lattice pattern."""
        points = []
        # Generate hexagonal lattice
        s = 0.06 * scale  # spacing
        k = int(1.0 / s) + 2
        for i in range(-k, k + 1):
            for j in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * s
                y = j * s * np.sqrt(3.0) / 2.0
                r = np.hypot(x, y)
                if r <= 1.0 - s / 2.0:
                    points.append([x, y])
        
        points = np.array(points)
        
        # If we have more than n_points, select the closest to center
        if len(points) > n_points:
            dists = np.linalg.norm(points, axis=1)
            idx = np.argsort(dists)[:n_points]
            points = points[idx]
        
        # If we have fewer, pad with random points inside the circle
        if len(points) < n_points:
            while len(points) < n_points:
                theta = np.random.uniform(0, 2 * np.pi)
                r = np.sqrt(np.random.uniform(0, 1.0 - s))
                points = np.vstack([points, [[r * np.cos(theta), r * np.sin(theta)]]])
        
        return points[:n_points]
    
    def make_random_init(n_points, seed=None):
        """Generate random initial points inside the unit circle."""
        if seed is not None:
            np.random.seed(seed)
        theta = np.random.uniform(0, 2 * np.pi, n_points)
        r = np.sqrt(np.random.uniform(0, 0.95, n_points))
        points = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        return points
    
    def make_ring_init(n_points, n_rings=10):
        """Generate points in concentric rings."""
        points = []
        # Center point
        points.append([0.0, 0.0])
        
        for ring in range(1, n_rings):
            # Number of points in this ring
            if ring == 1:
                n_pts = 6
            else:
                n_pts = 6 * ring
            
            # Add more points to reach n_points
            r = 0.9 / n_rings * ring
            theta = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
            for t in theta:
                if len(points) < n_points:
                    points.append([r * np.cos(t), r * np.sin(t)])
        
        # Pad if necessary
        while len(points) < n_points:
            theta = np.random.uniform(0, 2 * np.pi)
            r = np.sqrt(np.random.uniform(0.8, 0.95))
            points.append([r * np.cos(theta), r * np.sin(theta)])
        
        return np.array(points[:n_points])
    
    best_params = None
    best_score = -np.inf
    
    # Try multiple initialization strategies
    inits = []
    
    # 1. Hexagonal lattice
    inits.append(make_hexagon_init(N, scale=1.0))
    
    # 2. Random
    inits.append(make_random_init(N, seed=42))
    
    # 3. Concentric rings
    inits.append(make_ring_init(N, n_rings=15))
    
    # 4. Another random
    inits.append(make_random_init(N, seed=123))
    
    # 5. Hexagonal with different scale
    inits.append(make_hexagon_init(N, scale=0.9))
    
    for init_pts in inits:
        # Flatten for optimizer
        x0 = init_pts.flatten()
        
        # Optimize using BFGS (quasi-Newton method)
        try:
            result = minimize(objective, x0, args=(N,), method='BFGS', 
                             options={'maxiter': 500, 'gtol': 1e-6})
            
            if result.success or result.fun < best_score:
                score = -result.fun
                if score > best_score:
                    best_score = score
                    best_params = result.x.copy()
        except Exception:
            pass
    
    # If best_params is still None, fall back to hexagonal
    if best_params is None:
        pts = make_hexagon_init(N)
        return pts
    
    pts = best_params.reshape(-1, 2)
    
    # Ensure all points are strictly inside the unit circle
    # Clip any points that might be slightly outside
    dists = np.linalg.norm(pts, axis=1)
    for i in range(len(pts)):
        if dists[i] >= 1.0:
            pts[i] = pts[i] * (0.999 / dists[i])
    
    return pts

# EVOLVE-BLOCK-END
