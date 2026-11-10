# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the (x,y) coordinates of the 16 points.

    """
    n = 16
    d = 2
    deadline = time.time() + 600  # Safety margin

    def score(points):
        if points.shape != (n, d):
            return 0.0
        dists = pdist(points)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-12:
            return 0.0
        return (d_min / d_max) ** 2

    def objective(flat):
        pts = flat.reshape(n, d)
        dists = pdist(pts)
        d_max = np.max(dists)
        if d_max < 1e-12:
            return 0.0
        # Normalize to unit diameter to fix scale invariance
        pts_norm = pts / d_max
        dists_norm = pdist(pts_norm)
        d_min_norm = np.min(dists_norm)
        # Maximize d_min_norm => Minimize -d_min_norm
        return -d_min_norm

    def generate_hexagonal_lattice():
        # 1-2-3-4-3-2-1 hexagonal arrangement
        # Side length 1 in lattice units
        # Coordinates for a hexagonal lattice
        # Center at (0,0)
        # Rows:
        # y = -1.5, -1, -0.5, 0, 0.5, 1, 1.5 (scaled)
        
        # Let's use a standard hex grid generator
        points = []
        # Row 0 (1 point)
        points.append([0, 0])
        
        # This is getting complicated to get exactly 16 in a nice hex shape.
        # A 4x4 square grid is 16 points.
        # A hexagonal "diamond" of radius 2 (center + 6 + 12 = 19 points) is too many.
        # Radius 1 (center + 6 = 7 points).
        # Let's take a 4x4 grid and distort it?
        # Or take the 16 points closest to origin in a hex lattice.
        
        # Hex lattice basis vectors
        a1 = np.array([1, 0])
        a2 = np.array([0.5, np.sqrt(3)/2])
        
        pts_list = []
        for i in range(-4, 5):
            for j in range(-4, 5):
                p = i * a1 + j * a2
                pts_list.append(p)
        
        # Sort by distance to origin
        pts_arr = np.array(pts_list)
        dists_to_origin = np.linalg.norm(pts_arr, axis=1)
        indices = np.argsort(dists_to_origin)
        
        # Take first 16
        hex_pts = pts_arr[indices[:16]]
        
        # Center it
        hex_pts -= np.mean(hex_pts, axis=0)
        
        return hex_pts

    def generate_ring():
        # 1 center, 15 on a circle?
        # Or 2 rings: 1 center, 6 inner, 9 outer?
        # Let's try 1 center + 15 on a circle.
        center = np.array([0, 0])
        angles = np.linspace(0, 2 * np.pi, 15, endpoint=False)
        radius = 1.0
        ring_pts = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
        all_pts = np.vstack([center, ring_pts])
        return all_pts

    def generate_random(seed):
        np.random.seed(seed)
        pts = np.random.randn(n, d)
        # Center
        pts -= np.mean(pts, axis=0)
        return pts

    initial_guesses = [
        generate_hexagonal_lattice(),
        generate_ring(),
        generate_random(42),
        generate_random(123)
    ]

    best_points = None
    best_score = -1.0

    for i, init_pts in enumerate(initial_guesses):
        if time.time() > deadline - 30:
            break
            
        # Flatten for optimizer
        x0 = init_pts.flatten()
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
            
            if res.success or res.fun < objective(x0):
                opt_pts = res.x.reshape(n, d)
                s = score(opt_pts)
                
                if s > best_score:
                    best_score = s
                    best_points = opt_pts
                    
        except Exception:
            continue
            
    if best_points is None:
        # Fallback to the best initial guess
        for init_pts in initial_guesses:
            s = score(init_pts)
            if s > best_score:
                best_score = s
                best_points = init_pts
                
    return best_points

# EVOLVE-BLOCK-END
