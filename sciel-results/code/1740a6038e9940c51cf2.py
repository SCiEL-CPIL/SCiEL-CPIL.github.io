# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import itertools

N = 11

V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = 0.5 * np.sqrt(3) / 2  # area of unit equilateral triangle

def min_triangle_area_vec(points):
    """
    Vectorized calculation of the minimum triangle area.
    points: (N, 2) array
    """
    # Calculate all pairwise differences
    # diff[i, j, k] = points[k] - points[i] ? No, we need triples.
    # Let's use broadcasting to compute areas for all i < j < k.
    # This is O(N^3) but vectorized.
    # N=11, so 11*10*9/6 = 165 triangles. Very small.
    # We can just do it with a loop or vectorized indices.
    
    n = len(points)
    indices = np.arange(n)
    # Generate all unique triplets i, j, k with i < j < k
    i_idx, j_idx, k_idx = np.meshgrid(indices, indices, indices, indexing='ij')
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_flat = i_idx[mask]
    j_flat = j_idx[mask]
    k_flat = k_idx[mask]
    
    p_i = points[i_flat]
    p_j = points[j_flat]
    p_k = points[k_flat]
    
    # Cross product magnitude / 2
    # (p_j - p_i) x (p_k - p_i)
    vec1 = p_j - p_i
    vec2 = p_k - p_i
    cross = vec1[:, 0] * vec2[:, 1] - vec1[:, 1] * vec2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def objective(x):
    """
    Objective function: negative of minimum area + penalty for out-of-bounds.
    x: flat array of length 2*N
    """
    points = x.reshape(N, 2)
    
    # Penalty for being outside the triangle
    # Triangle inequalities for barycentric coords or direct line checks
    # y >= 0
    # y <= sqrt(3) * x
    # y <= -sqrt(3) * x + sqrt(3)
    
    penalty = 0.0
    # Vectorized penalty calculation
    y = points[:, 1]
    x_coords = points[:, 0]
    
    # dist to y=0
    pen1 = np.maximum(0, -y)
    # dist to y = sqrt(3)x => sqrt(3)x - y >= 0
    pen2 = np.maximum(0, y - np.sqrt(3) * x_coords)
    # dist to y = -sqrt(3)x + sqrt(3) => -sqrt(3)x + sqrt(3) - y >= 0
    pen3 = np.maximum(0, y - (-np.sqrt(3) * x_coords + np.sqrt(3)))
    
    # Sum of squared penalties
    penalty = np.sum(pen1**2) + np.sum(pen2**2) + np.sum(pen3**2)
    
    if penalty > 1e-6:
        return penalty # Return large value if invalid
    
    area = min_triangle_area_vec(points)
    return -(area / TRI_AREA)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_x = None
    best_score = -float('inf')
    
    # Generate initial points
    # 1. Grid initialization
    # 2. Random initialization
    
    inits = []
    
    # Grid
    # Create a grid of points in the triangle
    # Use barycentric coordinates for grid generation
    # u, v, w >= 0, u+v+w=1
    # Let's just use a simple Cartesian grid and filter
    x_grid = np.linspace(0, 1, 10)
    y_grid = np.linspace(0, np.sqrt(3)/2, 10)
    for x in x_grid:
        for y in y_grid:
            # Check if inside
            if y >= 0 and y <= np.sqrt(3)*x and y <= -np.sqrt(3)*x + np.sqrt(3):
                # Create a configuration by placing points on this grid?
                # No, we need N=11 points.
                # Strategy: Place points at specific locations.
                # Let's create a few "templates"
                pass
    
    # Better strategy for inits:
    # 1. All points at centroid (bad)
    # 2. Points distributed on vertices and edges
    # 3. Random points
    
    # Template 1: Vertices + Centroid + Others
    # Template 2: Random
    
    rng = np.random.RandomState(42)
    
    # Generate 20 random initial configurations
    for _ in range(20):
        pts = np.zeros((N, 2))
        for i in range(N):
            # Random point in triangle
            u = rng.random()
            v = rng.random()
            while u + v > 1.0:
                u = rng.random()
                v = rng.random()
            pts[i, 0] = u + 0.5 * v
            pts[i, 1] = np.sqrt(3) / 2 * v
        inits.append(pts.flatten())
        
    # Add some structured inits
    # E.g., points evenly spaced on a grid inside the triangle
    # Let's try a few "magic" configurations
    # 1. Vertices, Midpoints, Centroid
    # V0, V1, V2
    # Midpoints: (0.5, 0), (0.75, sqrt(3)/4), (0.25, sqrt(3)/4)
    # Centroid: (0.5, sqrt(3)/6)
    # That's 7 points. Need 4 more.
    
    # Let's just run NM on the random inits.
    
    for init in inits:
        if time.time() > deadline:
            break
            
        try:
            res = minimize(
                objective,
                init,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            
            x_opt = res.x.reshape(N, 2)
            
            # Verify validity
            valid = True
            for i in range(N):
                x_i, y_i = x_opt[i]
                if y_i < -1e-9 or y_i > np.sqrt(3) * x_i + 1e-9 or y_i > -np.sqrt(3) * x_i + np.sqrt(3) + 1e-9:
                    valid = False
                    break
            
            if valid:
                score = min_triangle_area_vec(x_opt) / TRI_AREA
                if score > best_score:
                    best_score = score
                    best_x = x_opt
        except Exception:
            pass

    if best_x is None:
        # Fallback
        best_x = np.array([
            [0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3)/2],
            [0.5, np.sqrt(3)/6], [0.25, np.sqrt(3)/12], [0.75, np.sqrt(3)/12], [0.5, np.sqrt(3)/4],
            [0.125, np.sqrt(3)/24], [0.875, np.sqrt(3)/24], [0.375, np.sqrt(3)/8], [0.625, np.sqrt(3)/8]
        ])
        
    return best_x
# EVOLVE-BLOCK-END
