import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time
import sys

N = 14

def get_score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    points: np.array of shape (N, 2)
    """
    # Calculate all triangle areas
    # Vectorized calculation
    # i, j, k indices
    # Area = 0.5 * | (y_j - y_i)(x_k - x_i) - (y_k - y_i)(x_j - x_i) |
    
    # To be efficient, we can loop over i and vectorize j, k
    # Or just use a simple loop since N=14 is small.
    # Number of triangles is 364.
    
    min_area = float('inf')
    
    # Precompute coordinates
    x = points[:, 0]
    y = points[:, 1]
    
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                # Area of triangle (i, j, k)
                # 0.5 * abs( (x_j - x_i)*(y_k - y_i) - (x_k - x_i)*(y_j - y_i) )
                area = 0.5 * abs((x[j] - x[i]) * (y[k] - y[i]) - (x[k] - x[i]) * (y[j] - y[i]))
                if area < min_area:
                    min_area = area
    
    # Calculate Convex Hull Area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume # In 2D, volume is area
    except Exception:
        # If points are collinear or something, return 0
        return 0.0
    
    if hull_area <= 0:
        return 0.0
        
    return min_area / hull_area

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Leave 10s margin
    
    best_score = 0.0
    best_points = None
    
    # Initial configurations
    seeds = []
    
    # 1. Regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    seeds.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 2. Grid 4x4 (16 points, remove 2)
    # Create 4x4 grid
    grid = np.array([[i, j] for i in range(4) for j in range(4)], dtype=float)
    # Normalize to [0,1]
    grid = grid / 3.0
    # Remove 2 points (e.g., corners) to get 14
    # Actually, let's just take first 14
    seeds.append(grid[:14])
    
    # 3. Hexagonal packing approximation
    # Center (0,0), ring 1 (6 points), ring 2 (12 points) -> 19 points.
    # Take center + ring 1 + 7 from ring 2?
    # Let's just use a spiral or random for variety.
    rng = np.random.default_rng(42)
    seeds.append(rng.random((N, 2)))
    
    # 4. Another random
    rng = np.random.default_rng(123)
    seeds.append(rng.random((N, 2)))
    
    # 5. Grid 3x5 = 15 points, remove 1
    grid2 = np.array([[i, j] for i in range(5) for j in range(3)], dtype=float)
    grid2 = grid2 / 4.0 # x in [0,1], y in [0, 0.5]
    # Stretch y
    grid2[:, 1] *= 2.0
    seeds.append(grid2[:14])
    
    # Optimization function
    def objective(p_flat):
        points = p_flat.reshape(N, 2)
        # Keep points in a reasonable box to prevent explosion, though Nelder-Mead handles it
        # Clip to [0, 1]
        points = np.clip(points, 0, 1)
        s = get_score(points)
        return -s # Minimize negative score
    
    for seed in seeds:
        if time.time() > deadline:
            break
            
        # Normalize seed to [0,1]
        seed = np.array(seed, dtype=float)
        min_val = seed.min(axis=0)
        max_val = seed.max(axis=0)
        diff = max_val - min_val
        diff[diff == 0] = 1
        seed = (seed - min_val) / diff
        # Center it? Not necessary, [0,1] is fine.
        
        x0 = seed.flatten()
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8})
            
            best_p = res.x.reshape(N, 2)
            # Clip to [0,1] again
            best_p = np.clip(best_p, 0, 1)
            
            score = get_score(best_p)
            
            if score > best_score:
                best_score = score
                best_points = best_p.copy()
                
        except Exception as e:
            pass
        
        # Check time
        if time.time() > deadline:
            break
            
    if best_points is None:
        # Fallback to first seed
        best_points = seeds[0]
        
    return best_points

# EVOLVE-BLOCK-END
