import numpy as np
import time
from scipy.optimize import minimize

N = 11

# Vertices of the equilateral triangle
# Area = 1/2 * base * height = 1/2 * 1 * sqrt(3)/2 = sqrt(3)/4
# The task says "unit-area triangle" in the text but the interface says "equilateral triangle (0,0), (1,0), (1/2, sqrt(3)/2)".
# Let's check the area of the given triangle.
# Base = 1, Height = sqrt(3)/2. Area = sqrt(3)/4.
# The task says "divides by its area".
# The evaluator measures the raw value.
# "The notebook uses an equilateral triangle and divides by its area."
# This implies the score is (min_area_found / min_area_seed) * (Area_Triangle / Area_Triangle)?
# No, "maximize the smallest area... divides by its area".
# Usually, this means the metric is normalized by the total area of the container.
# However, the interface says "The evaluator measures the raw value (higher is better). The search score is value/seed".
# If the seed program returns points, the evaluator calculates the min area of those points.
# Let's assume the evaluator calculates the min area of the triangles formed by the 11 points.
# The "divides by its area" part in the task description might be descriptive of the normalization used to define the "value" if it were a specific metric, but the contract says "measures the raw value".
# Given the seed program just returns points, the "value" is likely the minimum triangle area.
# If the task meant to normalize by the container area, the seed score would be a specific number.
# Since the seed score is 1.0, and the seed is random, the "value" must be the raw min area, and the "seed" value is the min area of the seed's points.
# So we just need to maximize the raw minimum triangle area.

# Precompute indices for all triplets
indices = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            indices.append((i, j, k))
indices = np.array(indices)

def min_triangle_area(points):
    """
    Calculate the minimum area of any triangle formed by 3 points.
    points: (N, 2) array
    """
    # points shape (N, 2)
    # indices shape (165, 3)
    p1 = points[indices[:, 0]]  # (165, 2)
    p2 = points[indices[:, 1]]  # (165, 2)
    p3 = points[indices[:, 2]]  # (165, 2)
    
    # Vector differences
    v1 = p2 - p1
    v2 = p3 - p1
    
    # Cross product (2D determinant)
    # det = v1_x * v2_y - v1_y * v2_x
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def objective(params):
    """
    Negative of the minimum triangle area.
    params: (2N,) array
    """
    points = params.reshape(N, 2)
    return -min_triangle_area(points)

def random_point_in_triangle():
    """
    Generate a random point inside the triangle (0,0), (1,0), (0.5, sqrt(3)/2).
    """
    u = np.random.rand()
    v = np.random.rand()
    if u + v > 1:
        u = 1 - u
        v = 1 - v
    x = u + 0.5 * v
    y = np.sqrt(3) / 2 * v
    return np.array([x, y])

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    best_params = None
    best_score = -np.inf
    
    # Try a few random restarts
    # Nelder-Mead is stochastic in its initial simplex, so we can just run it multiple times
    # or use different random seeds for the initial points
    
    num_restarts = 20  # Adjust based on time
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Generate random initial points
        init_points = np.array([random_point_in_triangle() for _ in range(N)])
        init_params = init_points.flatten()
        
        # Run Nelder-Mead
        try:
            res = minimize(objective, init_params, method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            
            if res.success or res.fun < best_score:
                # Check if points are still inside the triangle (Nelder-Mead might drift out)
                # If they drift out, we should project them back or discard.
                # For now, let's check validity.
                pts = res.x.reshape(N, 2)
                
                # Check if all points are inside the triangle
                # Triangle inequalities in barycentric coords or half-plane tests
                # Half-plane 1: y >= 0
                # Half-plane 2: y <= sqrt(3) * x  (Line from (0,0) to (0.5, sqrt(3)/2) is y = sqrt(3)x)
                # Half-plane 3: y <= -sqrt(3) * (x - 1) (Line from (1,0) to (0.5, sqrt(3)/2))
                
                valid = True
                for p in pts:
                    x, y = p
                    if y < -1e-9:
                        valid = False
                        break
                    if y > np.sqrt(3) * x + 1e-9:
                        valid = False
                        break
                    if y > -np.sqrt(3) * (x - 1) + 1e-9:
                        valid = False
                        break
                
                if valid:
                    # Clip tiny numerical errors to be safe
                    # Actually, if valid, we can use it.
                    score = -res.fun
                    if score > best_score:
                        best_score = score
                        best_params = res.x.copy()
                else:
                    # If invalid, try to project back? Or just ignore this result.
                    # Ignoring is safer.
                    pass
                    
        except Exception as e:
            # If an exception occurs, skip this restart
            pass
            
    if best_params is None:
        # Fallback to seed-like random points
        rng = np.random.RandomState(7)
        points = []
        while len(points) < N:
            u, v = rng.rand(), rng.rand()
            if u + v < 0.98 and u > 0.01 and v > 0.01:
                points.append([u + 0.5 * v, (3 ** 0.5) / 2 * v])
        return np.array(points)
        
    return best_params.reshape(N, 2)
