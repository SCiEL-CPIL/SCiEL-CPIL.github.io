import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import itertools
import random

N = 13

# EVOLVE-BLOCK-START

def calculate_score(points):
    """
    Calculate the minimum triangle area divided by the convex hull area.
    """
    if len(points) < 3:
        return 0.0
    
    # Calculate convex hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume # In 2D, volume is area
    except Exception:
        # If points are collinear or degenerate
        return 0.0
    
    if hull_area == 0:
        return 0.0
    
    # Calculate minimum triangle area
    min_tri_area = float('inf')
    # 13 points, 286 triangles
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                # Area = 0.5 * |cross(p2-p1, p3-p1)|
                v1 = p2 - p1
                v2 = p3 - p1
                cross = v1[0]*v2[1] - v1[1]*v2[0]
                area = 0.5 * abs(cross)
                if area < min_tri_area:
                    min_tri_area = area
                    
    return min_tri_area / hull_area

def optimize_configuration(base_points):
    """
    Optimize the affine transformation matrix A to maximize the score.
    A is 2x2. We fix the Frobenius norm to 2 to avoid scaling issues.
    Variables: a, b, c, d.
    Constraint: a^2 + b^2 + c^2 + d^2 = 4.
    """
    def objective(params):
        a, b, c, d = params
        A = np.array([[a, b], [c, d]])
        
        # Normalize A to have Frobenius norm 2
        norm = np.linalg.norm(A)
        if norm < 1e-6:
            return 0.0
        A = A / norm * 2.0
        
        transformed_points = base_points @ A.T
        return -calculate_score(transformed_points)
    
    # Multiple starting points
    best_score = -1
    best_params = None
    
    # Initial guesses
    starts = [
        [1.0, 0.0, 0.0, 1.0],  # Identity
        [1.0, 0.5, 0.0, 1.0],  # Shear
        [1.0, 0.0, 0.5, 1.0],  # Shear
        [1.2, 0.0, 0.0, 0.8],  # Scale
        [0.8, 0.0, 0.0, 1.2],  # Scale
        [1.0, 0.2, 0.2, 1.0],  # Mixed
    ]
    
    # Add random starts
    rng = random.Random(42)
    for _ in range(10):
        starts.append([rng.uniform(0.5, 1.5), rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5), rng.uniform(0.5, 1.5)])
    
    for start in starts:
        try:
            res = minimize(objective, start, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-6})
            score = -res.fun
            if score > best_score:
                best_score = score
                best_params = res.x
        except Exception:
            pass
            
    if best_params is None:
        return base_points
        
    a, b, c, d = best_params
    A = np.array([[a, b], [c, d]])
    norm = np.linalg.norm(A)
    A = A / norm * 2.0
    
    return base_points @ A.T

def construct():
    # Define base point sets
    # Set 1: 4x4 grid minus 3 corners
    # Grid coords: -1.5, -0.5, 0.5, 1.5
    coords = [-1.5, -0.5, 0.5, 1.5]
    all_points = np.array([(x, y) for x in coords for y in coords])
    # Remove (1.5, 1.5), (1.5, -1.5), (-1.5, 1.5)
    mask = np.ones(16, dtype=bool)
    # Indices:
    # (1.5, 1.5) is index 15
    # (1.5, -1.5) is index 12
    # (-1.5, 1.5) is index 3
    mask[15] = False
    mask[12] = False
    mask[3] = False
    base1 = all_points[mask]
    
    # Set 2: 3x3 grid plus 4 axis points
    coords3 = [-1, 0, 1]
    grid3 = np.array([(x, y) for x in coords3 for y in coords3])
    extra = np.array([[0, 2], [0, -2], [2, 0], [-2, 0]])
    base2 = np.vstack([grid3, extra])
    
    # Set 3: 4x4 grid minus 4 corners (12 pts) + center (0,0)
    mask3 = np.ones(16, dtype=bool)
    mask3[0] = False # (-1.5, -1.5)
    mask3[3] = False # (-1.5, 1.5)
    mask3[12] = False # (1.5, -1.5)
    mask3[15] = False # (1.5, 1.5)
    grid_no_corners = all_points[mask3]
    center = np.array([[0.0, 0.0]])
    base3 = np.vstack([grid_no_corners, center])
    
    # Optimize each set
    candidates = []
    for base in [base1, base2, base3]:
        opt_points = optimize_configuration(base)
        candidates.append(opt_points)
        
    # Pick the best candidate
    best_points = candidates[0]
    best_score = calculate_score(candidates[0])
    for pts in candidates[1:]:
        s = calculate_score(pts)
        if s > best_score:
            best_score = s
            best_points = pts
            
    return best_points

# EVOLVE-BLOCK-END
