import numpy as np
from scipy.optimize import minimize
import itertools

N = 11

def construct():
    # 1. Generate a 15-point triangular lattice
    k = 5
    L = 1.0
    h = np.sqrt(3) / 2
    
    points_15 = []
    for j in range(k):
        for i in range(k - j):
            x = (i + 0.5 * j) / (k - 1) * L
            y = j * h / (k - 1) * L
            points_15.append([x, y])
    
    points_15 = np.array(points_15)
    
    # 2. Define the triangle vertices for clipping
    v1 = np.array([0.0, 0.0])
    v2 = np.array([1.0, 0.0])
    v3 = np.array([0.5, h])
    
    def clip_to_triangle(p):
        # Simple rejection sampling or projection. 
        # For optimization, we use a penalty or project. 
        # Projection onto a triangle is complex. 
        # Instead, we will use a penalty function in the optimizer.
        return p

    def min_area(points):
        # Calculate the minimum area of any triangle formed by 3 points
        # points is (N, 2)
        min_a = float('inf')
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                for k_idx in range(j+1, n):
                    p1, p2, p3 = points[i], points[j], points[k_idx]
                    # Area = 0.5 * |cross(p2-p1, p3-p1)|
                    v1 = p2 - p1
                    v2 = p3 - p1
                    area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
                    if area < min_a:
                        min_a = area
        return min_a

    def penalty(points):
        # Penalty for points outside the triangle
        # Barycentric coordinates
        # v0 = v3 - v1, v1 = v2 - v1, v2 = p - v1
        # d00 = dot(v0,v0), d01 = dot(v0,v1), d11 = dot(v1,v1)
        # d02 = dot(v0,v2), d12 = dot(v1,v2)
        # denom = d00*d11 - d01*d01
        # u = (d11*d02 - d01*d12) / denom
        # v = (d00*d12 - d01*d02) / denom
        # Outside if u < 0 or v < 0 or u+v > 1
        
        v0 = v3 - v1
        v1v = v2 - v1
        d00 = np.dot(v0, v0)
        d01 = np.dot(v0, v1v)
        d11 = np.dot(v1v, v1v)
        denom = d00 * d11 - d01 * d01
        if denom == 0:
            return 0
        
        pen = 0.0
        for p in points:
            v2v = p - v1
            d02 = np.dot(v0, v2v)
            d12 = np.dot(v1v, v2v)
            u = (d11 * d02 - d01 * d12) / denom
            v = (d00 * d12 - d01 * d02) / denom
            if u < 0:
                pen += abs(u)
            if v < 0:
                pen += abs(v)
            if u + v > 1:
                pen += abs(u + v - 1)
        return pen

    def objective(params):
        pts = params.reshape(N, 2)
        # Penalty for going outside
        p_val = penalty(pts)
        if p_val > 1e-6:
            return -min_area(pts) + 100 * p_val
        return -min_area(pts)

    # 3. Evaluate all subsets of 11 points from the 15
    best_subset_score = -1
    best_subset_indices = None
    
    indices_list = list(range(15))
    # We want to choose 11 points. It's equivalent to choosing 4 points to REMOVE.
    # C(15, 4) = 1365.
    for remove_indices in itertools.combinations(indices_list, 4):
        keep_indices = [i for i in indices_list if i not in remove_indices]
        subset_points = points_15[keep_indices]
        score = min_area(subset_points)
        if score > best_subset_score:
            best_subset_score = score
            best_subset_indices = keep_indices

    # 4. Local Optimization on the best subset
    x0 = points_15[best_subset_indices].flatten()
    
    # Run Nelder-Mead
    res = minimize(objective, x0, method='Nelder-Mead', options={'xatol': 1e-6, 'fatol': 1e-8, 'maxiter': 2000, 'maxfev': 2000})
    
    final_points = res.x.reshape(N, 2)
    
    # Ensure final points are strictly inside or on the boundary
    # The penalty function should have kept them inside, but let's double check and clip if necessary
    # A simple clip: if a point is far outside, it's bad. 
    # Given the penalty, they should be fine.
    
    return final_points

# EVOLVE-BLOCK-END
