# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from itertools import combinations

N = 11

def _triangle_area(p1, p2, p3):
    """Area of triangle formed by three 2D points."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))

def _min_triangle_area(points):
    """Minimum area among all triangles formed by triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = _triangle_area(points[i], points[j], points[k])
                if a < min_area:
                    min_area = a
    return min_area

def _barycentric_to_cartesian(bary_coords, A, B, C):
    """
    Convert barycentric coordinates (u, v, w) for each point to Cartesian.
    bary_coords: (N, 3) array where each row sums to 1.
    A, B, C: Cartesian coordinates of triangle vertices.
    """
    # P = u*A + v*B + w*C
    # bary_coords shape (N, 3)
    # A, B, C shape (2,)
    # Result shape (N, 2)
    return bary_coords[:, 0:1] * A[None, :] + bary_coords[:, 1:2] * B[None, :] + bary_coords[:, 2:3] * C[None, :]

def construct():
    """
    Construct 11 points in the equilateral triangle (0,0), (1,0), (0.5, sqrt(3)/2)
    maximizing the minimum triangle area.
    
    Uses barycentric coordinates with regularization to a base grid topology.
    """
    sqrt3 = np.sqrt(3.0)
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    C = np.array([0.5, sqrt3 / 2.0])
    
    # Define a base grid in barycentric coordinates.
    # We want a 4-4-3 structure (bottom to top) or similar.
    # Barycentric coords (u, v, w) correspond to weights for A, B, C.
    # y-coordinate is proportional to w (height).
    # Let's define rows by w values.
    # Row 0 (bottom, w=0): Points on AB edge.
    # Row 1 (middle, w=0.5): Points in middle.
    # Row 2 (top, w=1.0): Points near C.
    
    # Since we can't put multiple points exactly at w=1 (they would coincide with C or each other if not spread),
    # we use w close to 1 for the top row.
    
    # Let's use a hexagonal-like pattern in barycentric space.
    # Base grid (u, v, w)
    base_bary = np.array([
        # Row 0 (w ~ 0.1): 4 points
        [0.1, 0.8, 0.1], # Near A
        [0.4, 0.5, 0.1],
        [0.7, 0.2, 0.1],
        [0.9, 0.0, 0.1], # Near B
        
        # Row 1 (w ~ 0.4): 4 points
        [0.2, 0.4, 0.4],
        [0.4, 0.2, 0.4],
        [0.6, 0.0, 0.4], # Wait, v=0 means on AC? No, v is weight for B.
        # Let's be more systematic.
        # Symmetry about the median from C to AB.
        # This means swapping u and v should map the set to itself.
        
        # Let's try a 3-4-4 grid?
        # Row 0 (bottom): 3 points
        # Row 1 (middle): 4 points
        # Row 2 (top): 4 points
        # Total 11.
        
        # Let's redefine base_bary with symmetry.
        # Center of mass is (1/3, 1/3, 1/3).
        
        # Row 0 (w=0.1): 4 points
        # Symmetric around u=v.
        # u+v = 0.9
        # Points: (0.1, 0.8), (0.35, 0.55), (0.55, 0.35), (0.8, 0.1)
        
        # Row 1 (w=0.4): 4 points
        # u+v = 0.6
        # Points: (0.1, 0.5), (0.3, 0.3), (0.5, 0.1), (0.6, 0.0)? No, v=0 is edge.
        # Let's use (0.15, 0.45), (0.35, 0.25), (0.55, 0.05), (0.65, -0.05)? No.
        # Let's use (0.1, 0.5), (0.3, 0.3), (0.5, 0.1), (0.6, 0.0) is bad.
        # Better: (0.2, 0.4), (0.4, 0.2), (0.5, 0.1), (0.6, 0.0) is still bad.
        # Let's use 3 points in middle? 4-3-4?
        
        # Let's stick to 4-4-3 as in the original but ensure symmetry.
        # Row 0 (w=0.1): 4 points
        # Row 1 (w=0.4): 4 points
        # Row 2 (w=0.8): 3 points
    ])
    
    # Let's construct the base grid carefully.
    # We want symmetry with respect to the axis of symmetry of the triangle (u=v).
    
    # Row 0 (bottom, w=0.1): 4 points
    # u+v = 0.9
    # Symmetric pairs: (u, v) and (v, u)
    # p0: (0.1, 0.8, 0.1)
    # p1: (0.35, 0.55, 0.1)
    # p2: (0.55, 0.35, 0.1)
    # p3: (0.8, 0.1, 0.1)
    
    # Row 1 (middle, w=0.4): 4 points
    # u+v = 0.6
    # p4: (0.1, 0.5, 0.4)
    # p5: (0.3, 0.3, 0.4)
    # p6: (0.5, 0.1, 0.4)
    # p7: (0.6, 0.0, 0.4) -> v=0 is on edge AC. Might be okay, but let's move in.
    # Let's use (0.15, 0.45), (0.35, 0.25), (0.55, 0.05), (0.65, -0.05) -> No.
    # Let's use 3 points in middle to keep symmetry simple?
    # If we use 4-3-4:
    # Row 0: 4
    # Row 1: 3
    # Row 2: 4
    # Total 11.
    
    # Let's try 4-3-4.
    # Row 0 (w=0.1): 4 points
    # Row 1 (w=0.45): 3 points
    # Row 2 (w=0.8): 4 points
    
    # Row 0 (w=0.1):
    # (0.1, 0.8), (0.35, 0.55), (0.55, 0.35), (0.8, 0.1)
    
    # Row 1 (w=0.45): u+v = 0.55
    # Symmetric 3 points:
    # Center: (0.275, 0.275, 0.45)
    # Left: (0.1, 0.45, 0.45)
    # Right: (0.45, 0.1, 0.45)
    
    # Row 2 (w=0.8): u+v = 0.2
    # Symmetric 4 points:
    # (0.05, 0.15), (0.15, 0.05) ... wait, only 2 symmetric pairs for 4 points?
    # (0.05, 0.15, 0.8), (0.15, 0.05, 0.8) are 2 points.
    # Need 2 more.
    # (0.02, 0.18, 0.8), (0.18, 0.02, 0.8)?
    # This gets crowded near the top vertex C (w=1).
    
    # Let's go back to 4-4-3.
    # Row 0 (w=0.1): 4 points
    # Row 1 (w=0.4): 4 points
    # Row 2 (w=0.7): 3 points
    
    base_bary = np.array([
        # Row 0
        [0.10, 0.80, 0.10],
        [0.35, 0.55, 0.10],
        [0.55, 0.35, 0.10],
        [0.80, 0.10, 0.10],
        
        # Row 1
        [0.10, 0.50, 0.40],
        [0.30, 0.30, 0.40],
        [0.50, 0.10, 0.40],
        [0.60, 0.00, 0.40], # On edge AC
        
        # Row 2
        [0.15, 0.15, 0.70],
        [0.35, -0.05, 0.70], # Invalid! v < 0
        [0.05, 0.25, 0.70]
    ])
    
    # The base grid above has issues. Let's use a simpler, valid base grid.
    # We will use a grid that is definitely inside.
    
    # Row 0 (w=0.1): 4 points
    # Row 1 (w=0.4): 4 points
    # Row 2 (w=0.7): 3 points
    
    # Ensure u, v, w >= 0 and sum 1.
    
    base_bary = np.array([
        # Row 0 (w=0.1)
        [0.10, 0.80, 0.10],
        [0.30, 0.60, 0.10],
        [0.60, 0.30, 0.10],
        [0.80, 0.10, 0.10],
        
        # Row 1 (w=0.4)
        [0.10, 0.50, 0.40],
        [0.30, 0.30, 0.40],
        [0.50, 0.10, 0.40],
        [0.60, 0.00, 0.40], # v=0 is valid (on edge)
        
        # Row 2 (w=0.7)
        [0.15, 0.15, 0.70],
        [0.30, 0.00, 0.70], # v=0
        [0.00, 0.30, 0.70]  # u=0
    ])
    
    # This grid is symmetric?
    # Row 0: (0.1, 0.8) and (0.8, 0.1) are symmetric. (0.3, 0.6) and (0.6, 0.3) are symmetric.
    # Row 1: (0.1, 0.5) and (0.5, 0.1) are symmetric. (0.3, 0.3) is symmetric. (0.6, 0.0) is not symmetric to a point in the set.
    # We need to fix Row 1 to be symmetric.
    # If we have 4 points, we can have 2 symmetric pairs.
    # Pair 1: (0.1, 0.5) and (0.5, 0.1).
    # Pair 2: (0.3, 0.3) and ... another (0.3, 0.3)? No, points must be distinct.
    # So 4 points in a symmetric set must be 2 pairs.
    # (0.2, 0.4) and (0.4, 0.2).
    # (0.3, 0.3) and ... we need another one.
    # Maybe (0.15, 0.45) and (0.45, 0.15)?
    # Let's use:
    # (0.15, 0.45, 0.40)
    # (0.45, 0.15, 0.40)
    # (0.30, 0.30, 0.40) -- this is on the axis.
    # We need one more.
    # (0.25, 0.35, 0.40) and (0.35, 0.25, 0.40)? That's 4 points.
    
    # Let's use 3 points in Row 1 to make it easier? 4-3-4 = 11.
    # Row 0: 4 points (2 pairs)
    # Row 1: 3 points (1 pair + 1 center)
    # Row 2: 4 points (2 pairs)
    
    base_bary = np.array([
        # Row 0 (w=0.1)
        [0.10, 0.80, 0.10],
        [0.30, 0.60, 0.10],
        [0.60, 0.30, 0.10],
        [0.80, 0.10, 0.10],
        
        # Row 1 (w=0.45)
        [0.15, 0.40, 0.45],
        [0.30, 0.25, 0.45], # Center-ish? No, u!=v.
        # Let's use:
        # (0.1, 0.45), (0.45, 0.1), (0.275, 0.275)
        [0.10, 0.45, 0.45],
        [0.45, 0.10, 0.45],
        [0.275, 0.275, 0.45],
        
        # Row 2 (w=0.8)
        # 4 points. 2 pairs.
        [0.05, 0.15, 0.80],
        [0.15, 0.05, 0.80],
        [0.10, 0.10, 0.80], # On axis
        [0.20, 0.00, 0.80] # Not symmetric.
        
        # Let's use 3 points in Row 2? 4-4-3.
        # Row 2 (w=0.8): 3 points.
        # (0.1, 0.1), (0.2, 0.0), (0.0, 0.2)
    ])
    
    # Let's restart the base grid definition to be clean and symmetric.
    # 4-4-3 Grid.
    # Row 0 (w=0.1): 4 points. Symmetric.
    # Row 1 (w=0.4): 4 points. Symmetric.
    # Row 2 (w=0.7): 3 points. Symmetric.
    
    base_bary = np.array([
        # Row 0
        [0.10, 0.80, 0.10],
        [0.30, 0.60, 0.10],
        [0.60, 0.30, 0.10],
        [0.80, 0.10, 0.10],
        
        # Row 1
        [0.15, 0.45, 0.40],
        [0.35, 0.25, 0.40],
        [0.25, 0.35, 0.40],
        [0.45, 0.15, 0.40],
        
        # Row 2
        [0.15, 0.15, 0.70],
        [0.25, 0.05, 0.70],
        [0.05, 0.25, 0.70]
    ])
    
    # Check symmetry:
    # Row 0: (0.1, 0.8)<->(0.8, 0.1), (0.3, 0.6)<->(0.6, 0.3). OK.
    # Row 1: (0.15, 0.45)<->(0.45, 0.15), (0.35, 0.25)<->(0.25, 0.35). OK.
    # Row 2: (0.15, 0.15) on axis. (0.25, 0.05)<->(0.05, 0.25). OK.
    
    # Optimization function
    # Variables: barycentric coordinates for all 11 points (33 vars)
    # Constraint: u+v+w=1 for each point.
    # We can parameterize with 22 vars (u, v) and set w = 1-u-v.
    # And add penalty if u<0, v<0, w<0.
    
    def objective(params):
        # params: 22 values (u, v) for each point
        uv = params.reshape(11, 2)
        u = uv[:, 0]
        v = uv[:, 1]
        w = 1.0 - u - v
        
        # Penalty for out of bounds
        pen = 0.0
        if np.any(u < 0) or np.any(v < 0) or np.any(w < 0):
            pen += 10.0 * (np.sum(np.minimum(u, 0)**2) + np.sum(np.minimum(v, 0)**2) + np.sum(np.minimum(w, 0)**2))
        
        # Regularization: keep close to base
        base_uv = base_bary[:, 0:2]
        reg = 0.1 * np.sum((uv - base_uv)**2)
        
        # Convert to Cartesian
        bary = np.column_stack((u, v, w))
        pts = _barycentric_to_cartesian(bary, A, B, C)
        
        # Min area
        min_area = _min_triangle_area(pts)
        
        return -min_area + pen + reg
    
    # Initial guess
    x0 = base_bary[:, 0:2].flatten()
    
    # Optimize
    res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-10})
    
    # Extract best points
    best_uv = res.x.reshape(11, 2)
    best_u = best_uv[:, 0]
    best_v = best_uv[:, 1]
    best_w = 1.0 - best_u - best_v
    
    # Clip to valid range just in case
    best_u = np.clip(best_u, 0, 1)
    best_v = np.clip(best_v, 0, 1)
    best_w = np.clip(best_w, 0, 1)
    
    # Normalize in case of slight drift
    sums = best_u + best_v + best_w
    best_u /= sums
    best_v /= sums
    best_w /= sums
    
    best_bary = np.column_stack((best_u, best_v, best_w))
    best_pts = _barycentric_to_cartesian(best_bary, A, B, C)
    
    return np.array(best_pts)
# EVOLVE-BLOCK-END
