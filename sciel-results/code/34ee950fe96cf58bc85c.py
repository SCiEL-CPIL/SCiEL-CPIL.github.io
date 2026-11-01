import numpy as np
import time
import sys

N = 11
SQRT3 = np.sqrt(3)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # 1. Define the rigid cluster of 11 unit hexagons
    # Using axial coordinates for a symmetric packing
    # Center: (0,0)
    # Ring 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Ring 2: (2,0), (0,2), (-2,0), (0,-2)
    # Note: In axial coords, distance between (q1,r1) and (q2,r2) is (|dq|+|dr|+|dq+dr|)/2
    # This specific set forms a compact cluster.
    
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (0, 2), (-2, 0), (0, -2)
    ]
    
    # Convert to Cartesian
    # x = SQRT3 * (q + r/2)
    # y = 1.5 * r
    base_pos = []
    for q, r in axial_coords:
        x = SQRT3 * (q + r / 2.0)
        y = 1.5 * r
        base_pos.append([x, y])
        
    base_pos = np.array(base_pos)
    
    # Expand slightly to ensure strict non-overlap (distance > SQRT3)
    # This prevents floating point errors from being interpreted as overlap
    expansion_factor = 1.0 + 1e-5
    base_pos = base_pos * expansion_factor
    
    # 2. Define the outer hexagon normals (fixed orientation, angle 0)
    # Normals point outwards from the center
    normals = np.array([
        [1.0, 0.0],
        [0.5, SQRT3 / 2.0],
        [-0.5, SQRT3 / 2.0],
        [-1.0, 0.0],
        [-0.5, -SQRT3 / 2.0],
        [0.5, -SQRT3 / 2.0]
    ])
    
    # Vertices of a single unit hexagon (angle 0)
    verts = np.array([
        [1.0, 0.0],
        [0.5, SQRT3 / 2.0],
        [-0.5, SQRT3 / 2.0],
        [-1.0, 0.0],
        [-0.5, -SQRT3 / 2.0],
        [0.5, -SQRT3 / 2.0]
    ])
    
    # Pre-compute all vertices for all 11 hexagons relative to their centers
    # Shape: (11, 6, 2)
    all_verts = np.zeros((N, 6, 2))
    for i in range(N):
        all_verts[i] = base_pos[i] + verts
        
    # Function to compute the required outer side length S given a rotation angle theta
    def compute_S(theta):
        # Rotation matrix
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        
        # Rotate all vertices
        # all_verts shape (11, 6, 2)
        # R shape (2, 2)
        # Result shape (11, 6, 2)
        rotated_verts = np.einsum('ij,klj->kli', R, all_verts)
        
        # Project onto all 6 normals
        # rotated_verts shape (11, 6, 2)
        # normals shape (6, 2)
        # We want max projection for each of the 6 normals across all 11*6 vertices
        # Flatten to (66, 2)
        flat_verts = rotated_verts.reshape(-1, 2)
        
        # Projections: (66, 6)
        projs = flat_verts @ normals.T
        
        # Max projection for each normal
        max_projs = np.max(projs, axis=0)
        
        # The side length S of the outer hexagon is determined by the max projection
        # For a regular hexagon with side S, the distance from center to side (apothem) is S * sqrt(3)/2
        # The projection of a point (x,y) onto normal n is x*n_x + y*n_y
        # If max projection is P, then P = S * sqrt(3)/2
        # So S = 2 * P / sqrt(3)
        
        P = np.max(max_projs)
        # Ensure P is positive (it should be, as hexagons have extent)
        S = (2.0 / SQRT3) * P
        
        return S
    
    # 3. Optimize the rotation angle theta
    # The function is periodic with period pi/3 (60 degrees) due to hexagonal symmetry
    # We can search in [0, pi/3]
    
    best_theta = 0.0
    best_S = compute_S(0.0)
    
    # Use scipy minimize for 1D problem
    try:
        from scipy.optimize import minimize_scalar
        
        res = minimize_scalar(compute_S, bounds=(0, np.pi/3), method='bounded', options={'xatol': 1e-8})
        
        if res.success and res.fun < best_S:
            best_S = res.fun
            best_theta = res.x
            
    except Exception as e:
        # Fallback: simple grid search if scipy fails
        thetas = np.linspace(0, np.pi/3, 1000)
        for th in thetas:
            s_val = compute_S(th)
            if s_val < best_S:
                best_S = s_val
                best_theta = th

    # 4. Construct the final output
    # Rotate the base positions by best_theta
    c, s = np.cos(best_theta), np.sin(best_theta)
    R = np.array([[c, -s], [s, c]])
    
    final_pos = np.array([R @ p for p in base_pos])
    
    inner = []
    for i in range(N):
        inner.append([float(final_pos[i][0]), float(final_pos[i][1]), float(best_theta * 180.0 / np.pi)])
        
    # Outer hexagon center is (0,0), side is best_S, rotation is 0.0
    return inner, (0.0, 0.0), float(best_S), 0.0
# EVOLVE-BLOCK-END
