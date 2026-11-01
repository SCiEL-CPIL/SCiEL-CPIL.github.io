import numpy as np
import math
import time
from scipy.optimize import minimize_scalar

N = 11
SQRT3 = math.sqrt(3.0)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit (using 115s)
    
    # 1. Define the rigid cluster of 11 unit hexagons
    # Using axial coordinates for a symmetric packing
    # Center: (0,0)
    # Ring 1: 6 neighbors
    # Ring 2: 4 specific positions to form a compact shape
    # This specific set is known to be a good packing for N=11.
    
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
    
    # The inner hexagons are rotated by 30 degrees relative to the standard orientation.
    # Why? If we use standard orientation (flat top/bottom), neighbors at distance SQRT3 (axial distance 1)
    # touch at vertices. To pack them tighter or just to define the shape clearly, 
    # rotating by 30 degrees makes them "pointy top".
    # However, the relative geometry between inner hexagons is fixed by their centers.
    # The angle of the inner hexagons matters for the "envelope" shape.
    # Let's fix the inner hexagon angle to 30 degrees (pi/6) for all of them.
    # This is a common optimal orientation for packings.
    inner_angle_rad = math.pi / 6.0
    inner_angle_deg = 30.0
    
    # Vertices of a single unit hexagon relative to its center, rotated by inner_angle_rad
    # Standard vertices (flat-top, angle 0): (1,0), (0.5, sqrt3/2), ...
    # Let's generate vertices for a pointy-top hexagon (angle 30 deg)
    # Or just rotate the standard ones.
    # Standard (flat-top) vertices:
    verts_std = np.array([
        [1.0, 0.0],
        [0.5, SQRT3 / 2.0],
        [-0.5, SQRT3 / 2.0],
        [-1.0, 0.0],
        [-0.5, -SQRT3 / 2.0],
        [0.5, -SQRT3 / 2.0]
    ])
    
    c, s = math.cos(inner_angle_rad), math.sin(inner_angle_rad)
    R_inner = np.array([[c, -s], [s, c]])
    verts = verts_std @ R_inner.T  # Rotate each vertex
    
    # Pre-compute all vertices for all 11 hexagons relative to the global origin
    # Shape: (11, 6, 2)
    all_verts = np.zeros((N, 6, 2))
    for i in range(N):
        all_verts[i] = base_pos[i] + verts
        
    # Flatten to (66, 2)
    flat_verts = all_verts.reshape(-1, 2)

    # 2. Define the objective function: Minimal side length S of the outer hexagon
    # The outer hexagon is centered at (0,0) but can be rotated by angle phi.
    # For a regular hexagon with side length S and rotation phi, the constraints are:
    # | v dot n_k(phi) | <= S * SQRT3 / 2  for k=0..5
    # where n_k(phi) are the outward normals of the hexagon sides.
    # n_k(phi) = (cos(phi + k*pi/3), sin(phi + k*pi/3))
    
    # The required side length for a given phi is:
    # S(phi) = (2/SQRT3) * max_k max_v | v dot n_k(phi) |
    
    # We want to minimize S(phi) over phi in [0, pi/3).
    # Note: Due to symmetry, we only need to search 60 degrees.
    
    def compute_S(phi):
        # Generate normals for the 6 sides
        angles = phi + np.arange(6) * (math.pi / 3.0)
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        
        # Calculate projections of all vertices onto all normals
        # flat_verts: (66, 2)
        # normals: (6, 2)
        # projs: (66, 6)
        projs = flat_verts @ normals.T
        
        # Take absolute values
        abs_projs = np.abs(projs)
        
        # Max projection for each normal
        max_projs = np.max(abs_projs, axis=0)
        
        # Overall max
        P = np.max(max_projs)
        
        S = (2.0 / SQRT3) * P
        return S

    # 3. Optimize phi
    # Initial guess: 0
    best_phi = 0.0
    best_S = compute_S(0.0)
    
    # Use scipy minimize_scalar
    try:
        # We can try multiple local optimizations starting from different points to be safe
        # But the function is likely unimodal or has few minima in [0, pi/3].
        # Let's do a coarse grid search first to find a good starting point.
        
        thetas = np.linspace(0, math.pi/3, 50)
        min_S = float('inf')
        min_th = 0.0
        for th in thetas:
            s_val = compute_S(th)
            if s_val < min_S:
                min_S = s_val
                min_th = th
        
        # Now refine around min_th
        res = minimize_scalar(compute_S, bounds=(min_th - 0.1, min_th + 0.1), method='bounded', options={'xatol': 1e-8})
        
        if res.success:
            best_S = res.fun
            best_phi = res.x
            
        # Also check the boundaries just in case (though periodicity handles it)
        # S(0) and S(pi/3) should be similar due to symmetry?
        # Actually, the cluster is not fully 6-fold symmetric, so S(phi) might not be symmetric around 0 or pi/3.
        # But the search space is [0, pi/3).
        
    except Exception as e:
        # Fallback to the grid search result
        best_S = min_S
        best_phi = min_th

    # 4. Construct the final output
    # Inner hexagons: positions are base_pos, angle is inner_angle_deg
    inner = []
    for i in range(N):
        inner.append([float(base_pos[i][0]), float(base_pos[i][1]), inner_angle_deg])
        
    # Outer hexagon: center (0,0), side length best_S, rotation best_phi
    return inner, (0.0, 0.0), float(best_S), float(best_phi)
# EVOLVE-BLOCK-END
