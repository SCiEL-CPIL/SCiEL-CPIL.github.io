import numpy as np
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 110.0 # Safety margin
    
    # Define the 11 hexagon centers
    # Cluster: 7-hex flower + 4 corners
    sqrt3 = np.sqrt(3)
    centers = [
        (0.0, 0.0),
        (2.0, 0.0),
        (1.0, sqrt3),
        (-1.0, sqrt3),
        (-2.0, 0.0),
        (-1.0, -sqrt3),
        (1.0, -sqrt3),
        (3.0, sqrt3),
        (3.0, -sqrt3),
        (-3.0, sqrt3),
        (-3.0, -sqrt3)
    ]
    
    # All inner hexagons have angle 0 (vertices at 0, 60, ...)
    inner = []
    for x, y in centers:
        inner.append([x, y, 0.0])
        
    # Precompute inner vertices
    # Angle 0: vertices at 0, 60, 120, 180, 240, 300
    angles_rad = np.arange(6) * np.pi / 3
    cos_a = np.cos(angles_rad)
    sin_a = np.sin(angles_rad)
    
    inner_verts = []
    for x, y, _ in inner:
        verts = np.column_stack((x + cos_a, y + sin_a))
        inner_verts.append(verts)
    inner_verts = np.array(inner_verts) # shape (11, 6, 2)

    # Function to check if all inner verts are inside outer hex
    def is_valid(S, outer_angle_deg):
        theta = np.radians(outer_angle_deg)
        # Normals to the sides of the outer hexagon
        # Side midpoints are at angles theta + k*pi/3 + pi/6
        # Distance from center to side is R = S * sqrt(3) / 2
        R = S * np.sqrt(3) / 2
        
        # Generate the 6 normal angles
        normal_angles = theta + np.arange(6) * np.pi / 3 + np.pi / 6
        cos_n = np.cos(normal_angles)
        sin_n = np.sin(normal_angles)
        
        # For each inner vertex, check if it satisfies x*cos(n) + y*sin(n) <= R for all 6 normals
        # inner_verts shape (11, 6, 2)
        # We can vectorize:
        # Dots = (inner_verts[..., 0] * cos_n[None, None, :] + inner_verts[..., 1] * sin_n[None, None, :])
        # Shape of Dots: (11, 6, 6) -> 11 hexes, 6 verts each, 6 normals each
        # Check if any dot > R
        
        dots = inner_verts[..., 0] * cos_n[None, None, :] + inner_verts[..., 1] * sin_n[None, None, :]
        return np.all(dots <= R + 1e-9)

    best_S = 10.0
    best_angle = 0.0
    
    # Coarse search
    # Try angles 0 to 30 (symmetry of hexagon is 60, but cluster is not symmetric, so 0-60 is safe, 0-30 is enough due to mirror symmetry if cluster was symmetric, but it's not fully. 0-60 is safer.)
    for angle_try in np.arange(0, 60, 5):
        # Binary search for S
        lo = 3.0
        hi = 10.0
        # Check if hi is valid
        if not is_valid(hi, angle_try):
            continue
        for _ in range(20):
            if time.time() > deadline:
                break
            mid = (lo + hi) / 2
            if is_valid(mid, angle_try):
                hi = mid
            else:
                lo = mid
        if hi < best_S:
            best_S = hi
            best_angle = angle_try
            
    return inner, (0.0, 0.0), best_S, best_angle
# EVOLVE-BLOCK-END
