import numpy as np
from itertools import combinations

def construct():
    sqrt3 = np.sqrt(3.0)
    # Lattice vectors for flat-to-flat hexagonal packing
    v1 = np.array([sqrt3, 0.0])
    v2 = np.array([sqrt3 / 2.0, 1.5])
    
    # Core: 7 hexagons (center + 6 neighbors)
    core_axial = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]
    core_pts = [i * v1 + j * v2 for i, j in core_axial]
    
    # Ring 2: 12 hexagons surrounding the core
    ring2_axial = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            # Hex distance in axial coordinates
            dist = (abs(i) + abs(j) + abs(i + j)) / 2.0
            if dist == 2.0:
                ring2_axial.append((i, j))
                
    ring2_pts = [i * v1 + j * v2 for i, j in ring2_axial]
    
    best_L = float('inf')
    best_config = None
    best_angle_rad = 0.0
    
    # We need 11 hexagons total. We have 7 in core. We need 4 from ring2.
    # Try all combinations of 4 points from the 12 in ring2.
    for combo in combinations(range(12), 4):
        pts = core_pts + [ring2_pts[i] for i in combo]
        
        # Find the minimal enclosing regular hexagon side length
        min_h = float('inf')
        best_alpha_for_combo = 0.0
        
        # Scan rotation angle of the container hexagon
        # Symmetry of hexagon is 60 deg, so we only need to scan 0 to 30 deg.
        # Step 0.5 degrees
        for alpha in np.linspace(0.0, np.pi / 6.0, 61):
            # Calculate required apothem h for this angle
            # h = max over all points p and all 6 normals n_k of |p . n_k|
            h_curr = 0.0
            for p in pts:
                max_dot = 0.0
                for k in range(6):
                    angle_k = alpha + k * np.pi / 3.0
                    nx = np.cos(angle_k)
                    ny = np.sin(angle_k)
                    dot = abs(p[0] * nx + p[1] * ny)
                    if dot > max_dot:
                        max_dot = dot
                if max_dot > h_curr:
                    h_curr = max_dot
            
            if h_curr < min_h:
                min_h = h_curr
                best_alpha_for_combo = alpha
                
        # Convert apothem h to side length L
        L = (2.0 / sqrt3) * min_h
        
        if L < best_L:
            best_L = L
            best_config = pts
            best_angle_rad = best_alpha_for_combo

    # Construct the return values
    inner = []
    for p in best_config:
        # Inner hexagons are aligned with the lattice (angle 0)
        inner.append([float(p[0]), float(p[1]), 0.0])
        
    center = (0.0, 0.0)
    side = float(best_L)
    angle_deg = float(np.degrees(best_angle_rad))
    
    return inner, center, side, angle_deg
