import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Generate candidate configurations on the hexagonal lattice
    # Lattice vectors: v1 = (sqrt(3), 0), v2 = (sqrt(3)/2, 1.5)
    # Distance between centers is sqrt(3)
    
    def get_L_from_centers(centers, orientation):
        """
        Calculate the minimum side length L of an outer hexagon
        containing the given centers of unit hexagons.
        orientation: 0 for flat-top (normals at 0, 60...), 1 for pointy-top (normals at 30, 90...)
        """
        centers = np.array(centers)
        # Center the configuration
        centers -= np.mean(centers, axis=0)
        
        # Normals for the outer hexagon
        if orientation == 0:
            # Flat top: vertices at 0, 60, ...
            # Normals are perpendicular to sides.
            # Sides are vertical? No.
            # Flat top hexagon:
            # Vertices: (L, 0), (L/2, L*sqrt(3)/2), ...
            # Side 1: between (L,0) and (L/2, L*sqrt(3)/2). Vector: (-L/2, L*sqrt(3)/2).
            # Normal: (sqrt(3)/2, 1/2) ?
            # Let's use the standard support function approach.
            # For a regular hexagon of side L, the apothem is a = L*sqrt(3)/2.
            # The condition is: max_i (C_i . n_k) + apothem_in <= apothem_out
            # apothem_in = sqrt(3)/2 (for unit hexagon)
            # apothem_out = L * sqrt(3)/2
            # So L >= (max_i(C_i . n_k) + sqrt(3)/2) / (sqrt(3)/2)
            
            # Normals for flat-top (sides are horizontal/vertical? No)
            # Flat top: Two horizontal sides, four slanted.
            # Normals are at 0, 60, 120, 180, 240, 300 degrees.
            angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180
        else:
            # Pointy top: Two vertical sides? No.
            # Pointy top: Two vertical vertices?
            # Rotated by 30 degrees.
            angles = np.array([30, 90, 150, 210, 270, 330]) * np.pi / 180
            
        normals = np.column_stack((np.cos(angles), np.sin(angles)))
        
        # Project centers onto normals
        # projections[i, k] = C_i . n_k
        projections = centers @ normals.T
        
        # Find max projection for each normal
        max_projections = np.max(projections, axis=0)
        
        # The critical normal is the one with the maximum projection
        M = np.max(max_projections)
        
        # Calculate L
        # M + sqrt(3)/2 <= L * sqrt(3)/2
        # L >= (M + sqrt(3)/2) * 2 / sqrt(3)
        L = (M + np.sqrt(3)/2) * 2 / np.sqrt(3)
        
        return L

    def make_grid(rows, cols, offset_pattern):
        """
        Create a grid of hexagons.
        offset_pattern: list of 0 or 0.5 for each row, indicating horizontal shift in units of sqrt(3)/2?
        Actually, lattice shift is sqrt(3)/2.
        """
        centers = []
        for r in range(rows):
            for c in range(cols):
                x = c * np.sqrt(3) + offset_pattern[r] * np.sqrt(3)/2
                y = r * 1.5
                centers.append([x, y])
        return centers

    candidates = []
    
    # 1. 4x3 Grid
    # Pattern: 0, 0.5, 0, 0.5
    c1 = make_grid(4, 3, [0, 0.5, 0, 0.5])
    candidates.append(c1)
    
    # 2. 3x4 Grid
    # Pattern: 0, 0.5, 0
    c2 = make_grid(3, 4, [0, 0.5, 0])
    candidates.append(c2)
    
    # 3. 2x6 Grid
    # Pattern: 0, 0.5
    c3 = make_grid(2, 6, [0, 0.5])
    candidates.append(c3)
    
    # 4. "Diamond" 1-3-3-3-1? No, 12 items.
    # Try 2-4-4-2?
    c4 = make_grid(4, 3, [0, 0.5, 0.5, 0]) # Skewed 4x3
    candidates.append(c4)
    
    # 5. Center + Ring + 5
    # Center (0,0)
    # Ring 1: 6 hexagons
    # Ring 2: Pick 5
    centers = [[0,0]]
    # Ring 1 coords (distance sqrt(3))
    ring1 = []
    for i in range(6):
        angle = i * np.pi / 3
        x = np.sqrt(3) * np.cos(angle)
        y = np.sqrt(3) * np.sin(angle)
        ring1.append([x, y])
    centers.extend(ring1)
    
    # Ring 2 coords (distance 2*sqrt(3) from center, but on lattice)
    # Lattice points at distance approx 2*sqrt(3)
    # (2*sqrt(3), 0), (sqrt(3), 1.5*sqrt(3))? No.
    # Let's just pick 5 points from the next lattice shell.
    # Lattice vectors: (sqrt(3), 0), (sqrt(3)/2, 1.5)
    # Points:
    # (2*sqrt(3), 0)
    # (1.5*sqrt(3), 1.5)
    # (sqrt(3), 3)
    # (0, 3)
    # (-sqrt(3), 3)
    # (-1.5*sqrt(3), 1.5)
    # (-2*sqrt(3), 0)
    # (-1.5*sqrt(3), -1.5)
    # (-sqrt(3), -3)
    # (0, -3)
    # (sqrt(3), -3)
    # (1.5*sqrt(3), -1.5)
    
    ring2_candidates = [
        [2*np.sqrt(3), 0],
        [1.5*np.sqrt(3), 1.5],
        [np.sqrt(3), 3],
        [0, 3],
        [-np.sqrt(3), 3],
        [-1.5*np.sqrt(3), 1.5],
        [-2*np.sqrt(3), 0],
        [-1.5*np.sqrt(3), -1.5],
        [-np.sqrt(3), -3],
        [0, -3],
        [np.sqrt(3), -3],
        [1.5*np.sqrt(3), -1.5]
    ]
    # Add first 5
    centers.extend(ring2_candidates[:5])
    candidates.append(centers)
    
    best_L = float('inf')
    best_config = None
    best_orientation = 0
    
    for config in candidates:
        for orientation in [0, 1]:
            L = get_L_from_centers(config, orientation)
            if L < best_L:
                best_L = L
                best_config = config
                best_orientation = orientation
                
    # Return the best configuration
    # inner: list of [x, y, angle_degrees]
    # center: [x, y]
    # side: L
    # angle: rotation of outer hexagon
    
    # The calculated L assumes the outer hexagon is aligned with the normals used.
    # If orientation 0 was best, outer angle is 0.
    # If orientation 1 was best, outer angle is 30.
    
    inner = []
    for x, y in best_config:
        # Inner hexagons are aligned with the lattice (angle 0)
        # But if we rotated the outer hexagon, do we rotate the inner?
        # The problem says inner may rotate.
        # In our calculation, we assumed inner are aligned with the lattice used to generate them.
        # And we checked two outer orientations.
        # So we just return the inner positions as is (angle 0) and set the outer angle.
        inner.append([x, y, 0.0])
        
    # Center the config for the return value?
    # The evaluator likely expects the center of the outer hexagon.
    # We centered the points in get_L_from_centers, but modified a copy.
    # We need to return the actual coordinates.
    # Let's re-center best_config for the output.
    centers_arr = np.array(best_config)
    center = np.mean(centers_arr, axis=0)
    # Subtract center to make it centered at 0 for the inner list?
    # The interface says: inner rows (x, y, angle), outer center (x, y).
    # It's cleaner to return inner relative to outer center (0,0) and outer center (0,0).
    
    inner_centered = []
    for x, y in best_config:
        inner_centered.append([x - center[0], y - center[1], 0.0])
        
    return inner_centered, (0.0, 0.0), best_L, best_orientation * 30.0

# EVOLVE-BLOCK-END
