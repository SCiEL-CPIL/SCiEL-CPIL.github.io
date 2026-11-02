"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize_scalar

N = 12

# EVOLVE-BLOCK-START
def construct():
    sqrt3 = np.sqrt(3)
    
    # Define 12 unit hexagon centers in a compact honeycomb cluster
    # Using a 3x4 rectangular block in the hex lattice
    # Lattice vectors: a1 = (sqrt(3), 0), a2 = (sqrt(3)/2, 3/2)
    
    positions = []
    for i in range(3):
        for j in range(4):
            x = i * sqrt3 + j * sqrt3 / 2
            y = j * 1.5
            positions.append((x, y))
    
    # Center the cluster at origin
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    
    centered_positions = [(x - cx, y - cy) for x, y in positions]
    
    # The 6 normal directions for a flat-top regular hexagon
    normals = np.array([
        [1, 0],
        [0.5, sqrt3/2],
        [-0.5, sqrt3/2],
        [-1, 0],
        [-0.5, -sqrt3/2],
        [0.5, -sqrt3/2]
    ])
    
    def min_outer_side(theta_deg):
        """Find minimum outer hexagon side length for given cluster rotation."""
        theta = np.radians(theta_deg)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        
        max_val = 0
        for x, y in centered_positions:
            # Rotate position
            rx = x * cos_t - y * sin_t
            ry = x * sin_t + y * cos_t
            
            for nx, ny in normals:
                dot_val = rx * nx + ry * ny
                # Required outer side length: |dot| / apothem_outer + 1
                # apothem_outer = S * sqrt(3)/2, so S = |dot| / (sqrt(3)/2) + 1
                # = |dot| * 2/sqrt(3) + 1
                s_req = abs(dot_val) * 2 / sqrt3 + 1
                if s_req > max_val:
                    max_val = s_req
        
        return max_val
    
    # Optimize over rotation angle (0 to 60 degrees due to hex symmetry)
    result = minimize_scalar(min_outer_side, bounds=(0, 60), method='bounded', 
                             options={'xatol': 1e-8})
    best_theta = result.x
    best_side = result.fun
    
    # Get final positions with inner hexagons aligned to outer hexagon
    theta = np.radians(best_theta)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    
    final_positions = []
    for x, y in centered_positions:
        rx = x * cos_t - y * sin_t
        ry = x * sin_t + y * cos_t
        final_positions.append([rx, ry, best_theta])
    
    return final_positions, (0.0, 0.0), best_side, best_theta
# EVOLVE-BLOCK-END

