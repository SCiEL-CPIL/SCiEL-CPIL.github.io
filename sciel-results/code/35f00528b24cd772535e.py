"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    sqrt3 = np.sqrt(3)
    
    # Define 12 hexagon centers in a compact cluster (flat-top orientation)
    # Axial coordinates for compact arrangement
    positions_axial = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)
    ]
    
    # Convert to Cartesian (flat-top hexagons)
    centers = []
    for q, r in positions_axial:
        x = 1.5 * q
        y = sqrt3 * (r + q / 2.0)
        centers.append((x, y))
    centers = np.array(centers)
    
    # Generate all vertices of all 12 hexagons
    # For flat-top hexagon of side 1, vertices at angles 0, 60, 120, 180, 240, 300 degrees
    # Vertex k: center + (cos(k*pi/3), sin(k*pi/3)) for k=0..5
    all_vertices = []
    for cx, cy in centers:
        for k in range(6):
            angle = k * np.pi / 3
            vx = cx + np.cos(angle)
            vy = cy + np.sin(angle)
            all_vertices.append((vx, vy))
    all_vertices = np.array(all_vertices)
    
    # Find minimum enclosing regular hexagon
    # Regular hexagon with center (cx, cy), side s, rotation theta
    # Apothem = s * sqrt(3)/2
    # A point is inside if its projection onto each of 6 outward normals <= apothem
    
    def required_side(cx, cy, theta):
        """Compute the side length of the smallest regular hexagon with center (cx,cy), rotation theta that contains all_vertices."""
        angles = theta + np.arange(6) * np.pi / 3
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        
        rel = all_vertices - np.array([cx, cy])
        projections = rel @ normals.T  # (72, 6)
        max_proj = np.max(projections, axis=0)  # (6,)
        max_apothem = np.max(max_proj)
        # side = 2 * apothem / sqrt(3)
        return max_apothem * 2.0 / sqrt3
    
    def objective(params):
        cx, cy, theta = params
        return required_side(cx, cy, theta)
    
    # Initial guess
    cx0, cy0 = np.mean(centers, axis=0)
    result = minimize(objective, [cx0, cy0, 0.0], method='Nelder-Mead',
                      options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
    
    cx, cy, theta = result.x
    side = max(result.fun, 0.0)
    
    # Build output
    inner = [[float(cx_i), float(cy_i), 0.0] for cx_i, cy_i in centers]
    outer_center = (float(cx), float(cy))
    outer_side = float(side)
    outer_angle = float(np.degrees(theta) % 360.0)
    
    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
