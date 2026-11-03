"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Hexagonal lattice positions for 12 unit hexagons
    # Using axial coordinates (q, r) to Cartesian
    # For a hexagon with side length 1, the distance between adjacent centers is sqrt(3)
    
    # Define positions in a compact hexagonal arrangement
    # We'll use a pattern that minimizes the bounding hexagon
    # Axial coordinates for 12 hexagons in a roughly hexagonal shape
    
    # Let's use these axial coordinates (q, r):
    # A 3-4-3-2 pattern or similar
    # Actually, let's use a well-known compact arrangement
    
    # Positions in axial coords (col, row) for a hex grid
    # Using "pointy-top" or "flat-top" - let's use flat-top
    # For flat-top hexagons, adjacent centers:
    #   East: (sqrt(3), 0)
    #   NE: (sqrt(3)/2, 3/2)
    #   NW: (-sqrt(3)/2, 3/2)
    #   West: (-sqrt(3), 0)
    #   SW: (-sqrt(3)/2, -3/2)
    #   SE: (sqrt(3)/2, -3/2)
    
    # Let me use a simple 4x3 grid with hex offset
    # Or better, a proper hexagonal cluster
    
    # 12 hexagons: let's do rows of 2, 3, 3, 3, 1 = 12
    # Or 3, 4, 3, 2 = 12
    # Let's try 4, 4, 4 = 12 in three rows
    
    sqrt3 = np.sqrt(3)
    
    # Three rows of 4 hexagons each, with hexagonal offset
    # Row spacing: 1.5 (vertical distance between rows in hex grid for flat-top)
    # Actually for flat-top hexagons with side 1:
    #   Horizontal spacing between columns: sqrt(3)
    #   Vertical spacing between rows: 1.5
    #   Odd rows shifted by sqrt(3)/2
    
    # Let's place them:
    positions = []
    # Row 0: 4 hexagons
    for i in range(4):
        x = i * sqrt3
        y = 0
        positions.append([x, y, 0.0])
    # Row 1: 4 hexagons, shifted right by sqrt3/2
    for i in range(4):
        x = (i + 0.5) * sqrt3
        y = 1.5
        positions.append([x, y, 0.0])
    # Row 2: 4 hexagons
    for i in range(4):
        x = i * sqrt3
        y = 3.0
        positions.append([x, y, 0.0])
    
    # Now find the minimum enclosing regular hexagon
    # A regular hexagon with center (cx, cy), side length s, rotation theta
    # has vertices at distance s from center, at angles theta + k*60 for k=0..5
    # The hexagon contains a point if the point is within all 6 half-planes
    
    # For a regular hexagon with side length s, the inradius (apothem) is s * sqrt(3)/2
    # The hexagon can be described as: for each of 6 directions (normals), 
    # the projection of any point onto the normal must be <= apothem
    
    # The 6 outward normals are at angles theta + k*60 for k=0..5
    # For each normal direction, the maximum projection of all hexagon vertices must be <= apothem
    
    # Actually, we need to check that all vertices of all inner hexagons are inside the outer hexagon.
    # But that's complex. Instead, let's use the fact that for convex shapes, if all vertices 
    # of the inner shapes are inside, the shapes are inside. But we can simplify by checking 
    # the extreme points.
    
    # Simpler approach: for a given outer hexagon orientation, compute the required side length
    # by finding the maximum "support function" value in each of the 6 normal directions.
    
    def get_hexagon_vertices(cx, cy, s, theta_deg):
        theta = np.radians(theta_deg)
        vertices = []
        for k in range(6):
            angle = theta + k * np.pi / 3
            vertices.append([cx + s * np.cos(angle), cy + s * np.sin(angle)])
        return np.array(vertices)
    
    def get_inner_hex_vertices(x, y, s, theta_deg):
        theta = np.radians(theta_deg)
        vertices = []
        for k in range(6):
            angle = theta + k * np.pi / 3
            vertices.append([x + s * np.cos(angle), y + s * np.sin(angle)])
        return np.array(vertices)
    
    def objective(params):
        cx, cy, theta_deg = params
        theta = np.radians(theta_deg)
        
        # The 6 outward normal directions of the outer hexagon
        normals = []
        for k in range(6):
            angle = theta + k * np.pi / 3 + np.pi / 6  # Normals are at 30 + k*60 degrees
            normals.append([np.cos(angle), np.sin(angle)])
        normals = np.array(normals)
        
        # For each inner hexagon, compute its vertices
        # The maximum projection in each normal direction determines the required apothem
        max_projections = np.zeros(6)
        
        for pos in positions:
            ix, iy, iangle = pos[0], pos[1], pos[2]
            # Vertices of this unit hexagon
            for k in range(6):
                vangle = np.radians(iangle) + k * np.pi / 3
                vx = ix + np.cos(vangle)
                vy = iy + np.sin(vangle)
                # Project onto each normal
                for j in range(6):
                    proj = vx * normals[j][0] + vy * normals[j][1]
                    # We need (proj - (cx * normals[j][0] + cy * normals[j][1])) <= apothem
                    # So apothem >= proj - (cx * normals[j][0] + cy * normals[j][1])
                    val = proj - (cx * normals[j][0] + cy * normals[j][1])
                    if val > max_projections[j]:
                        max_projections[j] = val
        
        # The required apothem is the max over all 6 directions
        apothem = np.max(max_projections)
        if apothem < 0:
            apothem = 0
        
        # Side length s = apothem / (sqrt(3)/2) = 2 * apothem / sqrt(3)
        s = 2 * apothem / sqrt3
        return s
    
    # Initial guess: center of the bounding box, rotation 0
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx0 = (min(xs) + max(xs)) / 2
    cy0 = (min(ys) + max(ys)) / 2
    
    # Optimize over center and rotation
    result = minimize(objective, [cx0, cy0, 0.0], method='Nelder-Mead',
                      options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
    
    cx, cy, theta_deg = result.x
    s = objective(result.x)
    
    inner = [[p[0], p[1], p[2]] for p in positions]
    return inner, (float(cx), float(cy)), float(s), float(theta_deg)
# EVOLVE-BLOCK-END

