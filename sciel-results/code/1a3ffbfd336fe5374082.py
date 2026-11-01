import math
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    """
    Pack 12 unit regular hexagons inside a regular hexagon.
    Uses a compact hexagonal cluster of 12 centers (a 'flower' of 6 around 1, plus 5 more)
    and optimizes the outer hexagon's rotation and side length.
    """
    # Define the 12 center positions for a compact hexagonal packing.
    # This is a subset of the hexagonal lattice.
    # Let's use a 4x4 block of a hex grid, or a specific compact shape.
    # A 12-hex cluster: 1 center, 6 around it, and 5 more from the next ring?
    # Or simply a 3x4 rectangular arrangement in hex grid coordinates?
    # Let's try the "compact" 12:
    # Row 0: 4 hexes
    # Row 1: 4 hexes
    # Row 2: 4 hexes
    # This forms a parallelogram.
    # Or:
    #   XX
    #  XXXX
    #  XXXX
    #   XX
    # That's 12.
    
    # Let's generate centers in a hex grid layout.
    # Axial coordinates (q, r).
    # Center is (0,0).
    # Neighbors: (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)
    
    # A 12-cell compact shape:
    # (0,0)
    # (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)  [7 total]
    # Next layer:
    # (2,0), (-2,0), (0,2), (0,-2), (2,-2), (-2,2)
    # (1,1), (-1,-1)
    # That's 13. We need 12.
    # Remove one corner, e.g., (2,0).
    
    centers_axial = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1),
        (2, -1), (1, -2), (-1, 2), (-2, 1), (0, 2)
    ]
    
    # Convert axial to pixel (x, y)
    # x = sqrt(3) * (q + r/2)
    # y = 3/2 * r
    centers = []
    for q, r in centers_axial:
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        centers.append((x, y))
    
    # Center of mass to center the cluster
    cx = sum(c[0] for c in centers) / len(centers)
    cy = sum(c[1] for c in centers) / len(centers)
    centers = [(x - cx, y - cy) for x, y in centers]
    
    # Each inner hexagon has side length 1 and angle 0.
    # Vertices of a unit hexagon with center (cx, cy) and angle 0:
    # (cx + cos(k*pi/3), cy + sin(k*pi/3)) for k in 0..5
    
    all_vertices = []
    for cx, cy in centers:
        for k in range(6):
            angle = k * math.pi / 3
            vx = cx + math.cos(angle)
            vy = cy + math.sin(angle)
            all_vertices.append([vx, vy])
    
    all_vertices = np.array(all_vertices)
    
    # Function to calculate required side length S for a given rotation theta
    # Outer hexagon normals are at angles theta + k * pi/3
    # Distance from center to edge is S * sqrt(3) / 2
    # So S >= (2 / sqrt(3)) * max_k | dot(v, n_k) |
    
    def required_S(theta):
        normals = []
        for k in range(6):
            a = theta + k * math.pi / 3
            normals.append([math.cos(a), math.sin(a)])
        normals = np.array(normals)
        
        # Project all vertices onto normals
        # Projections shape: (num_vertices, 6)
        projections = all_vertices @ normals.T
        max_abs_proj = np.max(np.abs(projections))
        
        return (2.0 / math.sqrt(3.0)) * max_abs_proj
    
    # Optimize theta
    res = minimize(required_S, x0=[0.0], bounds=[(0.0, math.pi/3)])
    
    best_theta = res.x[0]
    best_S = required_S(best_theta)
    
    # Construct the return values
    # inner: list of [x, y, angle_degrees]
    inner = [[c[0], c[1], 0.0] for c in centers]
    
    # Outer center
    center = [0.0, 0.0]
    
    # Outer side length
    side = best_S
    
    # Outer angle
    angle = best_theta * 180.0 / math.pi
    
    return inner, center, side, angle

# EVOLVE-BLOCK-END
