import numpy as np
from scipy.spatial import ConvexHull
import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Define the 11 centers in the hexagonal grid
    # Using axial coordinates (q, r)
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    
    centers_axial = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (1, 1), (-1, -1)
    ]
    
    sqrt3 = math.sqrt(3)
    centers_cart = []
    for q, r in centers_axial:
        x = sqrt3 * (q + r / 2.0)
        y = 1.5 * r
        centers_cart.append((x, y))
    
    # Generate vertices of each hexagon
    # A regular hexagon with side 1, centered at (cx, cy), rotated by 0 (flat-top? pointy-top?)
    # Standard hexagon vertices for side 1, pointy-top (vertex at 90 deg):
    # Angles: 90, 150, 210, 270, 330, 30
    # Or flat-top (vertex at 0 deg):
    # Angles: 0, 60, 120, 180, 240, 300
    # Let's use flat-top for simplicity, so vertices are at angles 0, 60, ...
    # Distance from center to vertex is 1.
    
    vertices = []
    for cx, cy in centers_cart:
        for i in range(6):
            angle = math.radians(60 * i)
            vx = cx + math.cos(angle)
            vy = cy + math.sin(angle)
            vertices.append((vx, vy))
            
    vertices = np.array(vertices)
    
    # Compute convex hull
    hull = ConvexHull(vertices)
    hull_vertices = vertices[hull.vertices]
    
    # Function to calculate minimal outer side length for a given rotation theta
    # We rotate the hull vertices by theta, then find the max projection onto the 6 fixed normals
    # Fixed normals for a flat-top outer hexagon (vertices at 0, 60, ...):
    # The sides are perpendicular to 30, 90, 150, 210, 270, 330 degrees.
    # Normals:
    normals = []
    for i in range(6):
        angle = math.radians(60 * i + 30)
        normals.append((math.cos(angle), math.sin(angle)))
    normals = np.array(normals)
    
    def calc_side(theta):
        # Rotate hull vertices by theta
        c = math.cos(theta)
        s = math.sin(theta)
        R = np.array([[c, -s], [s, c]])
        rotated = hull_vertices @ R.T
        # Project onto normals
        # proj = rotated @ normals.T
        # We need max absolute value
        # max_proj = np.max(np.abs(rotated @ normals.T))
        # Side length S = (2/sqrt(3)) * max_proj
        max_proj = np.max(np.abs(rotated @ normals.T))
        return (2.0 / sqrt3) * max_proj

    # Minimize calc_side(theta) over [0, pi/3]
    # Use ternary search or fine grid
    # Since the function is periodic and likely unimodal in [0, pi/6] due to symmetry?
    # Actually, the cluster has 2-fold symmetry (point symmetry), so the function might have period pi/2?
    # But the outer hexagon has period pi/3.
    # Let's just do a fine grid search for robustness, then refine.
    
    # Fine grid
    thetas = np.linspace(0, math.pi/3, 1000)
    sides = np.array([calc_side(t) for t in thetas])
    min_idx = np.argmin(sides)
    theta_opt = thetas[min_idx]
    
    # Refine around theta_opt
    # Ternary search in [theta_opt - step, theta_opt + step]
    step = thetas[1] - thetas[0]
    a = max(0, theta_opt - step)
    b = min(math.pi/3, theta_opt + step)
    
    for _ in range(100):
        m1 = a + (b - a) / 3.0
        m2 = b - (b - a) / 3.0
        f1 = calc_side(m1)
        f2 = calc_side(m2)
        if f1 < f2:
            b = m2
        else:
            a = m1
            
    theta_opt = (a + b) / 2.0
    min_side = calc_side(theta_opt)
    
    # Construct the inner list
    # Rotate the centers by theta_opt
    c = math.cos(theta_opt)
    s = math.sin(theta_opt)
    inner = []
    for cx, cy in centers_cart:
        nx = cx * c - cy * s
        ny = cx * s + cy * c
        # Angle of each inner hexagon. 
        # The problem says "each inner hexagon may translate and rotate".
        # We can keep them all at 0 rotation relative to the global frame, 
        # or rotate them with the cluster. 
        # Since they are rigidly connected in our cluster, we rotate their positions.
        # Their individual orientation can be anything, but to maintain the "rigid cluster" assumption 
        # for the hull calculation, we assumed they were all aligned with the cluster frame.
        # So we should rotate their orientation too? 
        # Actually, the hull calculation assumed the hexagons were aligned with the axes used for vertices.
        # If we rotate the positions by theta_opt, we should also rotate the hexagons by theta_opt 
        # to maintain the same relative shape.
        # So angle_degrees = theta_opt * 180 / pi
        angle_deg = math.degrees(theta_opt)
        inner.append([nx, ny, angle_deg])
        
    return inner, (0.0, 0.0), min_side, 0.0
# EVOLVE-BLOCK-END
