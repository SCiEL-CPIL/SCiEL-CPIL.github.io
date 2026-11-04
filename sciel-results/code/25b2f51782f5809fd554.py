import numpy as np
import math

N = 12

def get_hex_vertices(cx, cy, angle_deg, side=1.0):
    angle_rad = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3
        # Pointy top default vertices are at 30, 90, ...
        # Standard math angle 0 is right.
        # Let's use standard polar.
        x = cx + side * math.cos(theta)
        y = cy + side * math.sin(theta)
        verts.append((x, y))
    return verts

def min_hexagon_side(vertices, center=(0,0,0)):
    # vertices: list of (x,y)
    # center: (cx, cy)
    # We assume center is (0,0) for calculation, so shift vertices.
    cx, cy = center
    verts = np.array([(x - cx, y - cy) for x, y in vertices])
    
    best_L = float('inf')
    # Try rotations
    # The optimal rotation is likely aligned with the principal axes of the point set
    # or simply scanning 0 to 60 degrees.
    for theta_deg in np.linspace(0, 60, 121):
        theta = math.radians(theta_deg)
        # Normals to the 6 sides of the hexagon
        # Side k has normal angle theta + k * 60
        normals = []
        for k in range(6):
            alpha = theta + k * math.pi / 3
            normals.append((math.cos(alpha), math.sin(alpha)))
        
        # Calculate max projection
        max_proj = 0
        for v in verts:
            for n in normals:
                proj = v[0]*n[0] + v[1]*n[1]
                if proj > max_proj:
                    max_proj = proj
        
        # max_proj is the required apothem A
        L = 2 * max_proj / math.sqrt(3)
        if L < best_L:
            best_L = L
            best_theta = theta_deg
    return best_L, best_theta

def construct():
    # Candidate 1: 4x3 Grid (Pointy Top)
    # Rows i=0..3, Cols j=0..2
    centers1 = []
    for i in range(4):
        for j in range(3):
            x = j * math.sqrt(3) + (i % 2) * (math.sqrt(3) / 2)
            y = i * 1.5
            centers1.append((x, y))
    
    # Center the grid
    xs = [c[0] for c in centers1]
    ys = [c[1] for c in centers1]
    cx1 = (min(xs) + max(xs)) / 2
    cy1 = (min(ys) + max(ys)) / 2
    centers1 = [(x - cx1, y - cy1) for x, y in centers1]
    
    # Generate vertices
    verts1 = []
    for cx, cy in centers1:
        verts1.extend(get_hex_vertices(cx, cy, 0)) # 0 deg rotation for inner hexagons
        
    L1, theta1 = min_hexagon_side(verts1, (0,0))
    
    # Candidate 2: 1+6+5 Cluster
    centers2 = []
    centers2.append((0,0))
    for k in range(6):
        angle = k * math.pi / 3
        centers2.append((math.sqrt(3)*math.cos(angle), math.sqrt(3)*math.sin(angle)))
    # Add 5 more. Place them at distance 2*sqrt(3) ?
    # Or just take 5 from the next ring.
    # To keep it simple, let's try a 3x4 grid rotated 90 deg?
    # 3x4 is same as 4x3.
    
    # Let's try a "loose" 3x4 grid? No, 4x3 is better.
    
    # Let's try optimizing the inner hexagon rotation for the 4x3 grid.
    # If we rotate inner hexagons by 30 deg (flat top), does it help?
    verts1_flat = []
    for cx, cy in centers1:
        verts1_flat.extend(get_hex_vertices(cx, cy, 30))
    L1_flat, theta1_flat = min_hexagon_side(verts1_flat, (0,0))
    
    # Choose best
    if L1 < L1_flat:
        best_L = L1
        best_theta = theta1
        best_centers = centers1
        best_inner_angle = 0
    else:
        best_L = L1_flat
        best_theta = theta1_flat
        best_centers = centers1
        best_inner_angle = 30
        
    # Construct return values
    inner = []
    for x, y in best_centers:
        inner.append([x, y, float(best_inner_angle)])
        
    return inner, (0.0, 0.0), float(best_L), float(best_theta)
