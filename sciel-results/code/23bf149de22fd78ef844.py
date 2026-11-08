import numpy as np

N = 12

def construct():
    # Generate hexagonal lattice centers
    # Axial coordinates to Cartesian
    # q, r
    # x = sqrt(3) * (q + r/2)
    # y = 3/2 * r
    
    centers = []
    # Center
    centers.append((0.0, 0.0))
    
    # Ring 1
    # (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
    ring1_coords = [(1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)]
    for q, r in ring1_coords:
        x = np.sqrt(3) * (q + r/2.0)
        y = 1.5 * r
        centers.append((x, y))
        
    # Ring 2
    # We need 5 more.
    # Full ring 2 has 12.
    # Let's pick 5 that are roughly balanced.
    # Corners of the hexagon shape:
    # (2,0), (0,2), (-2,2), (-2,0), (0,-2), (2,-2)
    # If we pick 5 of these 6, we leave one out.
    # This minimizes the max projection?
    # Let's pick all except (2,0).
    ring2_corners = [(2,0), (0,2), (-2,2), (-2,0), (0,-2), (2,-2)]
    # Pick 5
    selected_ring2 = ring2_corners[1:] # Skip (2,0)
    
    for q, r in selected_ring2:
        x = np.sqrt(3) * (q + r/2.0)
        y = 1.5 * r
        centers.append((x, y))
        
    # Now we have 12 centers.
    # All hexagons are aligned (angle 0).
    inner = []
    for cx, cy in centers:
        inner.append([cx, cy, 0.0])
        
    # Calculate minimal enclosing hexagon side length
    # Vertices of a unit hexagon (aligned, flat-top? or pointy-top?)
    # Standard: vertices at angles 0, 60, ...?
    # If angle 0 is a vertex, then the flat side is at 30 deg.
    # Let's assume the "angle_degrees" in the output rotates the hexagon.
    # If angle is 0, let's define the vertices.
    # Usually, 0 degrees means a vertex is on the x-axis.
    # Vertices: (1,0), (0.5, sqrt(3)/2), ...
    
    # Outer hexagon is also regular.
    # We need to find S such that all inner vertices are inside.
    # Outer hexagon center (0,0), angle 0.
    # Normals of outer hexagon (pointing to side midpoints):
    # If vertices are at 0, 60..., normals are at 30, 90, ...
    # Apothem a = S * sqrt(3) / 2.
    # Condition: v . n <= a
    
    # Inner hexagon vertices (relative to center):
    # v_k = (cos(60*k), sin(60*k)) for k=0..5
    
    max_proj = 0.0
    for cx, cy, ang in inner:
        # Rotate vertices by ang
        rad = np.deg2rad(ang)
        for k in range(6):
            vx = np.cos(np.deg2rad(60*k) + rad)
            vy = np.sin(np.deg2rad(60*k) + rad)
            # Absolute position
            px = cx + vx
            py = cy + vy
            # Project onto outer normals
            # Outer normals at 30, 90, 150, 210, 270, 330
            for j in range(6):
                nx = np.cos(np.deg2rad(30 + 60*j))
                ny = np.sin(np.deg2rad(30 + 60*j))
                proj = px * nx + py * ny
                if proj > max_proj:
                    max_proj = proj
                    
    # max_proj is the max distance from center to a vertex along a normal.
    # This must be <= Apothem = S * sqrt(3) / 2
    S = max_proj * 2.0 / np.sqrt(3)
    
    return inner, (0.0, 0.0), S, 0.0

# EVOLVE-BLOCK-END
