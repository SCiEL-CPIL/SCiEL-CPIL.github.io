import math
import time

N = 11

def construct():
    start_time = time.time()
    
    sqrt3 = math.sqrt(3)
    
    # Hexagonal lattice positions for 11 unit hexagons (axial coords)
    positions_axial = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1),
        (2, 0), (-2, 0), (0, 2), (2, -2),
    ]
    
    # Convert to Cartesian for pointy-top hexagons (vertices at 0, 60, 120, ...)
    inner = []
    for q, r in positions_axial:
        x = sqrt3 * (q + r / 2.0)
        y = 1.5 * r
        inner.append([x, y, 0.0])
    
    # Compute minimum enclosing regular hexagon with angle=0
    # Vertices at 0, 60, 120, 180, 240, 300 degrees
    # Outward normals at 30, 90, 150, 210, 270, 330 degrees
    normals = []
    for i in range(6):
        rad = math.radians(30 + i * 60)
        normals.append((math.cos(rad), math.sin(rad)))
    
    def get_hex_vertices(cx, cy, side=1.0, angle_deg=0.0):
        rad = math.radians(angle_deg)
        verts = []
        for i in range(6):
            theta = rad + i * math.pi / 3
            verts.append((cx + side * math.cos(theta), cy + side * math.sin(theta)))
        return verts
    
    max_projections = [0.0] * 6
    
    for x, y, ang in inner:
        verts = get_hex_vertices(x, y, 1.0, ang)
        for i in range(6):
            nx, ny = normals[i]
            for vx, vy in verts:
                proj = nx * vx + ny * vy
                if proj > max_projections[i]:
                    max_projections[i] = proj
    
    apothem = max(max_projections)
    side = apothem * 2.0 / sqrt3
    
    return inner, (0.0, 0.0), side, 0.0
