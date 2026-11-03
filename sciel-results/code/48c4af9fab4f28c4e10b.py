import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Unit hexagon apothem
    a = math.sqrt(3) / 2
    
    # Lattice vectors for flat-top hexagons
    # v1 = (0, sqrt(3))
    # v2 = (1.5, sqrt(3)/2)
    
    # Generate 11 points
    points = []
    
    # Center
    points.append((0.0, 0.0))
    
    # 6 Neighbors
    # (0, sqrt(3))
    points.append((0.0, math.sqrt(3)))
    # (1.5, sqrt(3)/2)
    points.append((1.5, math.sqrt(3)/2))
    # (-1.5, sqrt(3)/2)
    points.append((-1.5, math.sqrt(3)/2))
    # (0, -sqrt(3))
    points.append((0.0, -math.sqrt(3)))
    # (-1.5, -sqrt(3)/2)
    points.append((-1.5, -math.sqrt(3)/2))
    # (1.5, -sqrt(3)/2)
    points.append((1.5, -math.sqrt(3)/2))
    
    # 4 from Ring 2
    # (1.5, 1.5*sqrt(3))
    points.append((1.5, 1.5 * math.sqrt(3)))
    # (-1.5, 1.5*sqrt(3))
    points.append((-1.5, 1.5 * math.sqrt(3)))
    # (-3, 0)
    points.append((-3.0, 0.0))
    # (-1.5, -1.5*sqrt(3))
    points.append((-1.5, -1.5 * math.sqrt(3)))
    
    # Calculate minimal side length S
    # Normals at 0, 60, 120, 180, 240, 300 degrees
    normals = []
    for k in range(6):
        theta = math.radians(60 * k)
        normals.append((math.cos(theta), math.sin(theta)))
    
    max_proj = 0.0
    for nx, ny in normals:
        max_in_dir = 0.0
        for px, py in points:
            proj = px * nx + py * ny
            if proj > max_in_dir:
                max_in_dir = proj
        if max_in_dir > max_proj:
            max_proj = max_in_dir
            
    # S * sqrt(3)/2 = max_proj + a
    # S = 2/sqrt(3) * (max_proj + a)
    S = 2.0 / math.sqrt(3) * (max_proj + a)
    
    # Format inner hexagons: [x, y, angle]
    inner = []
    for px, py in points:
        inner.append([px, py, 0.0])
        
    return inner, (0.0, 0.0), S, 0.0
# EVOLVE-BLOCK-END
