import numpy as np

N = 11

def construct():
    # Define centers of 11 unit hexagons
    # We use a compact "cluster" arrangement.
    # A size-2 hexagon (7 hexes) is a good base.
    # Add 4 more to the perimeter to keep it compact.
    
    # Centers for a size-2 hexagon (side 2)
    # Row 0: 2 hexes
    # Row 1: 3 hexes
    # Row 2: 2 hexes
    # This is 7 hexes.
    # Let's add 4 more to make it 11.
    
    # Coordinates for hex grid (flat-top)
    # x = 1.5 * col
    # y = sqrt(3) * (row + 0.5 * (col % 2))
    
    centers = []
    # Base 7 hexes (size 2 hexagon shape)
    # Col 0: Row 0, 1
    # Col 1: Row 0, 1, 2
    # Col 2: Row 0, 1
    # Wait, standard hex indexing is tricky. Let's use axial coordinates or just manual placement.
    
    # Let's place them manually in a compact shape.
    # 1 center
    # 6 around it
    # 4 more around the 6?
    
    # Let's try a 3x4 grid approximation.
    # Row 0: 3 hexes
    # Row 1: 4 hexes
    # Row 2: 3 hexes
    # Row 3: 1 hex
    # Total 11.
    
    sqrt3 = np.sqrt(3)
    
    # Generate positions
    # We will center the whole cluster at (0,0)
    
    raw_centers = []
    # Row 0 (y = 0)
    raw_centers.append([0.0, 0.0])
    raw_centers.append([1.5, 0.0])
    raw_centers.append([3.0, 0.0])
    
    # Row 1 (y = sqrt(3)/2)
    raw_centers.append([0.75, sqrt3/2])
    raw_centers.append([2.25, sqrt3/2])
    raw_centers.append([3.75, sqrt3/2])
    raw_centers.append([5.25, sqrt3/2])
    
    # Row 2 (y = sqrt(3))
    raw_centers.append([1.5, sqrt3])
    raw_centers.append([3.0, sqrt3])
    raw_centers.append([4.5, sqrt3])
    
    # Row 3 (y = 3*sqrt(3)/2)
    raw_centers.append([3.0, 3*sqrt3/2])
    
    # Total 11 centers.
    
    # Calculate centroid to center the cluster
    cx = np.mean([c[0] for c in raw_centers])
    cy = np.mean([c[1] for c in raw_centers])
    
    centers = [[c[0] - cx, c[1] - cy] for c in raw_centers]
    
    # Now, find the minimum side length S of a regular hexagon centered at (0,0) with angle 0
    # that contains all vertices of these unit hexagons.
    # A unit hexagon at (x,y) with angle 0 has vertices at distance 1 from (x,y).
    # The vertices are at (x + cos(theta), y + sin(theta)) for theta in 0, 60, ..., 300.
    
    # The outer hexagon has normals at 0, 60, ..., 300 degrees.
    # Condition: For all inner vertices V and all outer normals N, |V dot N| <= apothem_outer
    # apothem_outer = S * sqrt(3) / 2
    # So S >= max(|V dot N|) / (sqrt(3)/2)
    
    max_proj = 0.0
    outer_normals = []
    for i in range(6):
        angle = np.radians(60 * i)
        outer_normals.append((np.cos(angle), np.sin(angle)))
    
    inner_vertex_angles = [np.radians(60 * i) for i in range(6)]
    
    for cx_h, cy_h in centers:
        for va in inner_vertex_angles:
            vx = cx_h + np.cos(va)
            vy = cy_h + np.sin(va)
            for nx, ny in outer_normals:
                proj = abs(vx * nx + vy * ny)
                if proj > max_proj:
                    max_proj = proj
                    
    S = max_proj / (np.sqrt(3) / 2)
    
    # Add a small epsilon to ensure strict containment due to floating point
    S += 0.01
    
    # Construct the return values
    inner = [[c[0], c[1], 0.0] for c in centers]
    center = (0.0, 0.0)
    
    return inner, center, S, 0.0

# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    
    N = 11
    sqrt3 = np.sqrt(3)
    
    # Define a compact packing of 11 hexagons
    # This specific packing is a "blob" shape.
    raw_centers = [
        [0.0, 0.0],
        [1.5, 0.0],
        [3.0, 0.0],
        [0.75, sqrt3/2],
        [2.25, sqrt3/2],
        [3.75, sqrt3/2],
        [5.25, sqrt3/2],
        [1.5, sqrt3],
        [3.0, sqrt3],
        [4.5, sqrt3],
        [3.0, 3*sqrt3/2]
    ]
    
    # Center the cluster
    cx = np.mean([c[0] for c in raw_centers])
    cy = np.mean([c[1] for c in raw_centers])
    centers = [[c[0] - cx, c[1] - cy] for c in raw_centers]
    
    # Calculate minimal enclosing hexagon side length
    max_proj = 0.0
    outer_normals = [(np.cos(np.radians(60 * i)), np.sin(np.radians(60 * i))) for i in range(6)]
    inner_vertex_angles = [np.radians(60 * i) for i in range(6)]
    
    for cx_h, cy_h in centers:
        for va in inner_vertex_angles:
            vx = cx_h + np.cos(va)
            vy = cy_h + np.sin(va)
            for nx, ny in outer_normals:
                proj = abs(vx * nx + vy * ny)
                if proj > max_proj:
                    max_proj = proj
                    
    S = max_proj / (np.sqrt(3) / 2) + 0.01
    
    inner = [[c[0], c[1], 0.0] for c in centers]
    return inner, (0.0, 0.0), S, 0.0
# EVOLVE-BLOCK-END
