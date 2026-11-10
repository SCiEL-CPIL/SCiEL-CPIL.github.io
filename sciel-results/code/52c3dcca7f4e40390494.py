import numpy as np

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses a hexagonal grid layout with the outer hexagon rotated by 30 degrees for tighter fit.
    """
    n = 12
    
    # Define centers for 12 hexagons in a compact hexagonal cluster
    # Center + Ring 1 (6) + Ring 2 (5 selected)
    # Axial coordinates (q, r) to Cartesian (x, y)
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    
    centers = []
    # Center
    centers.append((0, 0))
    
    # Ring 1
    ring1 = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    for q, r in ring1:
        x = np.sqrt(3) * (q + r/2)
        y = 1.5 * r
        centers.append((x, y))
        
    # Ring 2: Select 5 hexagons to minimize max extent
    # The 6 hexagons at distance 3 are: (1,1), (-1,2), (-2,1), (-1,-1), (1,-2), (2,-1)
    # We pick 5 of them. Due to symmetry, any 5 will have the same max extent in the directions they cover.
    # We leave out (2,-1) for example.
    ring2_selected = [(1,1), (-1,2), (-2,1), (-1,-1), (1,-2)]
    for q, r in ring2_selected:
        x = np.sqrt(3) * (q + r/2)
        y = 1.5 * r
        centers.append((x, y))
        
    xs = np.array([c[0] for c in centers])
    ys = np.array([c[1] for c in centers])
    angles = np.zeros(n)
    
    inner_hex_data = np.column_stack((xs, ys, angles))
    
    # Outer hexagon centered at 0,0
    # Try angle 30 degrees
    outer_angle = 30.0
    
    # Calculate required side length s
    # Vertices of inner hexagons
    def get_hex_vertices(cx, cy, angle_deg, side=1.0):
        a = np.deg2rad(angle_deg)
        angles_rad = a + np.deg2rad(np.arange(6) * 60)
        return np.array([cx + side * np.cos(angles_rad), cy + side * np.sin(angles_rad)]).T

    all_vertices = []
    for i in range(n):
        verts = get_hex_vertices(xs[i], ys[i], angles[i])
        all_vertices.append(verts)
        
    all_vertices = np.vstack(all_vertices)
    
    # Outer hexagon with angle 30 deg
    # Vertices at 30, 90, 150, 210, 270, 330
    # Normals to edges at 0, 60, 120, 180, 240, 300
    # Constraint: dot(v, normal) <= apothem = s * sqrt(3) / 2
    
    normals = []
    for i in range(6):
        angle_rad = np.deg2rad(i * 60)
        normals.append([np.cos(angle_rad), np.sin(angle_rad)])
    normals = np.array(normals)
    
    # Project all vertices onto normals
    projections = np.dot(all_vertices, normals.T)
    
    max_projections = np.max(projections, axis=0)
    
    # s * sqrt(3) / 2 >= max_projection
    # s >= 2 * max_projection / sqrt(3)
    required_s = 2 * max_projections / np.sqrt(3)
    
    s = np.max(required_s)
    
    # Add a small margin to ensure validity
    s += 1e-9
    
    outer_hex_data = np.array([0, 0, outer_angle])
    
    return inner_hex_data, outer_hex_data, s
