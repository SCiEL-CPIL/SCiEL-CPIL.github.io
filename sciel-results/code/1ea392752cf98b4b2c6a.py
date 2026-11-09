# EVOLVE-BLOCK-START
import numpy as np


def _hex_vertices(cx, cy, d, angle_deg):
    """Vertices of a regular hexagon with center (cx,cy), side d, rotation angle_deg."""
    a = np.deg2rad(angle_deg)
    angles = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])


def _minimal_enclosing_hex_side(vertices, angle_deg):
    """
    Given a set of 2D vertices, find the minimal side length s of a regular hexagon
    with the given rotation angle (degrees) and some center that contains all vertices.
    Returns (s, cx, cy).
    """
    a = np.deg2rad(angle_deg)
    # Face normals of a regular hexagon with this orientation:
    # A regular hexagon with vertices at angles a+60i has edge normals at a+60i+30.
    normals = []
    for i in range(6):
        ang = a + np.deg2rad(60.0 * i + 30.0)
        normals.append(np.array([np.cos(ang), np.sin(ang)]))
    normals = np.array(normals)
    # For each normal n, the hexagon has two parallel supporting lines at distance s/sqrt(3) from center? 
    # Actually for a regular hexagon of side s, the apothem (distance from center to edge midpoint) is s * sqrt(3)/2.
    # The hexagon = { x : n_i . (x - c) <= apothem for all i, and n_i . (x-c) >= -apothem }
    # where n_i are the 6 outward face normals.
    # So for all vertices v: |n_i . (v - c)| <= apothem = s * sqrt(3) / 2
    # => s >= 2/sqrt(3) * max_i |n_i . (v - c)|
    # We want to choose c to minimize max_i max_v |n_i . (v-c)|.
    # For a fixed set of normals, the optimal c is the center of the minimal enclosing in the "dual" sense.
    # Simple approach: for each pair of opposite normals, the projection range must fit.
    # Let's just optimize c by noting that for each normal n, we need:
    #   max_v n.v - min_v n.v <= 2 * apothem = s * sqrt(3)
    # So s >= (max_v n.v - min_v n.v) / sqrt(3) for each normal direction n (using both +n and -n as same axis).
    # The 6 normals give 3 axes (opposite pairs). So:
    #   s >= max over 3 axes of (range of projection) / sqrt(3)
    # And the center c can be chosen as the midpoint of the projection range for each axis; 
    # but c must be consistent. For a regular hexagon, c is the average of the 6 vertex-midpoint constraints.
    # Easier: just pick c = centroid of vertices, then compute required s, then check.
    # But better: for each of the 3 axes (normal pairs), the hexagon's center projection is the midpoint.
    # Let's use: for axis k with unit normal u_k (one of 3 directions), 
    #   proj_min_k = min_v u_k . v, proj_max_k = max_v u_k . v
    #   required half-width = (proj_max_k - proj_min_k)/2
    #   center projection c_k = (proj_max_k + proj_min_k)/2
    # Then s_k = (proj_max_k - proj_min_k) / sqrt(3)
    # s = max_k s_k
    # And c is determined by the 3 projections c_k along the 3 axes (which are 120 deg apart, spanning 2D).
    # Solve for c from 2 of the 3 axes.
    
    # 3 axis directions (0, 120, 240 deg offset by 30 from vertex angle)
    axis_angles = a + np.deg2rad(30.0) + np.deg2rad(120.0 * np.arange(3))
    axes = np.column_stack([np.cos(axis_angles), np.sin(axis_angles)])  # (3,2)
    
    projections = vertices @ axes.T  # (N, 3)
    proj_min = projections.min(axis=0)
    proj_max = projections.max(axis=0)
    center_projs = (proj_min + proj_max) / 2.0
    ranges = proj_max - proj_min
    s_required = ranges / np.sqrt(3.0)
    s = s_required.max()
    
    # Solve for c: axes[0:2] @ c = center_projs[0:2]
    A = axes[0:2]  # (2,2)
    b = center_projs[0:2]
    c = np.linalg.solve(A.T, b)  # A is 2x2, c = A^{-1} b => np.linalg.solve(A.T, b) since A@c=b => c = A^{-1}b
    
    return s, c[0], c[1]


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon.
    Uses a hexagonal lattice cluster arrangement and computes the minimal enclosing regular hexagon.
    Returns (inner_hex_data, outer_hex_data, outer_hex_side_length).
    """
    # Hexagonal lattice: unit hexagons (side 1) in flat-top orientation (angle 0).
    # Adjacent hex centers are at distance 2 * apothem = 2 * (sqrt(3)/2) = sqrt(3) apart? 
    # No: for flat-top hexagons of side 1, the distance between centers of adjacent hexes is 2 * (sqrt(3)/2) = sqrt(3)? 
    # Actually: apothem = sqrt(3)/2 * side = sqrt(3)/2. Distance between parallel edges = 2*apothem = sqrt(3).
    # But centers of adjacent hexes (sharing an edge) are at distance = 2 * apothem = sqrt(3)? 
    # Let me verify: two unit hexagons sharing an edge. The shared edge has length 1. The distance from center to edge midpoint is apothem = sqrt(3)/2.
    # So center-to-center distance = 2 * sqrt(3)/2 = sqrt(3) ≈ 1.732.
    # Wait, that seems small. Let me recompute: for a regular hexagon of side s, apothem = s * sqrt(3)/2.
    # For s=1, apothem = sqrt(3)/2 ≈ 0.866. Two hexes sharing an edge: centers are 2*apothem = sqrt(3) ≈ 1.732 apart.
    # But the edge length is 1, and the hexagons are unit, so this is correct.
    
    # Actually I need to be more careful. For flat-top hexagons:
    # Vertex at angle 0: (s, 0) = (1, 0)
    # The rightmost vertex is at (1,0), leftmost at (-1,0). Width = 2.
    # Top vertices at angles 60, 120: (cos60, sin60)=(0.5, sqrt(3)/2), (-0.5, sqrt(3)/2). Height = sqrt(3).
    # Adjacent hex to the right shares the right edge. The right edge is from vertex at 0 deg (1,0) to vertex at -60 deg (1,0)? 
    # No, flat-top: vertices at 0, 60, 120, 180, 240, 300 deg.
    # Vertex 0: (1, 0), vertex 60: (0.5, sqrt(3)/2), vertex 120: (-0.5, sqrt(3)/2), etc.
    # The rightmost edge is between vertex 0 (1,0) and vertex 300 (1,0)? No, vertex 300 = (cos300, sin300) = (0.5, -sqrt(3)/2).
    # Hmm, flat-top hex has vertices at 0, 60, 120, 180, 240, 300. The rightmost point is vertex 0 at (1,0).
    # The edges are: (0,60), (60,120), (120,180), (180,240), (240,300), (300,0).
    # Edge (300, 0): from (0.5, -sqrt(3)/2) to (1, 0). This is the upper-right edge.
    # Edge (0, 60): from (1,0) to (0.5, sqrt(3)/2). Upper-right.
    # So the rightmost "column" of vertices is at x=1 (only vertex 0).
    # For a hexagonal lattice of flat-top hexagons, the spacing in x between columns is 1.5 (3/2 * side), and y spacing is sqrt(3).
    # Center-to-center distance for adjacent hexes: sqrt((1.5)^2 + (sqrt(3)/2)^2) = sqrt(2.25 + 0.75) = sqrt(3) ≈ 1.732. Correct.
    
    # So lattice vectors: 
    #   e1 = (1.5, sqrt(3)/2)  -- to upper-right neighbor
    #   e2 = (1.5, -sqrt(3)/2) -- to lower-right neighbor
    #   e3 = (0, sqrt(3))      -- to top neighbor (e1 + e2 rotated? no, e1 - e2 = (0, sqrt(3)))
    
    e1 = np.array([1.5, np.sqrt(3)/2])
    e2 = np.array([1.5, -np.sqrt(3)/2])
    e3 = e1 - e2  # (0, sqrt(3))
    
    # 12 hexagons in a compact cluster. Use a "hexagonal" shape: 
    # Ring 0: 1 hex at (0,0)
    # Ring 1: 6 hexes around it
    # Ring 2: 6 more hexes (alternating positions of the 12 in ring 2)
    # Total: 1 + 6 + 5 = 12? Or 1 + 6 + 6 = 13, too many.
    # Let's use: center + 6 ring + 5 from second ring = 12.
    # Or a 3-4-3-2 arrangement? Let me use a known compact 12-hex cluster.
    
    # Simpler: use a 3x4 rectangular-ish arrangement in hex lattice, or the "hexagon of side 2" (7 hexes) + 5 more.
    # Let me just place 12 hexes in a hexagonal lattice forming a compact blob.
    
    # Positions in lattice coordinates (i, j) where position = i*e1 + j*e2
    # A compact 12-hex shape:
    lattice_positions = [
        (0, 0),    # center
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1),  # ring 1 (6)
        (2, 0), (2, -1), (1, 1), (-2, 0), (-2, 1), (-1, -1),  # ring 2 partial (6)
    ]
    # That's 1 + 6 + 6 = 13. Remove one.
    lattice_positions = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1),
        (2, 0), (2, -1), (1, 1), (-2, 0), (-2, 1),
    ]
    # 1 + 6 + 5 = 12
    
    centers = []
    for (i, j) in lattice_positions:
        c = i * e1 + j * e2
        centers.append(c)
    centers = np.array(centers)
    
    # All inner hexagons have angle 0 (flat-top), side 1
    inner_hex_data = np.zeros((12, 3))
    for k, c in enumerate(centers):
        inner_hex_data[k, 0] = c[0]
        inner_hex_data[k, 1] = c[1]
        inner_hex_data[k, 2] = 0.0
    
    # Collect all vertices of all inner hexagons
    all_vertices = []
    for k in range(12):
        cx, cy, ang = inner_hex_data[k]
        verts = _hex_vertices(cx, cy, 1.0, ang)
        all_vertices.append(verts)
    all_vertices = np.vstack(all_vertices)  # (72, 2)
    
    # Find minimal enclosing regular hexagon. Try several angles.
    best_s = np.inf
    best_c = np.array([0.0, 0.0])
    best_angle = 0.0
    
    # The outer hexagon should probably have the same orientation as the inner lattice (angle 0)
    # or be rotated to minimize s. Let's try angles from 0 to 60 in small steps.
    for angle_deg in np.linspace(0, 60, 121):
        s, cx, cy = _minimal_enclosing_hex_side(all_vertices, angle_deg)
        if s < best_s:
            best_s = s
            best_c = np.array([cx, cy])
            best_angle = angle_deg
    
    outer_hex_data = np.array([best_c[0], best_c[1], best_angle])
    outer_hex_side_length = best_s
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
