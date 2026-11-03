"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 11

# EVOLVE-BLOCK-START
def _hexagon_vertices(cx, cy, side, angle_deg):
    """Get vertices of a regular hexagon with given center, side length, and rotation."""
    angle_rad = np.radians(angle_deg)
    # For pointy-top hexagon, vertices at 30, 90, 150, 210, 270, 330 degrees
    angles = np.array([30, 90, 150, 210, 270, 330]) * np.pi / 180 + angle_rad
    vertices = np.column_stack([
        cx + side * np.cos(angles),
        cy + side * np.sin(angles)
    ])
    return vertices

def _min_enclosing_hexagon_side(points, center, max_container_rotation=360, steps=360):
    """
    Find minimum side length of a regular hexagon (with given center) that contains all points.
    We try rotating the container hexagon and for each rotation, compute the required side length.
    
    For a hexagon with side s, center c, and rotation alpha, the 6 normals are at angles alpha + k*60.
    The apothem (distance from center to each side) is s * sqrt(3) / 2.
    A point p is inside if for all k: dot(p - c, n_k) <= s * sqrt(3) / 2
    So s >= max_k dot(p - c, n_k) / (sqrt(3)/2) for all points p.
    """
    best_s = float('inf')
    n_points = len(points)
    for step in range(steps):
        alpha = 2 * np.pi * step / steps
        normals = np.array([
            [np.cos(alpha + k * np.pi / 3), np.sin(alpha + k * np.pi / 3)]
            for k in range(6)
        ])  # shape (6, 2)
        
        # For each point, compute max projection onto any normal
        # points shape (n_points, 2), normals shape (6, 2)
        # projections shape (n_points, 6)
        rel = points - center  # (n_points, 2)
        projections = rel @ normals.T  # (n_points, 6)
        max_proj = np.max(projections, axis=1)  # (n_points,)
        overall_max = np.max(max_proj)
        
        s = overall_max / (np.sqrt(3) / 2)
        if s < best_s:
            best_s = s
    
    return best_s

def _try_configuration(positions, angles_deg, outer_center, outer_rotation_steps=720):
    """
    Try a configuration of inner hexagons and find the minimum enclosing outer hexagon.
    positions: list of (x, y)
    angles_deg: list of rotation angles for each inner hexagon
    outer_center: (cx, cy)
    Returns (inner_list, outer_center, outer_side, outer_angle_deg)
    """
    # Get all vertices of all inner hexagons
    all_points = []
    for (px, py), ang in zip(positions, angles_deg):
        verts = _hexagon_vertices(px, py, 1.0, ang)
        all_points.append(verts)
    all_points = np.vstack(all_points)  # (6*N, 2)
    
    # Find minimum enclosing hexagon with given center
    best_s = float('inf')
    best_rot = 0.0
    n_points = len(all_points)
    # Try many container rotations
    for step in range(outer_rotation_steps):
        alpha = 2 * np.pi * step / outer_rotation_steps
        normals = np.array([
            [np.cos(alpha + k * np.pi / 3), np.sin(alpha + k * np.pi / 3)]
            for k in range(6)
        ])
        rel = all_points - np.array(outer_center)
        projections = rel @ normals.T
        max_proj = np.max(projections, axis=1)
        overall_max = np.max(max_proj)
        s = overall_max / (np.sqrt(3) / 2)
        if s < best_s:
            best_s = s
            best_rot = np.degrees(alpha)
    
    inner_list = [[px, py, ang] for (px, py), ang in zip(positions, angles_deg)]
    return inner_list, outer_center, best_s, best_rot

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    best_result = None
    best_score = float('inf')
    
    # Candidate configurations: (positions, angles_deg, outer_center)
    # Each position is (x, y), angle is rotation of that hexagon
    
    candidates = []
    
    # Configuration 1: Honeycomb-like arrangement
    # Rows of hexagons with offset
    # Hexagon spacing: horizontal distance between adjacent centers in same row = 3 (for flat-to-flat alignment)
    # Actually for unit hexagons, center-to-center distance for non-overlapping is 2*sqrt(3)/2 * 2 = ... 
    # Let me think: two unit hexagons side by side (sharing an edge) have centers at distance sqrt(3) apart (flat-to-flat of one hexagon = sqrt(3), so center distance = sqrt(3))
    # Wait, no. For two hexagons sharing a full edge, the center distance is sqrt(3) (the apothem * 2 = sqrt(3)).
    # Actually apothem of unit hexagon = sqrt(3)/2, so two hexagons sharing an edge have centers at distance 2 * sqrt(3)/2 = sqrt(3).
    
    # Let me use a grid where hexagons are placed with center distance sqrt(3) in horizontal direction
    # and offset rows
    
    hex_spacing = np.sqrt(3)  # center-to-center for adjacent hexagons sharing edge
    
    # Configuration: 3-4-4 arrangement (rows of 3, 4, 4 hexagons)
    # Row 0: 3 hexagons
    # Row 1: 4 hexagons (offset)
    # Row 2: 4 hexagons
    
    # For honeycomb packing with flat-top hexagons:
    # Horizontal spacing: 3/2 * side = 1.5 for adjacent columns? No...
    # Let me just use center distance sqrt(3) for neighbors
    
    # Actually, let me try several simple arrangements:
    
    # Arrangement A: Linear row of 11
    positions_a = [(i * hex_spacing, 0.0) for i in range(11)]
    angles_a = [0.0] * 11
    candidates.append((positions_a, angles_a, (0.0, 0.0)))
    
    # Arrangement B: 3-4-4 honeycomb
    # Row 0 (y=0): 3 hexagons at x = -hex_spacing, 0, hex_spacing
    # Row 1 (y=1.5): 4 hexagons at x = -1.5*hex_spacing, -0.5*hex_spacing, 0.5*hex_spacing, 1.5*hex_spacing
    # Wait, for hexagonal lattice, vertical spacing is 1.5 (for unit side) and horizontal offset is hex_spacing/2
    # Actually for pointy-top hexagons in honeycomb:
    # dx = sqrt(3), dy = 1.5 for adjacent rows, with x offset of sqrt(3)/2
    
    dx = np.sqrt(3)
    dy = 1.5
    
    # 3-4-4 arrangement
    positions_b = []
    angles_b = []
    # Row 0: 3 hexagons
    for i in range(3):
        positions_b.append(((i - 1) * dx, 0.0))
        angles_b.append(0.0)
    # Row 1: 4 hexagons, offset by dx/2 in x, shifted by dy in y
    for i in range(4):
        positions_b.append(((i - 1.5) * dx, dy))
        angles_b.append(0.0)
    # Row 2: 4 hexagons
    for i in range(4):
        positions_b.append(((i - 1.5) * dx, 2 * dy))
        angles_b.append(0.0)
    candidates.append((positions_b, angles_b, (0.0, dy)))
    
    # Arrangement C: 4-3-4 arrangement
    positions_c = []
    angles_c = []
    for i in range(4):
        positions_c.append(((i - 1.5) * dx, 0.0))
        angles_c.append(0.0)
    for i in range(3):
        positions_c.append(((i - 1) * dx, dy))
        angles_c.append(0.0)
    for i in range(4):
        positions_c.append(((i - 1.5) * dx, 2 * dy))
        angles_c.append(0.0)
    candidates.append((positions_c, angles_c, (0.0, dy)))
    
    # Arrangement D: 2-3-3-3 arrangement
    positions_d = []
    angles_d = []
    for i in range(2):
        positions_d.append(((i - 0.5) * dx, 0.0))
        angles_d.append(0.0)
    for i in range(3):
        positions_d.append(((i - 1) * dx, dy))
        angles_d.append(0.0)
    for i in range(3):
        positions_d.append(((i - 1) * dx, 2 * dy))
        angles_d.append(0.0)
    for i in range(3):
        positions_d.append(((i - 1) * dx, 3 * dy))
        angles_d.append(0.0)
    candidates.append((positions_d, angles_d, (0.0, 1.5 * dy)))
    
    # Arrangement E: 3-5-3 arrangement
    positions_e = []
    angles_e = []
    for i in range(3):
        positions_e.append(((i - 1) * dx, 0.0))
        angles_e.append(0.0)
    for i in range(5):
        positions_e.append(((i - 2) * dx, dy))
        angles_e.append(0.0)
    for i in range(3):
        positions_e.append(((i - 1) * dx, 2 * dy))
        angles_e.append(0.0)
    candidates.append((positions_e, angles_e, (0.0, dy)))
    
    # Arrangement F: 5-1-5 (less likely but try)
    positions_f = []
    angles_f = []
    for i in range(5):
        positions_f.append(((i - 2) * dx, 0.0))
        angles_f.append(0.0)
    positions_f.append((0.0, dy))
    angles_f.append(0.0)
    for i in range(5):
        positions_f.append(((i - 2) * dx, 2 * dy))
        angles_f.append(0.0)
    candidates.append((positions_f, angles_f, (0.0, dy)))
    
    # Arrangement G: 2-4-2-3
    positions_g = []
    angles_g = []
    for i in range(2):
        positions_g.append(((i - 0.5) * dx, 0.0))
        angles_g.append(0.0)
    for i in range(4):
        positions_g.append(((i - 1.5) * dx, dy))
        angles_g.append(0.0)
    for i in range(2):
        positions_g.append(((i - 0.5) * dx, 2 * dy))
        angles_g.append(0.0)
    for i in range(3):
        positions_g.append(((i - 1) * dx, 3 * dy))
        angles_g.append(0.0)
    candidates.append((positions_g, angles_g, (0.0, 1.5 * dy)))
    
    # Try rotating some inner hexagons by 30 degrees (pointy-top to flat-top)
    # Arrangement B with 30-degree rotations
    candidates.append((positions_b, [30.0] * 11, (0.0, dy)))
    candidates.append((positions_c, [30.0] * 11, (0.0, dy)))
    candidates.append((positions_e, [30.0] * 11, (0.0, dy)))
    
    # Evaluate all candidates
    for idx, (positions, angles_deg, outer_center) in enumerate(candidates):
        if time.time() > deadline:
            break
        
        result = _try_configuration(positions, angles_deg, outer_center, outer_rotation_steps=720)
        inner_list, oc, s, rot = result
        score = s
        
        if score < best_score:
            best_score = score
            best_result = (inner_list, oc, s, rot)
    
    # If no result, use a fallback
    if best_result is None:
        inner = [[i * np.sqrt(3), 0.0, 0.0] for i in range(11)]
        return inner, (0.0, 0.0), 20.0, 0.0
    
    return best_result
# EVOLVE-BLOCK-END
