# EVOLVE-BLOCK-START
import numpy as np
import time

def _hex_vertices(cx, cy, d, angle_deg):
    """Vertices of a regular hexagon centered at (cx,cy) with side d and angle (deg)."""
    a = np.radians(angle_deg)
    thetas = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + d * np.cos(thetas),
                            cy + d * np.sin(thetas)])

def _hex_edge_normals(vertices):
    """Outward edge normals for a hexagon given its 6 vertices (CCW)."""
    v = vertices
    edges = np.roll(v, -1, axis=0) - v
    normals = np.column_stack([edges[:, 1], -edges[:, 0]])
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.where(norms == 0, 1.0, norms)
    return normals

def _separation_violation(verts1, verts2):
    """Separating axis test: returns min gap (negative = overlap)."""
    n1 = _hex_edge_normals(verts1)
    n2 = _hex_edge_normals(verts2)
    all_normals = np.vstack([n1, n2])
    
    min_gap = np.inf
    for normal in all_normals:
        proj1 = verts1 @ normal
        proj2 = verts2 @ normal
        gap1 = proj2.min() - proj1.max()
        gap2 = proj1.min() - proj2.max()
        min_gap = min(min_gap, gap1, gap2)
    return min_gap

def _point_in_hexagon(v, outer_verts):
    """Check if point v is inside the outer hexagon using cross products."""
    for i in range(6):
        p1 = outer_verts[i]
        p2 = outer_verts[(i + 1) % 6]
        cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
        if cross < -1e-6:
            return False
    return True

def _compute_s_for_centers(centers, angles, outer_angle=0.0):
    """
    Given centers and angles of 12 unit hexagons, compute the minimum s
    of a regular hexagon (centered at origin, given angle) that contains all of them.
    """
    max_s = 0.0
    outer_a = np.radians(outer_angle)
    for i in range(12):
        verts = _hex_vertices(centers[i, 0], centers[i, 1], 1.0, angles[i])
        for v in verts:
            px, py = v[0], v[1]
            for k in range(6):
                angle = outer_a + np.radians(30 + 60 * k)
                val = px * np.cos(angle) + py * np.sin(angle)
                s_required = val / (np.sqrt(3) / 2)
                if s_required > max_s:
                    max_s = s_required
    return max_s

def _validate_configuration(centers, angles, s, outer_angle=0.0):
    """Validate a configuration: check overlaps and boundary containment."""
    # Check s constraint
    if s < np.sqrt(12) - 1e-8:
        return False
    
    # Check overlaps between inner hexagons
    for i in range(12):
        verts_i = _hex_vertices(centers[i, 0], centers[i, 1], 1.0, angles[i])
        for j in range(i + 1, 12):
            verts_j = _hex_vertices(centers[j, 0], centers[j, 1], 1.0, angles[j])
            gap = _separation_violation(verts_i, verts_j)
            if gap < -1e-6:
                return False
    
    # Check boundary: all inner vertices must be inside outer hexagon
    outer_verts = _hex_vertices(0, 0, s, outer_angle)
    for i in range(12):
        verts_i = _hex_vertices(centers[i, 0], centers[i, 1], 1.0, angles[i])
        for v in verts_i:
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k + 1) % 6]
                cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
                if cross < -1e-6:
                    return False
    
    return True

def _try_configuration(centers, angles, outer_angle=0.0):
    """Try a configuration and return (valid, score) if valid."""
    s = _compute_s_for_centers(centers, angles, outer_angle)
    
    # Add small margin for numerical safety
    s_test = s + 1e-9
    
    if _validate_configuration(centers, angles, s_test, outer_angle):
        return True, 1.0 / s
    return False, 0.0

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 625  # Safety margin
    
    best_score = 0.0
    best_result = None
    
    # Hexagonal lattice basis vectors for unit hexagons (side 1)
    # Distance between adjacent hex centers: 2 * sqrt(3) / 2 * 2 = 2*sqrt(3) for touching
    # Actually for regular hexagons with side 1, the distance between centers of adjacent
    # touching hexagons is 2 * (sqrt(3)/2) * 2 = 2*sqrt(3)? No.
    # The apothem (inradius) of a unit hexagon is sqrt(3)/2.
    # Two touching hexagons have center distance = 2 * apothem = sqrt(3).
    # Wait, that's for flat-to-flat contact. For point-to-point, it's different.
    # For regular hexagons with side s, the distance between centers of two touching
    # hexagons (edge to edge) is 2 * s * sqrt(3) / 2 = s * sqrt(3).
    # So for s=1, distance = sqrt(3).
    
    d = np.sqrt(3)  # Distance between centers of touching unit hexagons
    
    # Configuration 1: 1 center + 6 ring + 5 more (total 12)
    # Center at origin
    # Ring 1: 6 hexagons at distance d, angles 0, 60, 120, 180, 240, 300
    # Ring 2: 5 more hexagons at distance 2*d, angles 30, 90, 150, 210, 270
    
    configs = []
    
    # Config A: Flower + 5 outer
    centers_a = np.array([[0.0, 0.0]])  # Center
    for k in range(6):
        angle = np.radians(60 * k)
        centers_a = np.vstack([centers_a, [d * np.cos(angle), d * np.sin(angle)]])
    # Add 5 more at 2d distance, offset by 30 degrees
    for k in range(5):
        angle = np.radians(30 + 60 * k)
        centers_a = np.vstack([centers_a, [2 * d * np.cos(angle), 2 * d * np.sin(angle)]])
    
    # Center the configuration
    centers_a -= centers_a.mean(axis=0)
    angles_a = np.zeros(12)
    configs.append((centers_a, angles_a, 0.0))
    
    # Config B: 3-4-3 arrangement (rows)
    # Row 0: 3 hexagons
    # Row 1: 4 hexagons  
    # Row 2: 3 hexagons
    row_h = d * np.sqrt(3) / 2  # Vertical spacing between rows
    row_w = d  # Horizontal spacing
    
    centers_b = []
    # Row 0: 3 hexagons
    for k in range(3):
        centers_b.append([(k - 1) * row_w, 0])
    # Row 1: 4 hexagons, offset by row_w/2
    for k in range(4):
        centers_b.append([(k - 1.5) * row_w, row_h])
    # Row 2: 3 hexagons
    for k in range(3):
        centers_b.append([(k - 1) * row_w, 2 * row_h])
    
    centers_b = np.array(centers_b)
    centers_b -= centers_b.mean(axis=0)
    angles_b = np.zeros(12)
    configs.append((centers_b, angles_b, 0.0))
    
    # Config C: 4-4-4 arrangement
    centers_c = []
    for row in range(3):
        for k in range(4):
            x_offset = (k - 1.5) * row_w
            if row % 2 == 1:
                x_offset += row_w / 2
            centers_c.append([x_offset, row * row_h])
    
    centers_c = np.array(centers_c)
    centers_c -= centers_c.mean(axis=0)
    angles_c = np.zeros(12)
    configs.append((centers_c, angles_c, 0.0))
    
    # Config D: Try different outer hexagon angles
    for outer_angle in [0, 30]:
        centers_d = centers_a.copy()
        angles_d = np.zeros(12)
        configs.append((centers_d, angles_d, float(outer_angle)))
    
    # Evaluate all configurations
    for centers, angles, outer_angle in configs:
        if time.time() > deadline:
            break
        
        valid, score = _try_configuration(centers, angles, outer_angle)
        if valid and score > best_score:
            best_score = score
            s = _compute_s_for_centers(centers, angles, outer_angle)
            best_result = (centers.copy(), angles.copy(), s, outer_angle)
    
    # If no valid configuration found, use a simple fallback
    if best_result is None:
        # Very conservative: large hexagon
        centers = np.array([
            [0.0, 0.0],
            [2.0, 0.0], [-2.0, 0.0],
            [1.0, 1.732], [-1.0, 1.732],
            [1.0, -1.732], [-1.0, -1.732],
            [3.0, 1.732], [-3.0, 1.732],
            [3.0, -1.732], [-3.0, -1.732],
            [0.0, 3.464],
        ])
        angles = np.zeros(12)
        s = 6.0
        outer_angle = 0.0
        best_result = (centers, angles, s, outer_angle)
        best_score = 1.0 / s
    
    centers, angles, s, outer_angle = best_result
    
    inner_hex_data = np.column_stack([centers, angles])
    outer_hex_data = np.array([0.0, 0.0, outer_angle])
    
    return inner_hex_data, outer_hex_data, float(s)

# EVOLVE-BLOCK-END
