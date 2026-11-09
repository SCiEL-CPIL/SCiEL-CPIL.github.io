# EVOLVE-BLOCK-START
import time
import numpy as np


def _hex_vertices(cx, cy, s, a_deg):
    """Vertices of a regular hexagon with center (cx,cy), side s, rotation a_deg."""
    a = np.radians(a_deg)
    angs = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angs), cy + s * np.sin(angs)])


def _hex_normals(a_deg):
    """12 edge normals for a hexagon of rotation a_deg."""
    a = np.radians(a_deg)
    edge_angles = a + np.deg2rad(60.0 * np.arange(6)) + np.deg2rad(30.0)
    normals = np.column_stack([np.cos(edge_angles), np.sin(edge_angles)])
    return np.vstack([normals, -normals])


def _separation_ok(v1, v2, n1, n2):
    """Check if two hexagons are separated on at least one of their 12 edge normals."""
    for normals in (n1, n2):
        for j in range(normals.shape[0]):
            n = normals[j]
            proj1 = v1 @ n
            proj2 = v2 @ n
            min1, max1 = proj1.min(), proj1.max()
            min2, max2 = proj2.min(), proj2.max()
            if max1 < min2 - 1e-6 or max2 < min1 - 1e-6:
                return True
    return False


def _hexagons_separated(v1, v2, a1, a2):
    n1 = _hex_normals(a1)
    n2 = _hex_normals(a2)
    return _separation_ok(v1, v2, n1, n2)


def _point_in_hexagon(p, outer_verts):
    """Check if point p is inside (or on boundary of) the outer hexagon."""
    for i in range(6):
        p1 = outer_verts[i]
        p2 = outer_verts[(i + 1) % 6]
        edge = p2 - p1
        to_p = p - p1
        cross = edge[0] * to_p[1] - edge[1] * to_p[0]
        if cross < -1e-6:
            return False
    return True


def _all_inner_in_outer(inner_verts_list, outer_verts):
    """Check all vertices of all inner hexagons are inside the outer hexagon."""
    for verts in inner_verts_list:
        for p in verts:
            if not _point_in_hexagon(p, outer_verts):
                return False
    return True


def _build_hexagons(r1, r2, theta1, theta2):
    """
    Build 11 hexagons: center + 6 inner ring + 4 outer ring.
    r1: radius of inner ring (6 hexagons)
    r2: radius of outer ring (4 hexagons)
    theta1: angular offset for inner ring (deg)
    theta2: angular offset for outer ring (deg)
    """
    hexagons = []
    # Center hexagon
    hexagons.append((0.0, 0.0, 0.0))
    # Inner ring: 6 hexagons at radius r1, angles theta1 + 60*i
    for i in range(6):
        ang = np.radians(theta1 + 60.0 * i)
        cx = r1 * np.cos(ang)
        cy = r1 * np.sin(ang)
        hexagons.append((cx, cy, 0.0))
    # Outer ring: 4 hexagons at radius r2, angles theta2 + 90*i
    for i in range(4):
        ang = np.radians(theta2 + 90.0 * i)
        cx = r2 * np.cos(ang)
        cy = r2 * np.sin(ang)
        hexagons.append((cx, cy, 0.0))
    return hexagons


def _compute_minimal_outer_hex(hexagons):
    """Compute minimal enclosing regular hexagon for the given inner hexagons.
    Returns (center_x, center_y, angle_deg, side_length)."""
    # Get all vertices of all inner hexagons
    all_verts = []
    for (cx, cy, ang) in hexagons:
        verts = _hex_vertices(cx, cy, 1.0, ang)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    # For a regular hexagon centered at origin with angle a, side s:
    # apothem = s * sqrt(3)/2
    # For each of 6 outward normals n_i, need:
    #   max(p @ n_i) <= apothem and min(p @ n_i) >= -apothem
    # So apothem >= max_i(max(p @ n_i), -min(p @ n_i))
    # s = apothem / (sqrt(3)/2)
    
    best_s = np.inf
    best_a = 0.0
    
    for a_deg in np.linspace(0, 60, 61):
        a = np.radians(a_deg)
        edge_angles = a + np.deg2rad(60.0 * np.arange(6)) + np.deg2rad(30.0)
        normals = np.column_stack([np.cos(edge_angles), np.sin(edge_angles)])
        
        proj = all_verts @ normals.T  # (N, 6)
        max_proj = proj.max(axis=0)  # (6,)
        min_proj = proj.min(axis=0)  # (6,)
        
        required_apothem = np.maximum(max_proj, -min_proj)
        apothem = required_apothem.max()
        s = apothem / (np.sqrt(3) / 2)
        
        if s < best_s:
            best_s = s
            best_a = a_deg
    
    return (0.0, 0.0, best_a, best_s)


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon.
    Returns (inner_hex_data, outer_hex_data, outer_hex_side_length).
    """
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin before 630s limit
    
    # Grid search over r1, r2, theta1, theta2
    # r1: distance from center to inner ring centers
    # r2: distance from center to outer ring centers
    # theta1: angular offset for inner ring
    # theta2: angular offset for outer ring
    
    # Initial estimates:
    # For unit hexagons touching edge-to-edge, center distance = sqrt(3) ≈ 1.732
    # So r1 ≈ sqrt(3), r2 ≈ 2*sqrt(3) ≈ 3.464
    
    r1_vals = np.linspace(1.5, 2.5, 21)  # 21 values
    r2_vals = np.linspace(3.0, 4.5, 31)  # 31 values
    theta1_vals = np.linspace(0, 60, 13)  # 13 values (0 to 60 deg, step 5)
    theta2_vals = np.linspace(0, 90, 19)  # 19 values (0 to 90 deg, step 5)
    
    best_s = np.inf
    best_params = None
    best_hexagons = None
    
    count = 0
    for r1 in r1_vals:
        if time.time() > deadline:
            break
        for r2 in r2_vals:
            if time.time() > deadline:
                break
            for theta1 in theta1_vals:
                if time.time() > deadline:
                    break
                for theta2 in theta2_vals:
                    if time.time() > deadline:
                        break
                    
                    count += 1
                    hexagons = _build_hexagons(r1, r2, theta1, theta2)
                    
                    # Check separation between all pairs
                    verts_list = [_hex_vertices(cx, cy, 1.0, ang) for (cx, cy, ang) in hexagons]
                    
                    valid = True
                    for i in range(11):
                        if not valid:
                            break
                        for j in range(i + 1, 11):
                            if not _hexagons_separated(verts_list[i], verts_list[j], hexagons[i][2], hexagons[j][2]):
                                valid = False
                                break
                    
                    if not valid:
                        continue
                    
                    # Compute minimal enclosing hexagon
                    ox, oy, oa, s = _compute_minimal_outer_hex(hexagons)
                    
                    # Check containment
                    outer_verts = _hex_vertices(ox, oy, s, oa)
                    if not _all_inner_in_outer(verts_list, outer_verts):
                        continue
                    
                    if s < best_s:
                        best_s = s
                        best_params = (r1, r2, theta1, theta2)
                        best_hexagons = hexagons
    
    if best_hexagons is None:
        # Fallback: use initial estimates
        r1, r2, theta1, theta2 = np.sqrt(3), 2 * np.sqrt(3), 0.0, 0.0
        best_hexagons = _build_hexagons(r1, r2, theta1, theta2)
        ox, oy, oa, best_s = _compute_minimal_outer_hex(best_hexagons)
    
    # Build final output
    inner_hex_data = np.array(best_hexagons)
    outer_hex_data = np.array([0.0, 0.0, oa])
    
    # Ensure s >= sqrt(11)
    best_s = max(best_s, np.sqrt(11.0))
    
    return inner_hex_data, outer_hex_data, float(best_s)


# EVOLVE-BLOCK-END
