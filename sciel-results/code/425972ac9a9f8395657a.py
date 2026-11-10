# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _hex_vertices(cx, cy, s, a_deg):
    """Compute 6 vertices of a regular hexagon with center (cx,cy), side s, rotation a_deg."""
    a = np.radians(a_deg)
    angles = a + np.radians(60) * np.arange(6)
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])


def _check_separation(hex1, hex2):
    """Check if two unit hexagons (given as vertex arrays) are separated by > 1e-6 on all 12 edge normals."""
    # For two convex polygons, check all edge normals of both
    # If on any normal the projections overlap, they intersect
    for poly in [hex1, hex2]:
        for i in range(6):
            p1 = poly[i]
            p2 = poly[(i + 1) % 6]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            normal = normal / (np.linalg.norm(normal) + 1e-30)
            # Project both polygons onto this normal
            proj1 = hex1 @ normal
            proj2 = hex2 @ normal
            # Check if intervals are apart by more than 1e-6
            # Intervals are [min1, max1] and [min2, max2]
            # They are apart if max1 + 1e-6 < min2 or max2 + 1e-6 < min1
            if not (np.max(proj1) + 1e-6 < np.min(proj2) or np.max(proj2) + 1e-6 < np.min(proj1)):
                # Overlap on this normal - but we need to check ALL 12 normals
                # Actually, the condition is: separated on AT LEAST ONE of 12 normals
                # So if they overlap on ALL normals, they intersect
                pass
    # Actually, re-reading: "on at least one of their 12 edge normals the projected intervals are apart"
    # This means: for the pair to be valid, there must exist at least one normal (from the 12) where they are apart
    # Wait, that doesn't sound right. Let me re-read.
    # "every pair of inner hexagons is separated: on at least one of their 12 edge normals the projected intervals are apart by more than 1e-6"
    # I think this means: for each pair, we check all 12 normals, and the pair is valid if on at least one normal they are apart.
    # But that's a very weak condition. Actually, I think the correct interpretation is:
    # The two hexagons are separated if for EVERY edge normal of both hexagons, the projections are apart.
    # No wait, that's the separating axis theorem - they are separated if there EXISTS a separating axis.
    # The 12 edge normals are candidate separating axes. If on ANY of these 12 normals the projections are apart,
    # then the hexagons are separated (not intersecting).
    # So: valid if exists a normal in the 12 such that projections are apart by > 1e-6.
    
    # Let me re-implement: return True if separated (valid), False if intersecting (invalid)
    for poly in [hex1, hex2]:
        for i in range(6):
            p1 = poly[i]
            p2 = poly[(i + 1) % 6]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            normal = normal / (np.linalg.norm(normal) + 1e-30)
            proj1 = hex1 @ normal
            proj2 = hex2 @ normal
            if np.max(proj1) + 1e-6 < np.min(proj2) or np.max(proj2) + 1e-6 < np.min(proj1):
                return True  # Found a separating axis
    return False  # No separating axis found among the 12


def _check_inner_in_outer(inner_hexes, outer_hex):
    """Check all inner vertices are inside or on the outer hexagon."""
    for hex_verts in inner_hexes:
        for p in hex_verts:
            # Check against each outer edge
            for i in range(6):
                p1 = outer_hex[i]
                p2 = outer_hex[(i + 1) % 6]
                cross = (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])
                if cross < -1e-6:
                    return False
    return True


def _validate(inner_hex_data, outer_hex_data, side):
    """Full validation."""
    if not np.all(np.isfinite(inner_hex_data)) or not np.all(np.isfinite(outer_hex_data)):
        return False
    if side < np.sqrt(11) - 1e-9:
        return False
    
    # Compute vertices
    inner_hexes = []
    for i in range(len(inner_hex_data)):
        x, y, a = inner_hex_data[i]
        verts = _hex_vertices(x, y, 1.0, a)
        inner_hexes.append(verts)
    
    outer_hex = _hex_vertices(outer_hex_data[0], outer_hex_data[1], side, outer_hex_data[2])
    
    # Check separation between all pairs
    for i in range(len(inner_hexes)):
        for j in range(i + 1, len(inner_hexes)):
            if not _check_separation(inner_hexes[i], inner_hexes[j]):
                return False
    
    # Check all inner vertices inside outer
    if not _check_inner_in_outer(inner_hexes, outer_hex):
        return False
    
    # Check edge lengths
    for verts in inner_hexes:
        for i in range(6):
            d = np.linalg.norm(verts[(i + 1) % 6] - verts[i])
            if abs(d - 1.0) > 1e-8:
                return False
    
    return True


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon,
    minimizing the outer hexagon side length (maximizing 1/side).
    
    Uses a hexagonal lattice arrangement with optimization.
    """
    # Hex grid arrangement for 11 hexagons
    # Using axial coordinates, convert to Cartesian
    # Pointy-top hex grid:
    # x = size * sqrt(3) * (q + r/2)
    # y = size * 3/2 * r
    # For unit hexagons, the center-to-center distance for adjacent is sqrt(3)
    # So size (circumradius) = 1, and the grid spacing is:
    # dx = sqrt(3) for horizontal neighbors
    # dy = 1.5 for vertical, with x offset of sqrt(3)/2
    
    # Let's place 11 hexagons in a compact arrangement
    # Using axial coordinates (q, r):
    # A compact shape for 11:
    axial_coords = [
        (0, 0),    # center
        (1, 0),    # right
        (0, 1),    # upper right
        (-1, 1),   # upper left
        (-1, 0),   # left
        (0, -1),   # lower left
        (1, -1),   # lower right
        (2, 0),    # far right
        (1, 1),    # far upper right
        (-2, 1),   # far upper left
        (-2, 0),   # far left
    ]
    
    # Convert to Cartesian
    inner_hex_data = np.zeros((11, 3))
    for i, (q, r) in enumerate(axial_coords):
        x = np.sqrt(3) * (q + r / 2)
        y = 1.5 * r
        inner_hex_data[i] = [x, y, 0]
    
    # Center the arrangement
    center = np.mean(inner_hex_data[:, :2], axis=0)
    inner_hex_data[:, 0] -= center[0]
    inner_hex_data[:, 1] -= center[1]
    
    # Now optimize the outer hexagon
    # Outer hexagon: center (0,0), angle a, side s
    # We want to minimize s such that all inner hexagons fit
    
    def _objective(params):
        s, a_deg = params
        if s < 0:
            return 1e10
        outer_hex_data = np.array([0, 0, a_deg])
        outer_hex = _hex_vertices(0, 0, s, a_deg)
        
        # Check if all inner vertices are inside
        inner_hexes = []
        for i in range(len(inner_hex_data)):
            x, y, ai = inner_hex_data[i]
            verts = _hex_vertices(x, y, 1.0, ai)
            inner_hexes.append(verts)
        
        for hex_verts in inner_hexes:
            for p in hex_verts:
                for i in range(6):
                    p1 = outer_hex[i]
                    p2 = outer_hex[(i + 1) % 6]
                    cross = (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])
                    if cross < -1e-9:
                        return 1e10 + s  # Penalize
        
        return s
    
    # Try multiple initial angles
    best_result = None
    best_s = 1e10
    
    for init_angle in range(0, 60, 5):
        result = minimize(_objective, [10.0, init_angle], method='Nelder-Mead',
                         options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})
        if result.success and result.fun < best_s:
            best_s = result.fun
            best_result = result
    
    if best_result is None:
        # Fallback: use a large enough hexagon
        # Compute the maximum distance from center to any inner vertex
        max_dist = 0
        for i in range(len(inner_hex_data)):
            x, y, a = inner_hex_data[i]
            verts = _hex_vertices(x, y, 1.0, a)
            for v in verts:
                d = np.linalg.norm(v)
                max_dist = max(max_dist, d)
        # For a regular hexagon with side s, the inradius is s*sqrt(3)/2
        # So we need s*sqrt(3)/2 >= max_dist
        s = max_dist * 2 / np.sqrt(3) + 0.1
        outer_hex_data = np.array([0, 0, 0])
        return inner_hex_data, outer_hex_data, s
    
    s, a_deg = best_result.x
    outer_hex_data = np.array([0, 0, a_deg])
    
    # Verify
    if not _validate(inner_hex_data, outer_hex_data, s):
        # Fallback
        max_dist = 0
        for i in range(len(inner_hex_data)):
            x, y, a = inner_hex_data[i]
            verts = _hex_vertices(x, y, 1.0, a)
            for v in verts:
                d = np.linalg.norm(v)
                max_dist = max(max_dist, d)
        s = max_dist * 2 / np.sqrt(3) + 0.5
        outer_hex_data = np.array([0, 0, 0])
    
    return inner_hex_data, outer_hex_data, s


# EVOLVE-BLOCK-END
