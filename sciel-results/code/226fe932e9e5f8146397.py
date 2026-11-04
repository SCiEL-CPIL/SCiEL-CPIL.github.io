import numpy as np
import time
from scipy.optimize import minimize_scalar, minimize

N = 11
DEADLINE = 140.0

def _hex_vertices(cx, cy, angle_deg, s=1.0):
    """Vertices of a regular hexagon with side s, centered at (cx,cy), rotated angle_deg."""
    angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]) + angle_deg)
    vs = np.column_stack([s * np.cos(angles), s * np.sin(angles)])
    return vs + np.array([cx, cy])

def _convex_hull_2d(points):
    """Compute convex hull of 2D points using Graham scan."""
    points = np.array(points)
    n = len(points)
    if n <= 3:
        return points
    
    # Find point with lowest y (and leftmost if tie)
    idx = np.lexsort((points[:, 0], points[:, 1]))[0]
    pivot = points[idx]
    
    # Sort by polar angle around pivot
    def angle_func(p):
        return np.arctan2(p[1] - pivot[1], p[0] - pivot[0])
    
    sorted_idx = np.argsort([angle_func(p) for p in points])
    sorted_pts = points[sorted_idx]
    
    # Build lower hull
    lower = []
    for p in sorted_pts:
        while len(lower) >= 2:
            a, b = lower[-2], lower[-1]
            cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
            if cross <= 1e-12:
                lower.pop()
            else:
                break
        lower.append(p)
    
    # Build upper hull
    upper = []
    for p in sorted_pts[::-1]:
        while len(upper) >= 2:
            a, b = upper[-2], upper[-1]
            cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
            if cross <= 1e-12:
                upper.pop()
            else:
                break
        upper.append(p)
    
    hull = lower[:-1] + upper[:-1]
    return np.array(hull)

def _min_enclosing_hex_side(hull, outer_angle_deg=30.0):
    """Compute minimum side length of regular hexagon (with given rotation) enclosing the hull."""
    outer_angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]) + outer_angle_deg)
    normals = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    
    # For each normal direction, compute max projection
    projections = hull @ normals.T  # (n_hull, 6)
    max_proj = np.max(projections, axis=0)  # (6,)
    
    # For a regular hexagon with side S and center at origin:
    # The support function in direction at angle theta is S * cos(30 - theta) for theta in [0,60]
    # More precisely, for flat-top hexagon (angle 30), the width in direction theta is:
    # S * (cos(30) * |cos(theta - 30)| + sin(30) * |sin(theta - 30)|) for appropriate theta
    # Actually, the support function of a regular hexagon with side S centered at origin:
    # h(theta) = S * max over vertices of dot(vertex, direction)
    # For unit side hexagon with vertices at angles 0,60,...,300:
    # The support in direction theta is S * cos(min distance from theta to nearest vertex angle)
    
    # Let's compute the required S for each normal direction
    # For direction i with angle alpha_i, the hexagon vertex in that direction has angle beta_i
    # The support is S * cos(alpha_i - beta_i) where beta_i is the nearest vertex angle
    # But actually, for a regular hexagon, the support function is:
    # h(theta) = S * cos(30 * ((theta % 60) - 30)) for theta in [0,60], etc.
    
    # Simpler: for each of the 6 normal directions, compute what S must be
    # such that the hexagon's support in that direction equals max_proj[i]
    # For a regular hexagon with side S, the support in direction at angle delta from 
    # the nearest vertex direction is S * cos(delta) where delta is in [-30, 30]
    
    # Vertex angles for outer hexagon (angle 30): 30, 90, 150, 210, 270, 330
    vertex_angles = np.deg2rad(np.array([30, 90, 150, 210, 270, 330]))
    
    # Normal directions are perpendicular to edges, so at angles: 0, 60, 120, 180, 240, 300
    # Wait, let me reconsider. For a regular hexagon with vertices at angles v_k = v_0 + k*60,
    # the edge normals are at angles v_0 + 30 + k*60
    
    # If outer_angle_deg = 30, vertices are at 30, 90, 150, 210, 270, 330
    # Edge normals are at 0, 60, 120, 180, 240, 300
    
    # For a regular hexagon with side S, the distance from center to each edge (apothem) is S*sqrt(3)/2
    # The support in the normal direction is exactly the apothem = S*sqrt(3)/2
    
    # So for each normal direction i, we need: S*sqrt(3)/2 >= max_proj[i]
    # But this assumes the hexagon is centered at origin. We also need to center it optimally.
    
    # For optimal centering, the center should be at the centroid of the min and max projections
    # For each normal direction i, the extent is max_proj[i] - min_proj[i]
    # The center offset in direction i is (max_proj[i] + min_proj[i]) / 2
    
    min_proj = np.min(projections, axis=0)  # (6,)
    extents = max_proj - min_proj  # (6,)
    
    # The side length S must satisfy: S*sqrt(3) >= extent_i for all i
    # (since the full width in normal direction is 2 * apothem = S*sqrt(3))
    S_from_extents = extents / np.sqrt(3)
    S = np.max(S_from_extents)
    
    return S

def _compute_min_side_for_rotation(cluster_angle_deg):
    """Compute minimum outer hexagon side length for given cluster rotation."""
    eps = 1e-9
    
    # Generate cluster centers
    centers = []
    centers.append((0.0, 0.0))
    
    for k in range(6):
        angle = np.deg2rad(k * 60)
        r = np.sqrt(3) + eps
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        centers.append((x, y))
    
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        r = 3.0 + 2 * eps
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        centers.append((x, y))
    
    # Rotate all centers by cluster_angle_deg
    theta = np.deg2rad(cluster_angle_deg)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    
    # Get all vertices of all inner hexagons
    all_vertices = []
    for x, y in centers:
        c = rot @ np.array([x, y])
        verts = _hex_vertices(c[0], c[1], cluster_angle_deg, 1.0)
        all_vertices.append(verts)
    
    all_vertices = np.vstack(all_vertices)
    
    # Compute convex hull
    hull = _convex_hull_2d(all_vertices)
    
    # Find minimum enclosing hexagon side length
    side = _min_enclosing_hex_side(hull, outer_angle_deg=30.0)
    
    return side

def _verify_packing(inner, outer_center, side, outer_angle):
    """Verify that all inner hexagons are inside the outer hexagon and don't overlap."""
    outer_verts = _hex_vertices(outer_center[0], outer_center[1], outer_angle, side)
    
    inner_verts_list = []
    for x, y, a in inner:
        verts = _hex_vertices(x, y, a, 1.0)
        inner_verts_list.append(verts)
        # Check containment
        for v in verts:
            if not _point_in_convex_polygon(v, outer_verts):
                return False
    
    # Check pairwise non-overlap
    for i in range(len(inner_verts_list)):
        for j in range(i + 1, len(inner_verts_list)):
            v1 = inner_verts_list[i]
            v2 = inner_verts_list[j]
            # Check if any vertex of v1 is inside v2
            for v in v1:
                if _point_in_convex_polygon(v, v2):
                    return False
            # Check if any vertex of v2 is inside v1
            for v in v2:
                if _point_in_convex_polygon(v, v1):
                    return False
    return True

def _point_in_convex_polygon(pt, poly):
    """Check if point is inside convex polygon using cross products."""
    n = len(poly)
    sign = 0
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        cross = (b[0] - a[0]) * (pt[1] - a[1]) - (b[1] - a[1]) * (pt[0] - a[0])
        if cross > 1e-12:
            if sign < 0:
                return False
            sign = 1
        elif cross < -1e-12:
            if sign > 0:
                return False
            sign = -1
    return True

def construct():
    start_time = time.time()
    
    # First, find the optimal rotation angle using minimize_scalar
    # The objective is the minimum enclosing hexagon side length
    
    def objective(angle_deg):
        return _compute_min_side_for_rotation(angle_deg)
    
    # Coarse grid search first
    best_angle = 0.0
    best_side = float('inf')
    
    # Sample at 6-degree intervals (hexagon symmetry)
    for a in range(0, 36, 2):  # 0 to 34 degrees, step 2
        if time.time() - start_time > DEADLINE - 30:
            break
        side = objective(a)
        if side < best_side:
            best_side = side
            best_angle = a
    
    # Fine search around the best angle
    if time.time() - start_time < DEADLINE - 20:
        try:
            result = minimize_scalar(
                objective, 
                bounds=(best_angle - 2, best_angle + 2),
                method='bounded',
                options={'xatol': 1e-6}
            )
            if result.success and result.fun < best_side:
                best_angle = result.x
                best_side = result.fun
        except Exception:
            pass
    
    # Even finer search
    if time.time() - start_time < DEADLINE - 10:
        try:
            result = minimize_scalar(
                objective, 
                bounds=(best_angle - 0.5, best_angle + 0.5),
                method='bounded',
                options={'xatol': 1e-8}
            )
            if result.success and result.fun < best_side:
                best_angle = result.x
                best_side = result.fun
        except Exception:
            pass
    
    # Now construct the final packing with the optimal rotation
    eps = 1e-9
    
    centers = []
    centers.append((0.0, 0.0))
    
    for k in range(6):
        angle = np.deg2rad(k * 60)
        r = np.sqrt(3) + eps
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        centers.append((x, y))
    
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        r = 3.0 + 2 * eps
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        centers.append((x, y))
    
    # Rotate all centers by best_angle
    theta = np.deg2rad(best_angle)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    
    inner = []
    for x, y in centers:
        c = rot @ np.array([x, y])
        inner.append([c[0], c[1], best_angle])
    
    # Compute the exact side length
    all_vertices = []
    for x, y in centers:
        c = rot @ np.array([x, y])
        verts = _hex_vertices(c[0], c[1], best_angle, 1.0)
        all_vertices.append(verts)
    
    all_vertices = np.vstack(all_vertices)
    hull = _convex_hull_2d(all_vertices)
    side = _min_enclosing_hex_side(hull, outer_angle_deg=30.0)
    
    # Add a small safety margin
    side += 1e-9
    
    outer_center = (0.0, 0.0)
    outer_angle = 30.0
    
    # Quick verification
    if time.time() - start_time < DEADLINE - 5:
        if not _verify_packing(inner, outer_center, side, outer_angle):
            side += 0.01
            if not _verify_packing(inner, outer_center, side, outer_angle):
                side += 0.1
    
    return inner, outer_center, side, outer_angle
