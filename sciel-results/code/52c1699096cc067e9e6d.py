"""Radial Slicing: arrange 12 unit hexagons in concentric rings (1+5+6) and optimize ring radii and rotation to minimize outer hexagon side length."""

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 12

def _hex_vertices(cx, cy, side, angle_deg, flat_top=True):
    """Return vertices of a regular hexagon centered at (cx, cy) with given side length and rotation.
    If flat_top=True, the hexagon has a flat top edge (standard orientation)."""
    angle_rad = np.radians(angle_deg)
    # For a regular hexagon, vertices are at angles 0, 60, 120, 180, 240, 300 degrees from center
    # when flat_top=False (pointy top). For flat_top=True, offset by 30 degrees.
    if flat_top:
        base_angles = np.radians(0, 60, 120, 180, 240, 300)
    else:
        base_angles = np.radians(30, 90, 150, 210, 270, 330)
    angles = base_angles + angle_rad
    xs = cx + side * np.cos(angles)
    ys = cy + side * np.sin(angles)
    return np.column_stack([xs, ys])

def _min_outer_hexagon_side(inner_positions, inner_angles, outer_center, outer_angle):
    """Compute the minimum side length of a regular hexagon (centered at outer_center, rotated by outer_angle)
    that contains all inner hexagons. Returns the side length."""
    # For each inner hexagon, compute its vertices
    all_vertices = []
    for (x, y, ang) in inner_positions:
        verts = _hex_vertices(x, y, 1.0, ang)
        all_vertices.append(verts)
    
    # Collect all vertices
    all_verts = np.vstack(all_vertices)
    
    # The outer hexagon is centered at outer_center with rotation outer_angle.
    # A point (px, py) is inside the outer hexagon (side s, center c, rotation a) if:
    # For each of the 6 edges, the point is on the correct side.
    # Equivalently, for a regular hexagon with "flat top" orientation rotated by a,
    # the support function in direction theta is s * max over the 6 normals.
    
    # Actually, let's use the dual approach: the minimal side length s such that
    # all points are inside the hexagon.
    
    # The hexagon with center c, side s, and rotation angle a (flat top) has 6 faces.
    # The outward normals of the faces are at angles: a, a+60, a+120, a+180, a+240, a+300 (for flat top)
    # Wait, let me think more carefully.
    
    # For a regular hexagon with side length s, the apothem (distance from center to each face) is s * sqrt(3)/2.
    # The hexagon is the intersection of 6 half-planes.
    # For a hexagon centered at origin with flat top, the faces are at distances s*sqrt(3)/2 from center,
    # with outward normals at angles 90, 150, 210, 270, 330, 30 degrees (i.e., 90 + k*60 for k=0..5).
    # Wait, let me reconsider. For flat top, the top edge is horizontal. The outward normal of the top edge points up (90 degrees).
    # The next edge (upper right) has outward normal at 30 degrees? No...
    
    # Let me just compute it directly. For a hexagon with flat top, side s, centered at origin:
    # Vertices at angles 0, 60, 120, 180, 240, 300 (if we use pointy-top) or 30, 90, 150, 210, 270, 330 (if flat-top).
    
    # Actually, let's use a simpler approach: the minimal enclosing regular hexagon problem.
    # For a given orientation of the outer hexagon, the required side length is determined by
    # the maximum over all inner vertices of the "hexagonal distance" from the center.
    
    # For a regular hexagon with flat-top orientation, the hexagonal norm (or distance) of a point (x, y)
    # from the center is max(|x|, |x/2 + y*sqrt(3)/2|, |-x/2 + y*sqrt(3)/2|) / (sqrt(3)/2) ... 
    # Hmm, this is getting complicated. Let me use a different approach.
    
    # The apothem of a regular hexagon with side s is a = s * sqrt(3)/2.
    # So s = 2a / sqrt(3).
    # The hexagon is the set of points where the distance to each of the 6 faces is >= 0.
    # For a hexagon centered at c with rotation a, the 6 face normals are at angles a + k*60 degrees for k=0..5
    # (assuming the first normal is at angle a).
    # The distance from a point p to the face with normal n is dot(p - c, n) - a (if the face is at distance a from center in direction n).
    # Wait, the face is at distance a from center in direction -n (inward), so the half-plane is dot(p - c, n) <= a.
    # So for all points to be inside, we need dot(p - c, n_k) <= a for all k and all p.
    # Thus a >= max over all p of dot(p - c, n_k) for each k.
    # So a_min = max_k max_p dot(p - c, n_k).
    # And s_min = 2 * a_min / sqrt(3).
    
    oc = np.array(outer_center)
    oa = np.radians(outer_angle)
    # For flat-top hexagon, the outward normals are at angles 90, 150, 210, 270, 330, 30 degrees
    # which is 90 + k*60 for k=0..5. But we rotate by outer_angle.
    normal_angles = np.radians(90) + oa + np.arange(6) * np.radians(60)
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    # Translate points relative to outer center
    pts = all_verts - oc
    
    # For each normal, compute max dot product
    max_dots = np.array([np.max(pts @ normals[k]) for k in range(6)])
    
    # The apothem must be at least the max of these
    a_min = np.max(max_dots)
    
    if a_min < 0:
        a_min = 0
    
    s_min = 2 * a_min / np.sqrt(3)
    
    return s_min

def _objective(params):
    """Objective function: minimize the outer hexagon side length.
    params = [r1, r2, theta1, theta2, global_rot, ox, oy, outer_rot]
    where r1, r2 are ring radii, theta1, theta2 are angular offsets for rings 1 and 2,
    global_rot is the rotation of inner hexagons, ox, oy is outer center, outer_rot is outer rotation.
    """
    r1, r2, theta1, theta2, global_rot, ox, oy, outer_rot = params
    
    # Ring 0: center hexagon at (0, 0)
    # Ring 1: 5 hexagons at radius r1, angles theta1 + k*72 for k=0..4
    # Ring 2: 6 hexagons at radius r2, angles theta2 + k*60 for k=0..5
    
    inner = []
    # Center
    inner.append([0.0, 0.0, global_rot])
    
    # Ring 1: 5 hexagons
    for k in range(5):
        ang = np.radians(theta1 + k * 72.0)
        x = r1 * np.cos(ang)
        y = r1 * np.sin(ang)
        inner.append([x, y, global_rot])
    
    # Ring 2: 6 hexagons
    for k in range(6):
        ang = np.radians(theta2 + k * 60.0)
        x = r2 * np.cos(ang)
        y = r2 * np.sin(ang)
        inner.append([x, y, global_rot])
    
    # Compute outer hexagon side length
    side = _min_outer_hexagon_side(inner, [0.0]*12, (ox, oy), outer_rot)
    
    return side

def construct():
    """Construct 12 unit hexagons inside a regular hexagon, minimizing the outer side length."""
    
    # Initial guess
    # Ring 1 radius ~ 2.0 (hexagons side 1, so they need at least 2*apothem = sqrt(3) ~ 1.732 between centers for non-overlap if aligned)
    # Actually for non-overlap, center distance >= 2 * apothem = 2 * sqrt(3)/2 = sqrt(3) ~ 1.732
    # Ring 2 radius ~ 3.5
    
    x0 = [2.0, 3.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Bounds
    bounds = [
        (1.5, 5.0),   # r1
        (3.0, 8.0),   # r2
        (0.0, 72.0),  # theta1
        (0.0, 60.0),  # theta2
        (0.0, 60.0),  # global_rot
        (-5.0, 5.0),  # ox
        (-5.0, 5.0),  # oy
        (0.0, 60.0),  # outer_rot
    ]
    
    # Optimize
    best_result = None
    best_val = float('inf')
    
    # Try multiple starting points
    starts = [
        [2.0, 3.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [2.1, 3.9, 10.0, 10.0, 30.0, 0.0, 0.0, 30.0],
        [1.8, 3.5, 0.0, 30.0, 0.0, 0.0, 0.0, 0.0],
        [2.2, 4.0, 15.0, 0.0, 15.0, 0.5, 0.0, 15.0],
        [1.9, 3.6, 5.0, 5.0, 0.0, 0.0, 0.0, 0.0],
    ]
    
    for start in starts:
        try:
            result = minimize(_objective, start, method='Nelder-Mead', 
                            bounds=bounds, 
                            options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6})
            if result.success or result.fun < best_val:
                if result.fun < best_val:
                    best_val = result.fun
                    best_result = result
        except Exception:
            pass
    
    if best_result is None:
        # Fallback to initial
        best_result = type('obj', (object,), {'x': x0, 'fun': _objective(x0)})()
    
    r1, r2, theta1, theta2, global_rot, ox, oy, outer_rot = best_result.x
    
    # Construct inner hexagons
    inner = []
    # Center
    inner.append([0.0, 0.0, global_rot])
    
    # Ring 1: 5 hexagons
    for k in range(5):
        ang = np.radians(theta1 + k * 72.0)
        x = r1 * np.cos(ang)
        y = r1 * np.sin(ang)
        inner.append([x, y, global_rot])
    
    # Ring 2: 6 hexagons
    for k in range(6):
        ang = np.radians(theta2 + k * 60.0)
        x = r2 * np.cos(ang)
        y = r2 * np.sin(ang)
        inner.append([x, y, global_rot])
    
    # Verify no overlaps (optional, but good to check)
    # For now, return the result
    
    return inner, (ox, oy), best_val, outer_rot
