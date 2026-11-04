import numpy as np
import time
from scipy.optimize import minimize

N = 11
DEADLINE = 130.0

def _hex_vertices(cx, cy, angle_deg, s=1.0):
    """Vertices of a regular hexagon with side s, centered at (cx,cy), rotated angle_deg."""
    angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]) + angle_deg)
    vs = np.column_stack([s * np.cos(angles), s * np.sin(angles)])
    return vs + np.array([cx, cy])

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

def _segment_intersects_segment(p1, p2, p3, p4):
    """Check if segment p1-p2 intersects segment p3-p4 (proper intersection)."""
    def orient(a, b, c):
        val = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
        if abs(val) < 1e-12:
            return 0
        return 1 if val > 0 else 2
    
    o1 = orient(p1, p2, p3)
    o2 = orient(p1, p2, p4)
    o3 = orient(p3, p4, p1)
    o4 = orient(p3, p4, p2)
    
    if o1 != o2 and o3 != o4:
        return True
    return False

def _polygons_overlap(v1, v2):
    """Check if two convex polygons overlap using SAT + vertex containment + edge intersection."""
    # Check vertex containment
    for v in v1:
        if _point_in_convex_polygon(v, v2):
            return True
    for v in v2:
        if _point_in_convex_polygon(v, v1):
            return True
    
    # Check edge intersections
    n1 = len(v1)
    n2 = len(v2)
    for i in range(n1):
        for j in range(n2):
            if _segment_intersects_segment(v1[i], v1[(i+1)%n1], v2[j], v2[(j+1)%n2]):
                return True
    return False

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
            if _polygons_overlap(inner_verts_list[i], inner_verts_list[j]):
                return False
    return True

def _compute_outer_side(centers, angles, outer_angle_deg=30.0):
    """
    Compute minimal outer hexagon side length given inner hexagon centers and angles.
    The outer hexagon is centered at origin with given rotation.
    """
    # Get all vertices of all inner hexagons
    all_verts = []
    for (cx, cy), ang in zip(centers, angles):
        verts = _hex_vertices(cx, cy, ang, 1.0)
        all_verts.append(verts)
    
    # For a hexagon centered at origin with angle theta and side S:
    # The support function in direction d is S * max over vertices of (v . d)
    # where v are unit hexagon vertices
    # Actually, for a regular hexagon with "radius" (center to vertex) = S,
    # the support in direction d is S * max(|d_x|, |d_y * sqrt(3)/2|, |(d_x + d_y*sqrt(3))/2|, ...)
    
    # Simpler: for a hexagon with vertices at angles [0,60,120,180,240,300]+offset,
    # the support function h(d) = S * max_i (cos(angle_i - d_angle))
    # But we need the hexagon to contain all points, so:
    # For each direction corresponding to a face normal of the outer hexagon,
    # the max projection of all inner vertices must be <= S * (apothem of unit hex in that dir)
    
    # Outer hexagon with angle outer_angle_deg: its 6 face normals are at angles
    # outer_angle_deg + 30 + k*60 for k=0..5 (for flat-top) or similar
    # Actually, for a hexagon with vertices at angles [0,60,...]+offset,
    # the face normals are at angles [30,90,...]+offset
    
    # Let me think differently. A regular hexagon centered at origin with side S
    # and rotation phi has vertices at distance S from center at angles phi + k*60.
    # It can be described as intersection of 6 half-planes.
    # The support function of a unit hexagon (S=1) in direction theta is:
    # h(theta) = max over vertices v of (v . (cos theta, sin theta))
    
    # For containment: for all points p in inner hexagons, and for all 6 face directions,
    # p . n_i <= S * h_unit(n_i_direction)
    
    # The 6 outward normals of the outer hexagon (with vertex angles phi+k*60) are at
    # angles phi + 30 + k*60 for k=0..5
    
    phi = np.deg2rad(outer_angle_deg)
    normal_angles = phi + np.pi/6 + np.arange(6) * np.pi/3
    
    # For unit hexagon with vertices at angles [0,60,...], support in direction alpha:
    # h(alpha) = max_k cos(alpha - k*pi/3)
    # For the outer hexagon with rotation phi, its normals are at phi+30+k*60
    # The support of unit hex (same orientation as outer) in its own normal direction is:
    # h = max_k cos(0) = 1 (for the normal aligned with a vertex)
    # Actually for a hexagon, the apothem (distance to face) = S * sqrt(3)/2
    # and the vertex distance = S
    # The support in the normal direction (perpendicular to a face) is S * sqrt(3)/2
    
    # Let me just compute directly:
    # Outer hexagon vertices (unit, S=1, rotation phi):
    unit_outer_verts = _hex_vertices(0, 0, outer_angle_deg, 1.0)
    
    # For each face of outer hex, compute the max projection of all inner vertices
    max_proj = 0.0
    for i in range(6):
        v1 = unit_outer_verts[i]
        v2 = unit_outer_verts[(i+1) % 6]
        # Edge vector
        edge = v2 - v1
        # Outward normal (perpendicular to edge, pointing outward)
        normal = np.array([-edge[1], edge[0]])
        normal = normal / np.linalg.norm(normal)
        # Check it points outward: should be in same direction as face midpoint from center
        face_mid = (v1 + v2) / 2
        if np.dot(normal, face_mid) < 0:
            normal = -normal
        
        # Max projection of all inner vertices in this normal direction
        proj = 0.0
        for verts in all_verts:
            for v in verts:
                p = np.dot(v, normal)
                if p > proj:
                    proj = p
        
        # The unit outer hex has support in this normal direction:
        unit_support = np.dot(face_mid, normal)
        # Actually, for a regular hexagon, all faces are at the same distance from center
        # = apothem = sqrt(3)/2 for side length 1
        # So required S >= proj / (sqrt(3)/2)
        
        if unit_support > 1e-12:
            required_S = proj / unit_support
            if required_S > max_proj:
                max_proj = required_S
    
    return max_proj

def construct():
    start_time = time.time()
    
    # Initial packing: 1 center + 6 ring + 4 outer
    # Center at origin
    # Ring 1: 6 hexagons at distance sqrt(3) from center
    # Ring 2: 4 hexagons at distance 3 from center, at angles 30, 90, 150, 210, 270, 330 minus 2
    
    centers = np.array([
        [0.0, 0.0],  # center
        [np.sqrt(3), 0.0],
        [np.sqrt(3)/2, 1.5],
        [-np.sqrt(3)/2, 1.5],
        [-np.sqrt(3), 0.0],
        [-np.sqrt(3)/2, -1.5],
        [np.sqrt(3)/2, -1.5],
        [1.5, np.sqrt(3)/2],
        [0.0, 3.0],
        [-1.5, np.sqrt(3)/2],
        [0.0, -3.0],
    ])
    
    # Initial angles: all 30 degrees (flat-top)
    angles = np.full(11, 30.0)
    
    # Optimize angles of the 4 outer hexagons (indices 7,8,9,10)
    # to minimize the outer side length
    
    def objective(x):
        if time.time() - start_time > DEADLINE - 5:
            return 1e10
        test_angles = angles.copy()
        test_angles[7] = x[0]
        test_angles[8] = x[1]
        test_angles[9] = x[2]
        test_angles[10] = x[3]
        return _compute_outer_side(centers, test_angles)
    
    def constraint_check(x):
        """Check that inner hexagons don't overlap."""
        if time.time() - start_time > DEADLINE - 5:
            return False
        test_angles = angles.copy()
        test_angles[7] = x[0]
        test_angles[8] = x[1]
        test_angles[9] = x[2]
        test_angles[10] = x[3]
        
        verts_list = []
        for i in range(11):
            verts_list.append(_hex_vertices(centers[i,0], centers[i,1], test_angles[i], 1.0))
        
        for i in range(11):
            for j in range(i+1, 11):
                if _polygons_overlap(verts_list[i], verts_list[j]):
                    return False
        return True
    
    # Try optimization
    best_side = _compute_outer_side(centers, angles)
    best_angles = angles.copy()
    
    if time.time() - start_time < DEADLINE - 20:
        # Use Nelder-Mead with constraint checking
        from scipy.optimize import minimize
        
        def obj_with_penalty(x):
            if time.time() - start_time > DEADLINE - 5:
                return 1e10
            if not constraint_check(x):
                return _compute_outer_side(centers, np.concatenate([angles[:7], x])) + 10.0
            return _compute_outer_side(centers, np.concatenate([angles[:7], x]))
        
        x0 = np.array([30.0, 30.0, 30.0, 30.0])
        
        try:
            result = minimize(obj_with_penalty, x0, method='Nelder-Mead',
                             options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
            if result.fun < best_side and constraint_check(result.x):
                best_angles[7:11] = result.x
                best_side = result.fun
        except Exception:
            pass
    
    # Also try optimizing the ring-2 positions slightly
    if time.time() - start_time < DEADLINE - 15:
        # Try small radial adjustments for the 4 outer hexagons
        def obj2(x):
            if time.time() - start_time > DEADLINE - 5:
                return 1e10
            test_centers = centers.copy()
            # Adjust radial positions of outer 4
            for idx, orig_idx in enumerate([7, 8, 9, 10]):
                orig = centers[orig_idx]
                r = np.linalg.norm(orig)
                theta = np.arctan2(orig[1], orig[0])
                new_r = r + x[idx]
                test_centers[orig_idx] = [new_r * np.cos(theta), new_r * np.sin(theta)]
            
            if not constraint_check_full(test_centers, best_angles):
                return _compute_outer_side(test_centers, best_angles) + 10.0
            return _compute_outer_side(test_centers, best_angles)
        
        def constraint_check_full(test_centers, test_angles):
            verts_list = []
            for i in range(11):
                verts_list.append(_hex_vertices(test_centers[i,0], test_centers[i,1], test_angles[i], 1.0))
            for i in range(11):
                for j in range(i+1, 11):
                    if _polygons_overlap(verts_list[i], verts_list[j]):
                        return False
            return True
        
        try:
            result2 = minimize(obj2, np.zeros(4), method='Nelder-Mead',
                             options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-7})
            if result2.fun < best_side:
                # Update centers
                for idx, orig_idx in enumerate([7, 8, 9, 10]):
                    orig = centers[orig_idx]
                    r = np.linalg.norm(orig)
                    theta = np.arctan2(orig[1], orig[0])
                    new_r = r + result2.x[idx]
                    centers[orig_idx] = [new_r * np.cos(theta), new_r * np.sin(theta)]
                best_side = result2.fun
        except Exception:
            pass
    
    # Build final result
    inner = []
    for i in range(11):
        inner.append([float(centers[i,0]), float(centers[i,1]), float(best_angles[i])])
    
    outer_center = (0.0, 0.0)
    outer_angle = 30.0
    side = best_side + 1e-9
    
    # Verify
    if time.time() - start_time < DEADLINE - 5:
        if not _verify_packing(inner, outer_center, side, outer_angle):
            side += 0.05
            if not _verify_packing(inner, outer_center, side, outer_angle):
                side += 0.2
    
    return inner, outer_center, float(side), outer_angle

