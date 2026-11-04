import numpy as np
import time
from itertools import combinations

N = 12
TIME_LIMIT = 145.0

def _hex_vertices(center, side, angle_deg):
    x, y = center
    theta = np.deg2rad(angle_deg)
    angles = theta + np.arange(6) * (np.pi / 3)
    return np.array([
        [x + side * np.cos(a), y + side * np.sin(a)] 
        for a in angles
    ])

def _seg_seg_dist(p1, p2, p3, p4):
    d1 = p2 - p1
    d2 = p4 - p3
    r = p1 - p3
    a = np.dot(d1, d1)
    e = np.dot(d2, d2)
    f = np.dot(d2, r)
    
    if a < 1e-12 and e < 1e-12:
        return np.linalg.norm(r)
    elif a < 1e-12:
        s = 0.0
        t = f / e
        t = max(0.0, min(1.0, t))
        return np.linalg.norm(r + t * d2)
    elif e < 1e-12:
        s = -np.dot(d1, r) / a
        s = max(0.0, min(1.0, s))
        return np.linalg.norm(s * d1 - r)
        
    c = np.dot(d1, r)
    b = np.dot(d1, d2)
    denom = a * e - b * b
    
    if denom < 1e-12:
        s = 0.0
        t = f / e
        t = max(0.0, min(1.0, t))
        s = max(0.0, min(1.0, (b * t - c) / a))
    else:
        s = (b * f - c * e) / denom
        if s < 0.0 or s > 1.0:
            s = 0.0
            t = f / e
            t = max(0.0, min(1.0, t))
        else:
            t = (b * s + f) / e
            t = max(0.0, min(1.0, t))
            if t < 0.0:
                t = 0.0
                s = max(0.0, min(1.0, -c / a))
            elif t > 1.0:
                t = 1.0
                s = max(0.0, min(1.0, (b - c) / a))
                
    return np.linalg.norm(s * d1 - r - t * d2)

def _polygons_overlap(v1, v2, eps=1e-9):
    n1, n2 = len(v1), len(v2)
    for i in range(n1):
        p1 = v1[i]
        p2 = v1[(i + 1) % n1]
        for j in range(n2):
            q1 = v2[j]
            q2 = v2[(j + 1) % n2]
            if _seg_seg_dist(p1, p2, q1, q2) < eps:
                return True
    return False

def _check_validity(inner_hexes, outer_side, outer_center, outer_angle, eps=1e-7):
    outer_verts = _hex_vertices(outer_center, outer_side, outer_angle)
    inner_verts = [_hex_vertices(h[0:2], 1.0, h[2]) for h in inner_hexes]
    
    for i in range(len(inner_hexes)):
        for j in range(i + 1, len(inner_hexes)):
            if _polygons_overlap(inner_verts[i], inner_verts[j], eps=1e-9):
                return False, "overlap"
                
        is_inside = True
        for p in inner_verts[i]:
            inside = True
            for k in range(6):
                a = outer_verts[k]
                b = outer_verts[(k+1)%6]
                cross_ab = (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0])
                if cross_ab < -eps:
                    inside = False
                    break
            if not inside:
                is_inside = False
                break
        if not is_inside:
            return False, "outside"
            
    return True, "valid"

def _compute_required_side(inner_hexes, outer_angle):
    max_proj = 0.0
    for h in inner_hexes:
        c = h[0:2]
        verts = _hex_vertices(c, 1.0, h[2])
        for k in range(6):
            angle = np.deg2rad(outer_angle + k * 60)
            normal = np.array([np.cos(angle), np.sin(angle)])
            for v in verts:
                proj = np.dot(v, normal)
                if proj > max_proj:
                    max_proj = proj
    return max_proj * 2.0 / np.sqrt(3.0)

def _generate_lattice_points():
    pts = []
    # Generate a large grid of potential centers
    # Spacing sqrt(3) allows side-sharing
    spacing = np.sqrt(3)
    for i in range(-5, 6):
        for j in range(-5, 6):
            x = i * spacing + (j % 2) * spacing / 2
            y = j * spacing * np.sqrt(3) / 2
            pts.append([x, y])
    return np.array(pts)

def _find_clusters():
    lattice_pts = _generate_lattice_points()
    clusters = []
    
    # We want to find sets of 12 points that are mutually non-overlapping.
    # Since they are on a lattice with spacing sqrt(3), any two distinct points are at least sqrt(3) apart.
    # However, we must ensure that the hexagons don't overlap.
    # For hexagons on this lattice, if they are at distance sqrt(3), they share a side.
    # If they are closer, they overlap.
    # Wait, the lattice spacing is sqrt(3). The distance between centers of adjacent hexagons in a honeycomb is sqrt(3).
    # So any two distinct points on this lattice are at distance >= sqrt(3).
    # Thus, any subset of this lattice is a valid non-overlapping configuration!
    
    # We just need to pick 12 points that form a compact cluster.
    # We'll generate many random subsets and keep the ones with the smallest bounding box area or diameter.
    
    indices = np.arange(len(lattice_pts))
    
    # Heuristic: Start with the origin and add nearest neighbors greedily?
    # Or random sampling.
    
    best_clusters = []
    
    # Try a few random subsets
    for _ in range(200):
        subset_idx = np.random.choice(indices, N, replace=False)
        pts = lattice_pts[subset_idx]
        
        # Compute diameter
        max_d = 0
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(pts[i] - pts[j])
                if d > max_d:
                    max_d = d
        
        # Compute bounding box area (rough proxy)
        min_x, min_y = np.min(pts, axis=0)
        max_x, max_y = np.max(pts, axis=0)
        area = (max_x - min_x) * (max_y - min_y)
        
        # We want compact clusters. Low area and low diameter.
        score = area + 0.1 * max_d**2
        best_clusters.append((score, pts))
        
    # Sort by score
    best_clusters.sort(key=lambda x: x[0])
    
    # Keep unique top clusters
    unique_clusters = []
    for score, pts in best_clusters:
        # Check if similar to existing
        is_unique = True
        for _, existing_pts in unique_clusters:
            # Simple check: are the sets of points the same?
            # Since points are from a fixed lattice, we can compare sorted tuples
            pts_set = set(map(tuple, np.round(pts, 5)))
            existing_set = set(map(tuple, np.round(existing_pts, 5)))
            if pts_set == existing_set:
                is_unique = False
                break
        if is_unique:
            unique_clusters.append((score, pts))
            if len(unique_clusters) >= 20:
                break
                
    return [pts for _, pts in unique_clusters]

def construct():
    start_time = time.time()
    
    # 1. Generate candidate topologies
    clusters = _find_clusters()
    
    best_side = float('inf')
    best_inner = None
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    # 2. For each topology, optimize the global rotation
    # The inner hexagons can also rotate individually, but let's start with them all aligned with the lattice (angle 0 relative to lattice, which is fixed).
    # Actually, the lattice is defined in the global frame.
    # If we rotate the whole cluster, we rotate the positions and the hexagon orientations.
    # Let's assume inner hexagons are aligned with the global frame (angle 0) initially, and we rotate the cluster positions?
    # No, if we rotate the cluster, the hexagons should rotate too to maintain the "honeycomb" fit?
    # In a honeycomb, the hexagons are aligned. If we rotate the whole packing, we rotate all hexagons by the same angle.
    # So, for a given cluster of centers (in local frame), we define the packing as:
    # Center_i = R(theta) * local_center_i
    # Angle_i = theta
    
    # We want to minimize the side length of the outer hexagon containing these rotated hexagons.
    # The outer hexagon can also rotate. Let's fix outer angle to 0 for simplicity and rotate the inner cluster?
    # Or fix inner cluster and rotate outer?
    # Due to symmetry, rotating the inner cluster by theta is equivalent to rotating the outer hexagon by -theta.
    # So we can fix inner angles to 0, and search for the optimal outer angle?
    # Wait, if we fix inner angles to 0, we are restricting the solution space.
    # The optimal solution might have inner hexagons at different angles.
    # However, for a "honeycomb" cluster, it is natural for them to be aligned.
    # Let's try:
    # 1. Fix inner angles to 0.
    # 2. Search for optimal outer angle in [0, 60).
    
    for pts in clusters:
        if time.time() - start_time > TIME_LIMIT - 5:
            break
            
        # Define the inner hexagons for a given outer angle
        # We will search for the outer angle that minimizes the required side length.
        # Required side length S(theta) is determined by the max projection of vertices onto normals.
        # This function is periodic with period 60 degrees.
        
        # Golden section search for theta in [0, 60]
        phi = (np.sqrt(5) - 1) / 2
        a = 0.0
        b = 60.0
        
        def side_len(theta_deg):
            # theta is the rotation of the outer hexagon
            # Inner hexagons are at fixed positions 'pts' with angle 0.
            # Wait, if we rotate the outer hexagon, the normals change.
            # The inner hexagons are fixed.
            # This is equivalent to rotating the inner hexagons by -theta and fixing the outer hexagon.
            # Let's just compute the side length for a given outer angle.
            
            max_proj = 0.0
            # Outer normals
            normals = []
            for k in range(6):
                angle = np.deg2rad(theta_deg + k * 60)
                normals.append(np.array([np.cos(angle), np.sin(angle)]))
            
            for p in pts:
                # Vertices of inner hexagon at p, angle 0
                verts = _hex_vertices(p, 1.0, 0.0)
                for n in normals:
                    for v in verts:
                        proj = np.dot(v, n)
                        if proj > max_proj:
                            max_proj = proj
            return max_proj * 2.0 / np.sqrt(3.0)
            
        # Golden section
        c = b - phi * (b - a)
        d = a + phi * (b - a)
        fc = side_len(c)
        fd = side_len(d)
        
        for _ in range(50):
            if time.time() - start_time > TIME_LIMIT - 2:
                break
            if fc < fd:
                b = d
                d = c
                fd = fc
                c = b - phi * (b - a)
                fc = side_len(c)
            else:
                a = c
                c = d
                fc = fd
                d = a + phi * (b - a)
                fd = side_len(d)
                
        theta_opt = (a + b) / 2.0
        side_opt = side_len(theta_opt)
        
        # Verify validity
        # Inner hexagons are at 'pts' with angle 0.
        inner_hexes = [[p[0], p[1], 0.0] for p in pts]
        
        # Check if they are inside the outer hexagon of side side_opt, angle theta_opt, center (0,0)
        # And if they are disjoint.
        is_valid, reason = _check_validity(inner_hexes, side_opt, (0.0, 0.0), theta_opt)
        
        if is_valid and side_opt < best_side:
            best_side = side_opt
            best_inner = inner_hexes
            best_outer_center = (0.0, 0.0)
            best_outer_angle = theta_opt
            
    if best_inner is None:
        # Fallback
        return [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)], (0.0, 0.0), 40.0, 0.0
        
    return (best_inner, best_outer_center, best_side, best_outer_angle)
