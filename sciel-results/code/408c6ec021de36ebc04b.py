import numpy as np
import time

N = 11

def hex_vertices(cx, cy, s, angle_deg):
    """Vertices of regular hexagon centered at (cx,cy), side s, rotated angle_deg."""
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * np.pi / 3
        verts.append([cx + s * np.cos(theta), cy + s * np.sin(theta)])
    return np.array(verts)

def point_in_poly(px, py, poly):
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

def seg_intersect(p1, p2, p3, p4):
    def ccw(a, b, c):
        return (c[1]-a[1])*(b[0]-a[0]) > (b[1]-a[1])*(c[0]-a[0])
    return ccw(p1,p3,p4) != ccw(p2,p3,p4) and ccw(p1,p2,p3) != ccw(p1,p2,p4)

def hexagons_disjoint(v1, v2, s, eps=1e-9):
    """Check if two hexagons (vertex arrays) are disjoint (no interior intersection)."""
    for v in v1:
        if point_in_poly(v[0], v[1], v2):
            return False
    for v in v2:
        if point_in_poly(v[0], v[1], v1):
            return False
    for i in range(6):
        p1 = v1[i]
        p2 = v1[(i+1)%6]
        for j in range(6):
            p3 = v2[j]
            p4 = v2[(j+1)%6]
            if seg_intersect(p1, p2, p3, p4):
                return False
    return True

def compute_side_length(all_verts, center, angle_deg):
    """Compute minimal side length of outer hexagon with given center and angle to contain all points."""
    sqrt3 = np.sqrt(3)
    a = np.radians(angle_deg)
    normals = np.array([[np.cos(a + np.pi/6 + i * np.pi / 3), 
                         np.sin(a + np.pi/6 + i * np.pi / 3)] for i in range(6)])
    
    dx = all_verts[:, 0] - center[0]
    dy = all_verts[:, 1] - center[1]
    
    max_side = 0.0
    for i in range(6):
        proj = dx * normals[i, 0] + dy * normals[i, 1]
        max_proj = proj.max()
        min_proj = proj.min()
        required_s = max(max_proj, -min_proj) / (sqrt3 / 2)
        max_side = max(max_side, required_s)
    return max_side

def find_optimal_center(all_verts, angle_deg):
    """Find optimal center for given angle that minimizes the containing hexagon side length."""
    sqrt3 = np.sqrt(3)
    a = np.radians(angle_deg)
    normals = np.array([[np.cos(a + np.pi/6 + i * np.pi / 3), 
                         np.sin(a + np.pi/6 + i * np.pi / 3)] for i in range(6)])
    
    # Candidate centers: mean, and midpoints of projection extremes for each direction
    candidates = []
    
    # Mean center
    candidates.append((all_verts[:, 0].mean(), all_verts[:, 1].mean()))
    
    # For each normal direction, the optimal center along that axis is the midpoint
    # of the min and max projections. We can use this to generate candidate centers.
    for i in range(6):
        proj = all_verts[:, 0] * normals[i, 0] + all_verts[:, 1] * normals[i, 1]
        min_proj = proj.min()
        max_proj = proj.max()
        mid_proj = (min_proj + max_proj) / 2
        
        # The center should be such that the projection of all points is centered at mid_proj
        # center = point_on_line - mid_proj * normal (approximately)
        # Use the first point as reference
        ref_x, ref_y = all_verts[0, 0], all_verts[0, 1]
        ref_proj = ref_x * normals[i, 0] + ref_y * normals[i, 1]
        offset = mid_proj - ref_proj
        cx = ref_x - offset * normals[i, 0]
        cy = ref_y - offset * normals[i, 1]
        candidates.append((cx, cy))
    
    # Also try centroids of vertex subsets
    candidates.append((all_verts[:, 0].mean(), all_verts[:, 1].mean()))
    
    # Evaluate all candidates
    best_side = float('inf')
    best_center = candidates[0]
    
    for cx, cy in candidates:
        side = compute_side_length(all_verts, (cx, cy), angle_deg)
        if side < best_side:
            best_side = side
            best_center = (cx, cy)
    
    # Refine with local search around best center
    cx, cy = best_center
    step = 0.01
    for _ in range(20):
        improved = False
        for dx in [-step, 0, step]:
            for dy in [-step, 0, step]:
                if dx == 0 and dy == 0:
                    continue
                new_side = compute_side_length(all_verts, (cx + dx, cy + dy), angle_deg)
                if new_side < best_side - 1e-12:
                    best_side = new_side
                    cx += dx
                    cy += dy
                    improved = True
        if not improved:
            break
        step *= 0.5
    
    return (cx, cy), best_side

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    sqrt3 = np.sqrt(3)
    
    # 11 hexagons in hex pattern:
    hex_positions_axial = [
        (0, 0),
        (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
        (2, 0), (2, -1), (1, -2), (0, -2),
    ]
    
    centers = []
    for q, r in hex_positions_axial:
        x = 1.5 * q
        y = sqrt3 * (r + 0.5 * q)
        centers.append((x, y))
    
    centers = np.array(centers)
    
    # Compute all vertices
    inner_verts = [hex_vertices(cx, cy, 1.0, 0.0) for cx, cy in centers]
    all_verts = np.vstack(inner_verts)
    
    # Verify inner hexagons are disjoint
    valid = True
    for i in range(N):
        for j in range(i + 1, N):
            if not hexagons_disjoint(inner_verts[i], inner_verts[j], 1.0):
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback: scale positions slightly
        scale = 1.001
        centers = centers * scale
        inner_verts = [hex_vertices(cx, cy, 1.0, 0.0) for cx, cy in centers]
        all_verts = np.vstack(inner_verts)
        for i in range(N):
            for j in range(i + 1, N):
                if not hexagons_disjoint(inner_verts[i], inner_verts[j], 1.0):
                    break
    
    # Coarse search over outer hexagon angles
    best_side = float('inf')
    best_center = (all_verts[:, 0].mean(), all_verts[:, 1].mean())
    best_angle = 0.0
    
    for outer_angle in np.linspace(0, 60, 121):
        if time.time() > deadline:
            break
        center, side = find_optimal_center(all_verts, outer_angle)
        if side < best_side:
            best_side = side
            best_center = center
            best_angle = outer_angle
    
    # Fine search around best angle
    if best_side < float('inf') and time.time() < deadline:
        for outer_angle in np.linspace(best_angle - 2.0, best_angle + 2.0, 401):
            if time.time() > deadline:
                break
            center, side = find_optimal_center(all_verts, outer_angle)
            if side < best_side:
                best_side = side
                best_center = center
                best_angle = outer_angle
    
    # Final high-precision polish on center for the best angle
    if time.time() < deadline:
        best_center, best_side = find_optimal_center(all_verts, best_angle)
    
    # Build inner list
    inner = [[float(cx), float(cy), float(0.0)] for cx, cy in centers]
    
    # Final verification: ensure all inner hex vertices are inside outer hex
    # Add small safety margin
    best_side *= 1.0001
    all_check_verts = np.vstack(inner_verts)
    
    a = np.radians(best_angle)
    normals = np.array([[np.cos(a + np.pi/6 + i * np.pi / 3), 
                         np.sin(a + np.pi/6 + i * np.pi / 3)] for i in range(6)])
    
    dx = all_check_verts[:, 0] - best_center[0]
    dy = all_check_verts[:, 1] - best_center[1]
    
    max_violation = 0
    for i in range(6):
        proj = dx * normals[i, 0] + dy * normals[i, 1]
        max_violation = max(max_violation, proj.max() - best_side * sqrt3 / 2)
        max_violation = max(max_violation, -proj.min() - best_side * sqrt3 / 2)
    
    if max_violation > 1e-9:
        best_side += max_violation / (sqrt3 / 2) * 1.0001
    
    return inner, (float(best_center[0]), float(best_center[1])), float(best_side), float(best_angle)
