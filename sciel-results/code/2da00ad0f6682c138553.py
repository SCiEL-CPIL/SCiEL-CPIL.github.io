import numpy as np
from scipy.optimize import minimize
import time

N = 11
DEADLINE = 110.0  # safety margin before 120s limit

def hexagon_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon."""
    angle_rad = np.radians(angle_deg)
    vertices = []
    for i in range(6):
        a = angle_rad + i * np.pi / 3.0
        x = cx + side * np.cos(a)
        y = cy + side * np.sin(a)
        vertices.append([x, y])
    return np.array(vertices)

def point_in_polygon(px, py, poly):
    """Ray casting point-in-polygon test."""
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

def segments_intersect(p1, p2, p3, p4):
    """Check if segment p1-p2 intersects segment p3-p4 (including touching)."""
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - a[0])
    
    d1 = ccw(p3, p4, p1)
    d2 = ccw(p3, p4, p2)
    d3 = ccw(p1, p2, p3)
    d4 = ccw(p1, p2, p4)
    
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    # Check collinear overlap
    if d1 == 0 and ccw(p1, p2, p3) >= -1e-10 and ccw(p1, p2, p3) <= 1e-10:
        return True
    if d2 == 0 and ccw(p1, p2, p4) >= -1e-10 and ccw(p1, p2, p4) <= 1e-10:
        return True
    if d3 == 0 and ccw(p3, p4, p1) >= -1e-10 and ccw(p3, p4, p1) <= 1e-10:
        return True
    if d4 == 0 and ccw(p3, p4, p2) >= -1e-10 and ccw(p3, p4, p2) <= 1e-10:
        return True
    return False

def hexagons_overlap(v1, v2, gap=0.001):
    """Check if two hexagons (as vertex arrays) overlap or touch within gap."""
    # Check if any vertex of v1 is inside v2 (with gap)
    for pt in v1:
        if point_in_polygon(pt[0], pt[1], v2):
            return True
    for pt in v2:
        if point_in_polygon(pt[0], pt[1], v1):
            return True
    # Check edge intersections
    for i in range(6):
        for j in range(6):
            if segments_intersect(v1[i], v1[(i+1)%6], v2[j], v2[(j+1)%6]):
                return True
    return False

def all_inside_hexagon(inner_verts_list, outer_verts):
    """Check all inner hexagon vertices are inside outer hexagon."""
    for verts in inner_verts_list:
        for pt in verts:
            if not point_in_polygon(pt[0], pt[1], outer_verts):
                return False
    return True

def are_disjoint(inner_verts_list, gap=0.001):
    """Check all pairs of inner hexagons are disjoint."""
    n = len(inner_verts_list)
    for i in range(n):
        for j in range(i+1, n):
            if hexagons_overlap(inner_verts_list[i], inner_verts_list[j], gap):
                return False
    return True

def compute_outer_side(inner_positions, inner_angles, outer_center, outer_angle):
    """Compute minimum outer side length needed to contain all inner hexagons."""
    # Get all vertices of all inner hexagons
    all_verts = []
    for (x, y, ang) in zip(inner_positions[:, 0], inner_positions[:, 1], inner_angles):
        verts = hexagon_vertices(x, y, 1.0, ang)
        all_verts.extend(verts.tolist())
    all_verts = np.array(all_verts)
    
    # For a regular hexagon centered at outer_center with rotation outer_angle,
    # the distance from center to each edge (inradius) must be >= max projection
    # of all points onto the normal directions of the 6 edges.
    angle_rad = np.radians(outer_angle)
    # The 6 edge normals for a hexagon with "flat top" orientation
    # Edge normals point outward at angles: 0, 60, 120, 180, 240, 300 (for pointy-top)
    # For a regular hexagon, the inradius r = side * sqrt(3)/2
    # A point (px, py) relative to center is inside if for all 6 normals n_k:
    #   dot(point - center, n_k) <= r
    # So we need r >= max over all points and all 6 normals of dot(point-center, n_k)
    
    rel = all_verts - np.array(outer_center)
    max_proj = 0.0
    for k in range(6):
        a = angle_rad + k * np.pi / 3.0
        nx = np.cos(a)
        ny = np.sin(a)
        projs = rel[:, 0] * nx + rel[:, 1] * ny
        max_proj = max(max_proj, np.max(projs))
    
    # inradius r = side * sqrt(3)/2, so side = r / (sqrt(3)/2) = 2r/sqrt(3)
    side = 2.0 * max_proj / np.sqrt(3.0)
    return side

def initial_hexagonal_packing():
    """Create initial hexagonal packing for 11 hexagons using a gap factor."""
    positions = []
    angles = []
    
    gap_factor = 1.005
    d1 = np.sqrt(3.0) * gap_factor
    d2 = 2.0 * np.sqrt(3.0) * gap_factor
    
    # Central hexagon
    positions.append([0.0, 0.0])
    angles.append(0.0)
    
    # 6 surrounding
    for i in range(6):
        a = i * np.pi / 3.0
        positions.append([d1 * np.cos(a), d1 * np.sin(a)])
        angles.append(0.0)
        
    # 4 more: Pick 4 from the 12 outer positions of a 19-hex cluster
    # Outer positions are at distance d2, angles i*pi/6
    # We want to minimize the bounding box.
    # Picking angles 0, pi/2, pi, 3pi/2 (0, 90, 180, 270) creates a cross.
    # Picking angles pi/6, pi/2, 5pi/6, 7pi/6 etc?
    # Let's try to pick 4 that are symmetric.
    # Indices 1, 3, 5, 7 (angles 30, 90, 150, 210) or similar.
    # Actually, for a hexagonal cluster, the "corners" are at 0, 60, 120, 180, 240, 300.
    # The "edges" midpoints are at 30, 90, 150, 210, 270, 330.
    # If we pick 4 "edge" midpoints, we get a symmetric shape.
    # Let's pick 30, 90, 150, 210? No, that's not symmetric under 180.
    # Pick 30, 150, 210, 330? 
    # Let's just pick 4 adjacent ones? No, that's elongated.
    # Let's pick 4 that form a square-ish shape.
    # Angles: 45, 135, 225, 315? Hex grid doesn't align well with 45.
    
    # Simplest robust choice: Pick 4 from the 12 outer, spaced out.
    # Indices 0, 3, 6, 9 in the 12 outer ring (angles 0, 90, 180, 270).
    outer_indices = [0, 3, 6, 9]
    for i in outer_indices:
        a = i * np.pi / 6.0
        positions.append([d2 * np.cos(a), d2 * np.sin(a)])
        angles.append(0.0)
        
    return np.array(positions), np.array(angles)

def construct():
    start_time = time.time()
    
    # Initial packing
    positions, angles = initial_hexagonal_packing()
    outer_center = [0.0, 0.0]
    outer_angle = 0.0
    
    # Compute initial outer side
    current_side = compute_outer_side(positions, angles, outer_center, outer_angle)
    best_side = current_side
    best_positions = positions.copy()
    best_angles = angles.copy()
    
    # Verify initial solution
    inner_verts_list = []
    for i in range(N):
        verts = hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
        inner_verts_list.append(verts)
    
    outer_verts = hexagon_vertices(outer_center[0], outer_center[1], current_side, outer_angle)
    
    if not (all_inside_hexagon(inner_verts_list, outer_verts) and are_disjoint(inner_verts_list)):
        # Fallback: spread out more to ensure validity, even if side is larger
        for i in range(N):
            a = i * 2 * np.pi / N
            r = 5.0
            positions[i, 0] = r * np.cos(a)
            positions[i, 1] = r * np.sin(a)
        current_side = compute_outer_side(positions, angles, outer_center, outer_angle)
        best_side = current_side
        best_positions = positions.copy()
        best_angles = angles.copy()

    # Optimization: Minimize outer side by adjusting global rotation of the cluster
    # and outer hexagon orientation.
    # We treat the cluster as rigid to avoid overlap issues during optimization.
    
    def objective_rigid(params):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 5:
            return best_side + 1000
        
        cluster_rot = params[0]
        outer_rot = params[1]
        
        # Rotate positions
        c = np.cos(cluster_rot)
        s = np.sin(cluster_rot)
        rot_mat = np.array([[c, -s], [s, c]])
        new_pos = best_positions @ rot_mat.T
        
        side = compute_outer_side(new_pos, angles, outer_center, outer_rot)
        return side

    x0 = np.array([0.0, 0.0])
    
    try:
        elapsed = time.time() - start_time
        if elapsed < DEADLINE - 30:
            result = minimize(objective_rigid, x0, method='Nelder-Mead', 
                            options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8})
            
            if result.fun < best_side:
                new_cluster_rot = result.x[0]
                new_outer_rot = result.x[1]
                
                c = np.cos(new_cluster_rot)
                s = np.sin(new_cluster_rot)
                rot_mat = np.array([[c, -s], [s, c]])
                new_pos = best_positions @ rot_mat.T
                
                new_side = compute_outer_side(new_pos, angles, outer_center, new_outer_rot)
                
                # Verify
                inner_verts_list = []
                for i in range(N):
                    verts = hexagon_vertices(new_pos[i, 0], new_pos[i, 1], 1.0, angles[i])
                    inner_verts_list.append(verts)
                outer_verts = hexagon_vertices(outer_center[0], outer_center[1], new_side, new_outer_rot)
                
                if all_inside_hexagon(inner_verts_list, outer_verts) and are_disjoint(inner_verts_list):
                    best_side = new_side
                    best_positions = new_pos
                    best_angles = angles.copy()
                    outer_angle = new_outer_rot
    except Exception:
        pass
    
    # Build result
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])
    
    return inner, (outer_center[0], outer_center[1]), float(best_side), float(outer_angle)
