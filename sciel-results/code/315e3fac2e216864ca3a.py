"""Packing 12 unit hexagons in a regular hexagon: smooth objective + SA + Nelder-Mead."""
import numpy as np
import time
import math

N = 12
DEADLINE = 145.0  # safety margin before 150s kill

def _hex_verts(cx, cy, angle_deg, side=1.0):
    """Vertices of a regular hexagon."""
    a = np.radians(angle_deg)
    base_angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180
    angles = base_angles + a
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def _all_inner_verts(inner):
    """Get all vertices of all inner hexagons."""
    verts = []
    for (cx, cy, ang) in inner:
        verts.append(_hex_verts(cx, cy, ang))
    return np.vstack(verts)

def _min_dist_between_hexes(v1, v2):
    """
    Compute the minimum distance between two convex hexagons.
    If they overlap, return negative value (penalty).
    Uses edge-to-edge and vertex-to-edge distances.
    """
    min_dist = np.inf
    n1, n2 = len(v1), len(v2)
    
    # Check all edges of poly1 vs all vertices of poly2, and vice versa
    for i in range(n1):
        p1 = v1[i]
        p2 = v1[(i + 1) % n1]
        edge = p2 - p1
        edge_len = np.linalg.norm(edge)
        if edge_len < 1e-12:
            continue
        edge_dir = edge / edge_len
        # Project each vertex of poly2 onto this edge
        for j in range(n2):
            vp = v2[j]
            t = np.dot(vp - p1, edge_dir)
            if t < 0 or t > edge_len:
                # Closest point is endpoint
                d1 = np.linalg.norm(vp - p1)
                d2 = np.linalg.norm(vp - p2)
                d = min(d1, d2)
            else:
                closest = p1 + t * edge_dir
                d = np.linalg.norm(vp - closest)
            min_dist = min(min_dist, d)
    
    for i in range(n2):
        p1 = v2[i]
        p2 = v2[(i + 1) % n2]
        edge = p2 - p1
        edge_len = np.linalg.norm(edge)
        if edge_len < 1e-12:
            continue
        edge_dir = edge / edge_len
        for j in range(n1):
            vp = v1[j]
            t = np.dot(vp - p1, edge_dir)
            if t < 0 or t > edge_len:
                d1 = np.linalg.norm(vp - p1)
                d2 = np.linalg.norm(vp - p2)
                d = min(d1, d2)
            else:
                closest = p1 + t * edge_dir
                d = np.linalg.norm(vp - closest)
            min_dist = min(min_dist, d)
    
    return min_dist

def _max_distance_to_hex_boundary(pt, outer_center, outer_side, outer_angle):
    """
    Compute how far a point is from the boundary of the outer hexagon.
    Positive means inside, negative means outside.
    The outer hexagon has circumradius = outer_side.
    For a point at distance d from center in direction phi, the boundary distance is
    R * cos(delta) where delta is the angular distance to the nearest vertex direction.
    """
    ox, oy = outer_center
    dx = pt[0] - ox
    dy = pt[1] - oy
    d = math.sqrt(dx*dx + dy*dy)
    if d < 1e-12:
        return outer_side * math.sqrt(3) / 2  # inradius
    
    phi = math.atan2(dy, dx)
    a = math.radians(outer_angle)
    
    # Vertex angles
    vertex_angles = [a + k * math.pi / 3 for k in range(6)]
    
    # Find angular distance to nearest vertex
    min_delta = np.inf
    for va in vertex_angles:
        delta = abs(phi - va)
        # Normalize to [0, pi]
        delta = delta % (2 * math.pi)
        if delta > math.pi:
            delta = 2 * math.pi - delta
        min_delta = min(min_delta, delta)
    
    # The boundary in direction phi is at distance R * cos(min_delta) if min_delta <= pi/6
    # Actually for a regular hexagon with circumradius R, the radial distance to boundary
    # in direction phi is R / cos(delta_from_nearest_vertex) ... no.
    # Let me use the half-plane formulation.
    # Edge normals are at angles a + (k + 0.5) * pi/3
    # The half-plane constraint: dot(p - center, n_k) <= R * cos(pi/6) = R * sqrt(3)/2
    
    min_margin = np.inf
    for k in range(6):
        na = a + (k + 0.5) * math.pi / 3
        nx = math.cos(na)
        ny = math.sin(na)
        dot = dx * nx + dy * ny
        margin = outer_side * math.sqrt(3) / 2 - dot
        min_margin = min(min_margin, margin)
    
    return min_margin

def _objective(params, outer_center, outer_angle=0.0):
    """
    Objective: minimize outer side length with penalty for overlaps and containment violations.
    params = [x0,y0,a0, x1,y1,a1, ..., x11,y11,a11]
    """
    inner = []
    for i in range(N):
        x = params[3*i]
        y = params[3*i+1]
        a = params[3*i+2]
        inner.append((x, y, a))
    
    # Compute outer side as max of required distances
    ox, oy = outer_center
    all_verts = _all_inner_verts(inner)
    
    # For each vertex, compute the required circumradius
    dx = all_verts[:, 0] - ox
    dy = all_verts[:, 1] - oy
    
    a_rad = math.radians(outer_angle)
    normals = []
    for k in range(6):
        na = a_rad + (k + 0.5) * math.pi / 3
        normals.append([math.cos(na), math.sin(na)])
    normals = np.array(normals)
    
    proj = np.column_stack([dx, dy]) @ normals.T  # (M, 6)
    max_proj = proj.max(axis=1)
    R_req = (max_proj / (math.sqrt(3) / 2)).max()
    outer_side = max(R_req, 0.0)
    
    # Check containment: all vertices must be inside
    containment_penalty = 0.0
    for v in all_verts:
        margin = _max_distance_to_hex_boundary(v, (ox, oy), outer_side, outer_angle)
        if margin < 0:
            containment_penalty += (-margin) ** 2
    
    # Check disjointness: penalize overlaps
    overlap_penalty = 0.0
    verts_list = []
    for (cx, cy, ang) in inner:
        verts_list.append(_hex_verts(cx, cy, ang))
    
    for i in range(N):
        for j in range(i + 1, N):
            # Check if hexagons overlap
            # Use SAT-like check: if no separating axis, they overlap
            # For penalty, compute penetration depth approximately
            # Simple: check if any vertex of one is inside the other
            v1 = verts_list[i]
            v2 = verts_list[j]
            
            # Check if any vertex of v1 is inside v2
            for vp in v1:
                # Point in convex polygon test
                inside = True
                n2 = len(v2)
                signs = []
                for k in range(n2):
                    p1 = v2[k]
                    p2 = v2[(k + 1) % n2]
                    cross = (p2[0] - p1[0]) * (vp[1] - p1[1]) - (p2[1] - p1[1]) * (vp[0] - p1[0])
                    signs.append(cross)
                if not (all(s >= -1e-9 for s in signs) or all(s <= 1e-9 for s in signs)):
                    inside = False
                if inside:
                    overlap_penalty += 1.0
                    break
            
            # Also check edge intersections roughly by min distance
            min_d = _min_dist_between_hexes(v1, v2)
            if min_d < 1e-6:
                overlap_penalty += (1e-6 - min_d) ** 2
    
    return outer_side + 100.0 * overlap_penalty + 10.0 * containment_penalty

def _construct():
    t0 = time.time()
    
    # Initial configuration: honeycomb-like
    # 12 hexagons: center + 6 around + 5 in a ring
    # For unit hexagons, center-to-center distance for touching is 2 * apothem = sqrt(3)
    d = math.sqrt(3)  # distance between centers of touching hexagons
    
    init_params = np.zeros(3 * N)
    positions = []
    # Center
    positions.append((0, 0, 0))
    # 6 around center
    for k in range(6):
        angle = k * math.pi / 3
        x = d * math.cos(angle)
        y = d * math.sin(angle)
        positions.append((x, y, 0))
    # 5 more: extend the ring
    # Place them at distance 2*d in some directions
    for k in range(5):
        angle = k * 2 * math.pi / 5
        x = 2 * d * math.cos(angle)
        y = 2 * d * math.sin(angle)
        positions.append((x, y, 0))
    
    for i, (x, y, a) in enumerate(positions):
        init_params[3*i] = x
        init_params[3*i+1] = y
        init_params[3*i+2] = a
    
    # Outer center: centroid
    cx = np.mean([init_params[3*i] for i in range(N)])
    cy = np.mean([init_params[3*i+1] for i in range(N)])
    outer_center = (float(cx), float(cy))
    
    # Simulated annealing
    best_params = init_params.copy()
    best_val = _objective(best_params, outer_center, 0.0)
    current_params = init_params.copy()
    current_val = best_val
    
    T_start = 10.0
    T_end = 0.01
    n_iterations = 5000
    
    for it in range(n_iterations):
        if time.time() - t0 > DEADLINE - 20:
            break
        
        # Cooling
        progress = it / n_iterations
        T = T_start * (T_end / T_start) ** progress
        
        # Generate neighbor: perturb 1-2 hexagons
        n_perturb = np.random.randint(1, 3)
        new_params = current_params.copy()
        for _ in range(n_perturb):
            idx = np.random.randint(0, N)
            new_params[3*idx] += np.random.normal(0, 0.5)
            new_params[3*idx+1] += np.random.normal(0, 0.5)
            new_params[3*idx+2] += np.random.normal(0, 15)
        
        new_val = _objective(new_params, outer_center, 0.0)
        
        # Accept if better, or with probability exp(-delta/T)
        if new_val < current_val:
            current_params = new_params
            current_val = new_val
            if current_val < best_val:
                best_val = current_val
                best_params = current_params.copy()
        else:
            delta = new_val - current_val
            if delta < 100:  # Don't accept huge penalties
                prob = math.exp(-delta / T)
                if np.random.random() < prob:
                    current_params = new_params
                    current_val = new_val
    
    # Local polish with Nelder-Mead
    if time.time() - t0 < DEADLINE - 10:
        from scipy.optimize import minimize
        try:
            res = minimize(lambda p: _objective(p, outer_center, 0.0),
                          best_params, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except:
            pass
    
    # Build result
    inner = []
    for i in range(N):
        inner.append([float(best_params[3*i]), float(best_params[3*i+1]), float(best_params[3*i+2])])
    
    cx = float(np.mean([p[0] for p in inner]))
    cy = float(np.mean([p[1] for p in inner]))
    side = float(best_val)
    
    # Final verification
    all_verts = _all_inner_verts(inner)
    # Recompute side
    dx = all_verts[:, 0] - cx
    dy = all_verts[:, 1] - cy
    a_rad = 0.0
    normals = []
    for k in range(6):
        na = a_rad + (k + 0.5) * math.pi / 3
        normals.append([math.cos(na), math.sin(na)])
    normals = np.array(normals)
    proj = np.column_stack([dx, dy]) @ normals.T
    max_proj = proj.max(axis=1)
    R_req = (max_proj / (math.sqrt(3) / 2)).max()
    side = max(float(R_req), side)
    
    # Ensure validity: if overlaps exist, scale up
    verts_list = [_hex_verts(cx, cy, ang) for (cx, cy, ang) in inner]
    has_overlap = False
    for i in range(N):
        for j in range(i + 1, N):
            min_d = _min_dist_between_hexes(verts_list[i], verts_list[j])
            if min_d < 1e-6:
                has_overlap = True
                break
        if has_overlap:
            break
    
    if has_overlap:
        # Fallback: spread out
        inner = []
        for k in range(N):
            angle = k * 2 * math.pi / N
            r = 2.5
            inner.append([r * math.cos(angle), r * math.sin(angle), 0.0])
        cx, cy = 0.0, 0.0
        all_verts = _all_inner_verts(inner)
        dx = all_verts[:, 0] - cx
        dy = all_verts[:, 1] - cy
        proj = np.column_stack([dx, dy]) @ normals.T
        max_proj = proj.max(axis=1)
        R_req = (max_proj / (math.sqrt(3) / 2)).max()
        side = float(R_req) + 0.1
    
    return inner, (cx, cy), float(side), 0.0

def construct():
    return _construct()
