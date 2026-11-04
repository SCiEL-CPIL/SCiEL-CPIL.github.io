import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

def _hex_vertices(cx, cy, angle_deg):
    """Get vertices of a unit regular hexagon centered at (cx,cy) with rotation angle_deg."""
    angle_rad = np.deg2rad(angle_deg)
    vertices = []
    for i in range(6):
        theta = angle_rad + i * np.pi / 3
        x = cx + np.cos(theta)
        y = cy + np.sin(theta)
        vertices.append((x, y))
    return np.array(vertices)

def _get_outer_normals(angle_deg):
    """Get the 6 outward face normals of a regular hexagon with given rotation."""
    angle_rad = np.deg2rad(angle_deg)
    normals = []
    for i in range(6):
        theta = angle_rad + np.pi/6 + i * np.pi / 3
        normals.append((np.cos(theta), np.sin(theta)))
    return np.array(normals)

def _min_enclosing_hex_side(hull_verts, outer_angle_deg, outer_center):
    """
    Compute the side length of the smallest regular hexagon (with given angle and center)
    that contains all hull vertices.
    
    For a regular hexagon with side length S, the distance from center to each face 
    (apothem) is S * sqrt(3) / 2.
    So S = 2 * max_projection / sqrt(3) where max_projection is the max of |v·n| over 
    all vertices v (relative to center) and all 6 face normals n.
    """
    normals = _get_outer_normals(outer_angle_deg)
    rel_verts = hull_verts - outer_center
    dots = rel_verts @ normals.T
    max_proj = np.max(np.abs(dots))
    return 2 * max_proj / np.sqrt(3)

def _compute_cluster_hull(centers, angles):
    """Compute the convex hull of all vertices of all hexagons in the cluster."""
    all_verts = []
    for i, (cx, cy) in enumerate(centers):
        verts = _hex_vertices(cx, cy, angles[i])
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    try:
        hull = ConvexHull(all_verts)
        return all_verts[hull.vertices]
    except Exception:
        return all_verts

def _build_rigid_cluster():
    """
    Build the standard 11-hexagon cluster:
    - 1 center hexagon at origin
    - 6 hexagons in inner ring at distance sqrt(3)
    - 4 hexagons in outer ring at distance 3
    
    All at 30 degrees (flat-top orientation).
    """
    centers = []
    angles = []
    
    # Center
    centers.append([0.0, 0.0])
    angles.append(30.0)
    
    # Inner ring: 6 hexagons
    for k in range(6):
        angle = np.deg2rad(k * 60)
        r = np.sqrt(3)
        centers.append([r * np.cos(angle), r * np.sin(angle)])
        angles.append(30.0)
    
    # Outer ring: 4 hexagons at 30, 90, 150, 210 degrees
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        r = 3.0
        centers.append([r * np.cos(angle), r * np.sin(angle)])
        angles.append(30.0)
    
    return np.array(centers), np.array(angles)

def _compute_S_of_theta(hull_verts, theta_deg, center):
    """Compute enclosing hex side length for a given outer rotation angle."""
    return _min_enclosing_hex_side(hull_verts, theta_deg, center)

def _find_optimal_rotation(hull_verts, center, n_samples=3600):
    """
    Find the optimal rotation angle for the enclosing hexagon.
    The function S(theta) is piecewise-linear and periodic with period 60 degrees.
    We sample densely and then refine with golden section search.
    """
    # Coarse grid search
    thetas = np.linspace(0, 60, n_samples, endpoint=False)
    S_vals = np.array([_compute_S_of_theta(hull_verts, t, center) for t in thetas])
    
    # Find the minimum
    idx = np.argmin(S_vals)
    best_theta = thetas[idx]
    best_S = S_vals[idx]
    
    # Refine with golden section search in a small neighborhood
    # The function is piecewise linear, so the minimum is at a breakpoint
    # Let's search in [thetas[idx-1], thetas[idx+1]]
    if idx > 0:
        a = thetas[idx-1]
    else:
        a = 0.0
    if idx < len(thetas)-1:
        b = thetas[idx+1]
    else:
        b = 60.0
    
    # Golden section search
    gr = (np.sqrt(5) - 1) / 2
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc = _compute_S_of_theta(hull_verts, c, center)
    fd = _compute_S_of_theta(hull_verts, d, center)
    
    for _ in range(80):
        if abs(b - a) < 1e-12:
            break
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - gr * (b - a)
            fc = _compute_S_of_theta(hull_verts, c, center)
        else:
            a = c
            c = d
            fc = fd
            d = a + gr * (b - a)
            fd = _compute_S_of_theta(hull_verts, d, center)
    
    best_theta = (a + b) / 2
    best_S = _compute_S_of_theta(hull_verts, best_theta, center)
    
    return best_theta, best_S

def _hexagon_distance_field(cx1, cy1, a1, cx2, cy2, a2):
    """
    Compute minimum distance between two unit hexagons (positive if disjoint, negative if overlapping).
    """
    v1 = _hex_vertices(cx1, cy1, a1)
    v2 = _hex_vertices(cx2, cy2, a2)
    
    min_dist = float('inf')
    
    # Check each vertex of hex1 against each edge of hex2
    for vi in v1:
        for j in range(6):
            vj = v2[j]
            vj1 = v2[(j+1)%6]
            d = np.linalg.norm(vj1 - vj)
            if d < 1e-12:
                dist = np.linalg.norm(vi - vj)
            else:
                t = np.clip(np.dot(vi - vj, vj1 - vj) / (d * d), 0, 1)
                closest = vj + t * (vj1 - vj)
                dist = np.linalg.norm(vi - closest)
            min_dist = min(min_dist, dist)
    
    # Check each vertex of hex2 against each edge of hex1
    for vi in v2:
        for j in range(6):
            vj = v1[j]
            vj1 = v1[(j+1)%6]
            d = np.linalg.norm(vj1 - vj)
            if d < 1e-12:
                dist = np.linalg.norm(vi - vj)
            else:
                t = np.clip(np.dot(vi - vj, vj1 - vj) / (d * d), 0, 1)
                closest = vj + t * (vj1 - vj)
                dist = np.linalg.norm(vi - closest)
            min_dist = min(min_dist, dist)
    
    # Check edge-edge distances
    for i1 in range(6):
        p1 = v1[i1]
        p2 = v1[(i1+1)%6]
        for i2 in range(6):
            q1 = v2[i2]
            q2 = v2[(i2+1)%6]
            r = p2 - p1
            s = q2 - q1
            r1 = p1 - q1
            a = np.dot(r, r)
            e = np.dot(s, s)
            f = np.dot(s, r1)
            
            if a < 1e-12 and e < 1e-12:
                dist = np.linalg.norm(r1)
            elif a < 1e-12:
                t = np.clip(f / e, 0, 1)
                dist = np.linalg.norm(p1 - (q1 + t * s))
            elif e < 1e-12:
                t = np.clip(-np.dot(r, r1) / a, 0, 1)
                dist = np.linalg.norm(p1 + t * r - q1)
            else:
                c = np.dot(r, r1)
                d = np.dot(r, s)
                denom = a * e - d * d
                if denom < 1e-12:
                    t = 0.0
                    u = np.clip(f / e, 0, 1)
                else:
                    t = np.clip((d * f - c * e) / denom, 0, 1)
                    u = np.clip((a * f - d * c) / denom, 0, 1)
                closest1 = p1 + t * r
                closest2 = q1 + u * s
                dist = np.linalg.norm(closest1 - closest2)
            min_dist = min(min_dist, dist)
    
    return min_dist

def _overlap_penalty(centers, angles, beta=50.0):
    """Smooth penalty for overlapping hexagons."""
    n = len(centers)
    penalty = 0.0
    for i in range(n):
        for j in range(i+1, n):
            dist = _hexagon_distance_field(
                centers[i][0], centers[i][1], angles[i],
                centers[j][0], centers[j][1], angles[j]
            )
            if dist < 0:
                penalty += np.exp(-beta * dist) - 1.0
    return penalty

def _objective_joint(vars, angles):
    """
    Joint objective: optimize inner positions and outer hexagon parameters.
    vars: [cx1, cy1, ..., cx11, cy11, outer_cx, outer_cy, outer_angle_deg]
    """
    n_inner = len(angles)
    centers = np.array([[vars[2*i], vars[2*i+1]] for i in range(n_inner)])
    outer_cx = vars[2*n_inner]
    outer_cy = vars[2*n_inner + 1]
    outer_angle = vars[2*n_inner + 2]
    
    hull = _compute_cluster_hull(centers, angles)
    S = _min_enclosing_hex_side(hull, outer_angle, np.array([outer_cx, outer_cy]))
    
    pen = _overlap_penalty(centers, angles)
    
    return S + 100.0 * pen

def construct():
    start_time = time.time()
    DEADLINE = 145.0
    
    # Build the standard rigid cluster
    centers, angles = _build_rigid_cluster()
    n_inner = len(centers)
    
    # Compute hull of the rigid cluster
    hull = _compute_cluster_hull(centers, angles)
    
    # Initial estimate: outer hexagon centered at centroid of cluster
    centroid = np.mean(centers, axis=0)
    
    # Phase 1: Find optimal rotation angle for the rigid cluster
    best_theta, best_S = _find_optimal_rotation(hull, centroid, n_samples=7200)
    
    # Also try with outer center at origin (symmetric case)
    best_theta_origin, best_S_origin = _find_optimal_rotation(hull, np.array([0.0, 0.0]), n_samples=3600)
    
    # Choose the better one
    if best_S_origin < best_S:
        best_theta = best_theta_origin
        best_S = best_S_origin
        best_center = np.array([0.0, 0.0])
    else:
        best_center = centroid
    
    # Phase 2: Fine-tune outer center position
    def outer_center_obj(vars):
        ocx, ocy = vars
        return _min_enclosing_hex_side(hull, best_theta, np.array([ocx, ocy]))
    
    x0 = np.array([best_center[0], best_center[1]])
    
    try:
        if time.time() < DEADLINE - 30:
            res = minimize(outer_center_obj, x0, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            if res.fun < best_S:
                best_S = res.fun
                best_center = res.x
    except Exception:
        pass
    
    # Phase 3: Fine-tune outer angle with the optimized center
    def angle_obj(theta_deg):
        return _min_enclosing_hex_side(hull, theta_deg, best_center)
    
    try:
        if time.time() < DEADLINE - 25:
            # Golden section search around the current best angle
            a = max(0, best_theta - 5)
            b = min(60, best_theta + 5)
            gr = (np.sqrt(5) - 1) / 2
            c = b - gr * (b - a)
            d = a + gr * (b - a)
            fc = angle_obj(c)
            fd = angle_obj(d)
            
            for _ in range(60):
                if abs(b - a) < 1e-12:
                    break
                if fc < fd:
                    b = d
                    d = c
                    fd = fc
                    c = b - gr * (b - a)
                    fc = angle_obj(c)
                else:
                    a = c
                    c = d
                    fc = fd
                    d = a + gr * (b - a)
                    fd = angle_obj(d)
            
            new_theta = (a + b) / 2
            new_S = angle_obj(new_theta)
            if new_S < best_S:
                best_S = new_S
                best_theta = new_theta
    except Exception:
        pass
    
    # Phase 4: Try optimizing inner positions slightly (allow small perturbations)
    # This could help if the rigid assumption is not quite optimal
    try:
        if time.time() < DEADLINE - 20:
            x0_joint = np.zeros(2 * n_inner + 3)
            for i in range(n_inner):
                x0_joint[2*i] = centers[i][0]
                x0_joint[2*i+1] = centers[i][1]
            x0_joint[2*n_inner] = best_center[0]
            x0_joint[2*n_inner + 1] = best_center[1]
            x0_joint[2*n_inner + 2] = best_theta
            
            res = minimize(_objective_joint, x0_joint, args=(angles,),
                          method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            
            if res.fun < best_S:
                new_centers = np.array([[res.x[2*i], res.x[2*i+1]] for i in range(n_inner)])
                pen = _overlap_penalty(new_centers, angles)
                if pen < 1e-6:
                    new_hull = _compute_cluster_hull(new_centers, angles)
                    new_S = _min_enclosing_hex_side(new_hull, res.x[2*n_inner+2], 
                                                    np.array([res.x[2*n_inner], res.x[2*n_inner+1]]))
                    if new_S < best_S:
                        best_S = new_S
                        best_center = np.array([res.x[2*n_inner], res.x[2*n_inner+1]])
                        best_theta = res.x[2*n_inner+2]
                        centers = new_centers
    except Exception:
        pass
    
    # Phase 5: Try different outer ring configurations
    # Maybe 6 outer ring hexagons at different positions
    try:
        if time.time() < DEADLINE - 15:
            # Try full outer ring (6 hexagons at distance 3)
            centers2 = []
            angles2 = []
            centers2.append([0.0, 0.0])
            angles2.append(30.0)
            for k in range(6):
                angle = np.deg2rad(k * 60)
                r = np.sqrt(3)
                centers2.append([r * np.cos(angle), r * np.sin(angle)])
                angles2.append(30.0)
            for k in range(6):
                angle = np.deg2rad(k * 60)
                r = 3.0
                centers2.append([r * np.cos(angle), r * np.sin(angle)])
                angles2.append(30.0)
            
            # This gives 13 hexagons, not 11. Skip.
            # Instead try different outer ring positions
            # Try 4 outer hexagons at 0, 90, 180, 270
            centers3 = []
            angles3 = []
            centers3.append([0.0, 0.0])
            angles3.append(30.0)
            for k in range(6):
                angle = np.deg2rad(k * 60)
                r = np.sqrt(3)
                centers3.append([r * np.cos(angle), r * np.sin(angle)])
                angles3.append(30.0)
            for k in range(4):
                angle = np.deg2rad(k * 90)  # 0, 90, 180, 270
                r = 3.0
                centers3.append([r * np.cos(angle), r * np.sin(angle)])
                angles3.append(30.0)
            
            hull3 = _compute_cluster_hull(np.array(centers3), np.array(angles3))
            centroid3 = np.mean(centers3, axis=0)
            theta3, S3 = _find_optimal_rotation(hull3, centroid3, n_samples=3600)
            
            if S3 < best_S:
                best_S = S3
                best_theta = theta3
                best_center = centroid3
                centers = np.array(centers3)
                angles = np.array(angles3)
                n_inner = len(centers)
    except Exception:
        pass
    
    # Final computation
    hull_final = _compute_cluster_hull(centers, angles)
    final_S = _min_enclosing_hex_side(hull_final, best_theta, best_center)
    
    # Build output
    inner = []
    for i in range(n_inner):
        inner.append([centers[i][0], centers[i][1], angles[i]])
    
    outer_center_out = (best_center[0], best_center[1])
    
    return inner, outer_center_out, final_S, best_theta
