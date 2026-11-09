# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, d, angle_deg):
    a = np.radians(angle_deg)
    angles = a + np.radians(60) * np.arange(6)
    return np.column_stack((cx + d * np.cos(angles), cy + d * np.sin(angles)))

def _hex_edge_normals(vertices):
    n = 6
    normals = np.zeros((n, 2))
    for i in range(n):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % n]
        edge = p2 - p1
        normal = np.array([edge[1], -edge[0]])
        norm = np.linalg.norm(normal)
        if norm > 1e-15:
            normals[i] = normal / norm
    return normals

def _separation_check(v1, v2):
    n1 = _hex_edge_normals(v1)
    n2 = _hex_edge_normals(v2)
    all_normals = np.vstack([n1, n2])
    for normal in all_normals:
        proj1 = v1 @ normal
        proj2 = v2 @ normal
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        if max1 < min2 - 1e-6 or max2 < min1 - 1e-6:
            return True
    return False

def _is_inside_outer(inner_vertices, outer_vertices):
    n_out = len(outer_vertices)
    for v in inner_vertices:
        inside = True
        for i in range(n_out):
            p1 = outer_vertices[i]
            p2 = outer_vertices[(i + 1) % n_out]
            cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
            if cross < -1e-6:
                inside = False
                break
        if not inside:
            return False
    return True

def _compute_s(inner_positions, outer_center, outer_angle_deg, inner_angles):
    outer_center = np.array(outer_center)
    inner_centers = np.array(inner_positions)
    shifted = inner_centers - outer_center
    
    a = np.radians(outer_angle_deg)
    normal_angles = a + np.radians(60) * np.arange(6) + np.radians(30)
    normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
    
    all_vertices = []
    for i in range(len(inner_centers)):
        cx, cy = shifted[i]
        verts = _hex_vertices(cx, cy, 1, inner_angles[i])
        all_vertices.append(verts)
    all_vertices = np.vstack(all_vertices)
    
    projections = all_vertices @ normals.T
    max_proj = projections.max()
    
    s = 2 * max_proj / np.sqrt(3)
    return s

def _build_honeycomb_cluster():
    """
    Build a fixed honeycomb cluster of 12 unit hexagons.
    Uses a 3-4-3-2 arrangement (rows of 3, 4, 3, 2 hexagons).
    All hexagons have angle 0 (flat-top orientation).
    Returns positions (12, 2) and angles (12,).
    """
    # For flat-top hexagons with side 1:
    # Horizontal spacing between adjacent centers in same row: 2 * (sqrt(3)/2) = sqrt(3)
    # Vertical spacing between rows: 1.5 (for flat-top)
    # Actually for pointy-top: horizontal = 1.5, vertical = sqrt(3)
    # Let's use pointy-top orientation (angle 90 or 0 depending on convention)
    
    # With angle_deg=0, vertex 0 is at angle 0 (right), so it's flat-top
    # Flat-top: width = 2, height = sqrt(3)
    # Adjacent in same row: dx = sqrt(3), dy = 0
    # Adjacent in next row: dx = sqrt(3)/2, dy = 1.5
    
    sqrt3 = np.sqrt(3)
    
    positions = []
    angles = []
    
    # Row 0: 3 hexagons
    for j in range(3):
        x = j * sqrt3
        y = 0
        positions.append([x, y])
        angles.append(0)
    
    # Row 1: 4 hexagons (offset)
    for j in range(4):
        x = (j + 0.5) * sqrt3
        y = 1.5
        positions.append([x, y])
        angles.append(0)
    
    # Row 2: 3 hexagons
    for j in range(3):
        x = j * sqrt3
        y = 3.0
        positions.append([x, y])
        angles.append(0)
    
    # Row 3: 2 hexagons (offset)
    for j in range(2):
        x = (j + 0.5) * sqrt3
        y = 4.5
        positions.append([x, y])
        angles.append(0)
    
    return np.array(positions), np.array(angles)

def _compute_minimal_s_for_cluster(cluster_positions, cluster_angles, outer_angle_deg):
    """
    Given a fixed cluster (positions and angles all fixed), find the minimal
    enclosing regular hexagon with the given outer angle.
    The outer hexagon center is free to move.
    """
    cluster_positions = np.array(cluster_positions)
    cluster_angles = np.array(cluster_angles)
    
    # Compute all vertices of the cluster in local coordinates
    all_vertices = []
    for i in range(len(cluster_positions)):
        cx, cy = cluster_positions[i]
        verts = _hex_vertices(cx, cy, 1, cluster_angles[i])
        all_vertices.append(verts)
    all_vertices = np.vstack(all_vertices)
    
    # The outer hexagon has angle outer_angle_deg
    # Its edge normals are at angles outer_angle_deg + 30 + 60*k
    a = np.radians(outer_angle_deg)
    normal_angles = a + np.radians(30) + np.radians(60) * np.arange(6)
    normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
    
    # For a regular hexagon with center c and side s, a point p is inside iff
    # for all normals n_k: (p - c) . n_k <= s / sqrt(3)
    # So we need: max_k max_p (p - c) . n_k <= s / sqrt(3)
    # => s >= sqrt(3) * max_k [max_p (p . n_k) - c . n_k]
    # To minimize s over c, for each k we want c . n_k to be as large as possible
    # subject to the constraint being tight for all k.
    
    # Actually, the minimal s is achieved when the hexagon is centered such that
    # the projections are balanced. Let's use the fact that for a regular hexagon,
    # the inradius is s * sqrt(3) / 2, so s = 2 * inradius / sqrt(3).
    
    # The inradius must be at least max over all normals of (max projection - min projection) / 2
    # when centered optimally. Actually, for a convex body, the minimal enclosing regular
    # hexagon with fixed orientation has inradius = max_k (range_k) / 2 where range_k is
    # the range of projections onto normal k, and the center is chosen to center each projection.
    
    # More precisely: for each normal direction n_k, the projection of the cluster has
    # a range [min_k, max_k]. The hexagon centered at c with inradius r satisfies
    # (p - c) . n_k <= r for all p, so c . n_k >= max_k - r and c . n_k <= min_k + r.
    # For feasibility, we need max_k - r <= min_k + r => r >= (max_k - min_k) / 2.
    # The minimal r is max_k (max_k - min_k) / 2, and c is chosen to satisfy all constraints.
    
    projections = all_vertices @ normals.T  # shape (72, 6)
    mins = projections.min(axis=0)  # shape (6,)
    maxs = projections.max(axis=0)  # shape (6,)
    ranges = maxs - mins  # shape (6,)
    r_min = np.max(ranges) / 2  # minimal inradius
    
    s = 2 * r_min / np.sqrt(3)
    
    # Find center c such that for all k: maxs[k] - r <= c . n_k <= mins[k] + r
    # with r = r_min
    # Use linear programming or just pick c to satisfy the tightest constraints
    # For simplicity, use the midpoint of the feasible region for each constraint
    
    # Solve for c: minimize 0 subject to A c <= b
    # A = -normals (6x2), b = r_min - maxs
    # A = normals (6x2), b = r_min + mins
    # So: -normals @ c <= r_min - maxs  =>  normals @ c >= maxs - r_min
    # And: normals @ c <= mins + r_min
    
    # Lower bounds: normals @ c >= maxs - r_min
    # Upper bounds: normals @ c <= mins + r_min
    
    lb = maxs - r_min
    ub = mins + r_min
    
    # Check feasibility: for each k, lb[k] <= ub[k] which is maxs[k] - r_min <= mins[k] + r_min
    # => maxs[k] - mins[k] <= 2 * r_min, which is true by construction.
    
    # Find a point c satisfying all. Use scipy linprog or just average.
    # Simple approach: c = 0, then adjust. Or use least squares to find c that
    # satisfies normals @ c = (lb + ub) / 2
    
    target = (lb + ub) / 2  # shape (6,)
    # Solve normals @ c = target in least squares sense
    c, residuals, rank, sv = np.linalg.lstsq(normals, target, rcond=None)
    
    return s, c

def _validate_solution(inner_positions, inner_angles, outer_center, outer_angle_deg, s):
    """Validate the full solution against all constraints."""
    # Check inner hexagons don't overlap
    for i in range(12):
        for j in range(i+1, 12):
            v1 = _hex_vertices(inner_positions[i,0], inner_positions[i,1], 1, inner_angles[i])
            v2 = _hex_vertices(inner_positions[j,0], inner_positions[j,1], 1, inner_angles[j])
            if not _separation_check(v1, v2):
                return False
    
    # Check all inner vertices are inside outer hexagon
    outer_verts = _hex_vertices(outer_center[0], outer_center[1], s, outer_angle_deg)
    for i in range(12):
        inner_verts = _hex_vertices(inner_positions[i,0], inner_positions[i,1], 1, inner_angles[i])
        if not _is_inside_outer(inner_verts, outer_verts):
            return False
    
    return True

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 580
    
    # Build fixed honeycomb cluster
    cluster_positions, cluster_angles = _build_honeycomb_cluster()
    
    best_s = np.inf
    best_outer_center = None
    best_outer_angle = None
    
    # Strategy: search over outer hexagon angle
    # The cluster is fixed, so we just need to find the best outer orientation
    # and center.
    
    # First, try a coarse grid of outer angles
    angle_range = np.linspace(0, 60, 121)  # 60-degree period
    
    for oa in angle_range:
        if time.time() > deadline - 10:
            break
        
        s, oc = _compute_minimal_s_for_cluster(cluster_positions, cluster_angles, oa)
        
        if s < best_s:
            best_s = s
            best_outer_center = oc
            best_outer_angle = oa
    
    # Refine with local optimization around the best angle found
    if best_outer_angle is not None and time.time() < deadline - 20:
        def obj_angle(params):
            oa = params[0]
            s, _ = _compute_minimal_s_for_cluster(cluster_positions, cluster_angles, oa)
            return s
        
        res = minimize(obj_angle, np.array([best_outer_angle]), method='Nelder-Mead',
                       options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 200})
        
        refined_angle = res.x[0] % 60
        s_refined, oc_refined = _compute_minimal_s_for_cluster(cluster_positions, cluster_angles, refined_angle)
        
        if s_refined < best_s:
            best_s = s_refined
            best_outer_center = oc_refined
            best_outer_angle = refined_angle
    
    # Now we have the best configuration. Let's also try to optimize the cluster
    # slightly - allow small translations and rotations of individual hexagons
    # to see if we can improve.
    
    if time.time() < deadline - 30:
        # Try perturbing the cluster positions slightly
        def obj_cluster(params):
            pos = params[:24].reshape(12, 2)
            angles = params[24:36]
            oa = params[36]
            
            # Check for overlaps
            penalty = 0.0
            for i in range(12):
                for j in range(i+1, 12):
                    v1 = _hex_vertices(pos[i,0], pos[i,1], 1, angles[i])
                    v2 = _hex_vertices(pos[j,0], pos[j,1], 1, angles[j])
                    n1 = _hex_edge_normals(v1)
                    n2 = _hex_edge_normals(v2)
                    all_normals = np.vstack([n1, n2])
                    min_sep = np.inf
                    for normal in all_normals:
                        proj1 = v1 @ normal
                        proj2 = v2 @ normal
                        min1, max1 = proj1.min(), proj1.max()
                        min2, max2 = proj2.min(), proj2.max()
                        if max1 < min2:
                            sep = min2 - max1
                        elif max2 < min1:
                            sep = min1 - max2
                        else:
                            sep = -min(max1 - min2, max2 - min1)
                        min_sep = min(min_sep, sep)
                    if min_sep < 0:
                        penalty += 1000 * min_sep * min_sep
            
            if penalty > 1:
                return 1000 + penalty
            
            s, _ = _compute_minimal_s_for_cluster(pos, angles, oa)
            return s + penalty
        
        initial_params = np.concatenate([cluster_positions.flatten(), cluster_angles, [best_outer_angle]])
        
        try:
            res = minimize(obj_cluster, initial_params, method='Nelder-Mead',
                           options={'xatol': 1e-6, 'fatol': 1e-8, 'maxiter': 500, 'maxfev': 3000})
            
            if res.fun < best_s and res.fun < 1000:
                pos_opt = res.x[:24].reshape(12, 2)
                angles_opt = res.x[24:36]
                oa_opt = res.x[36] % 60
                
                s_opt, oc_opt = _compute_minimal_s_for_cluster(pos_opt, angles_opt, oa_opt)
                
                if s_opt < best_s:
                    # Validate
                    if _validate_solution(pos_opt, angles_opt, oc_opt, oa_opt, s_opt):
                        best_s = s_opt
                        cluster_positions = pos_opt
                        cluster_angles = angles_opt
                        best_outer_center = oc_opt
                        best_outer_angle = oa_opt
                        
        except Exception as e:
            pass
    
    # Final validation
    if best_outer_center is not None:
        valid = _validate_solution(cluster_positions, cluster_angles, best_outer_center, best_outer_angle, best_s)
        if not valid:
            # Fallback: recompute
            best_s, best_outer_center = _compute_minimal_s_for_cluster(cluster_positions, cluster_angles, best_outer_angle)
            valid = _validate_solution(cluster_positions, cluster_angles, best_outer_center, best_outer_angle, best_s)
            if not valid:
                # Last resort: use original cluster with angle 0
                best_s, best_outer_center = _compute_minimal_s_for_cluster(cluster_positions, cluster_angles, 0)
                best_outer_angle = 0
    
    # Construct return values
    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i, 0] = cluster_positions[i, 0]
        inner_hex_data[i, 1] = cluster_positions[i, 1]
        inner_hex_data[i, 2] = cluster_angles[i]
    
    outer_hex_data = np.array([best_outer_center[0], best_outer_center[1], best_outer_angle])
    
    return inner_hex_data, outer_hex_data, best_s

# EVOLVE-BLOCK-END
