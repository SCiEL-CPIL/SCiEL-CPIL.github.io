import math
import time
import numpy as np
from itertools import combinations
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    sqrt3 = math.sqrt(3.0)

    def get_hex_vertices(cx, cy, angle_deg):
        angles = [math.radians(angle_deg + 60*k) for k in range(6)]
        return [(cx + math.cos(a), cy + math.sin(a)) for a in angles]

    def point_in_polygon(pt, poly):
        x, y = pt
        n = len(poly)
        inside = False
        p1x, p1y = poly[0]
        for i in range(1, n + 1):
            p2x, p2y = poly[i % n]
            if (y > p1y) != (y > p2y):
                xint = (p2x - p1x) * (y - p1y) / (p2y - p1y) + p1x
                if x < xint:
                    inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def segments_intersect(p1, p2, p3, p4):
        def ccw(A, B, C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
        return (ccw(p1, p3, p4) != ccw(p2, p3, p4)) and (ccw(p1, p2, p3) != ccw(p1, p2, p4))

    def polygons_disjoint(v1, v2, eps=1e-9):
        for v in v1:
            if point_in_polygon(v, v2):
                return False
        for v in v2:
            if point_in_polygon(v, v1):
                return False
        n1 = len(v1)
        n2 = len(v2)
        for i in range(n1):
            p1 = v1[i]
            p2 = v1[(i+1)%n1]
            for j in range(n2):
                p3 = v2[j]
                p4 = v2[(j+1)%n2]
                if segments_intersect(p1, p2, p3, p4):
                    return False
        return True

    def point_polygon_distance(pt, poly):
        min_dist = float('inf')
        n = len(poly)
        for i in range(n):
            p1 = poly[i]
            p2 = poly[(i+1)%n]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            if dx == 0 and dy == 0:
                d = math.hypot(pt[0]-p1[0], pt[1]-p1[1])
            else:
                t = ((pt[0]-p1[0])*dx + (pt[1]-p1[1])*dy) / (dx*dx + dy*dy)
                t = max(0, min(1, t))
                proj_x = p1[0] + t*dx
                proj_y = p1[1] + t*dy
                d = math.hypot(pt[0]-proj_x, pt[1]-proj_y)
            if d < min_dist:
                min_dist = d
        return min_dist

    def get_min_L(vertices_list):
        max_proj = 0.0
        normals = [math.radians(30 + 60*k) for k in range(6)]
        for v in vertices_list:
            for ang in normals:
                proj = v[0]*math.cos(ang) + v[1]*math.sin(ang)
                if proj > max_proj:
                    max_proj = proj
        return max_proj / (sqrt3 / 2.0)

    # Generate all 19 centers of the side-3 hexagon cluster
    all_centers = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            if abs(q + r) <= 3:
                x = 1.5 * q
                y = sqrt3 * (r + q / 2.0)
                all_centers.append((x, y, q, r))

    normal_angles = [math.radians(30 + 60 * j) for j in range(6)]
    normal_cos = np.array([math.cos(a) for a in normal_angles])
    normal_sin = np.array([math.sin(a) for a in normal_angles])

    def min_L_for_hexagon(cx, cy, theta_rad):
        """Find the minimum L needed for a unit hexagon at (cx,cy) with rotation theta to fit in outer hex of side L."""
        vertex_angles = [math.radians(60 * k) + theta_rad for k in range(6)]
        vx = cx + np.array([math.cos(a) for a in vertex_angles])
        vy = cy + np.array([math.sin(a) for a in vertex_angles])
        projs = vx[:, np.newaxis] * normal_cos[np.newaxis, :] + vy[:, np.newaxis] * normal_sin[np.newaxis, :]
        max_proj = np.max(projs)
        return max_proj / (sqrt3 / 2.0)

    def optimal_theta_for_hexagon(cx, cy):
        """Find rotation theta that minimizes the required L for this hexagon."""
        best_L = float('inf')
        best_theta = 0.0
        for t_deg in range(0, 61, 2):
            t = math.radians(t_deg)
            L = min_L_for_hexagon(cx, cy, t)
            if L < best_L:
                best_L = L
                best_theta = t
        # Refine
        for _ in range(4):
            step = [1.0, 0.5, 0.1, 0.01][_ % 4]
            lo = best_theta - step * 5
            hi = best_theta + step * 5
            for t_deg in np.arange(lo, hi + step, step):
                t = math.radians(t_deg)
                L = min_L_for_hexagon(cx, cy, t)
                if L < best_L:
                    best_L = L
                    best_theta = t
        return best_L, best_theta

    SCALE = 1.0000001

    best_L = float('inf')
    best_subset = None
    best_thetas = None
    best_cluster_angle = 0.0

    # Search over cluster rotation angles
    cluster_angles_to_try = []
    for a_deg in range(0, 61, 3):
        cluster_angles_to_try.append(math.radians(a_deg))
    for extra_deg in [30, 15, 45, 10, 20, 25, 35, 40, 50, 55, 5, 35]:
        if math.radians(extra_deg) not in cluster_angles_to_try:
            cluster_angles_to_try.append(math.radians(extra_deg))

    for cluster_angle in cluster_angles_to_try:
        if time.time() > deadline - 30.0:
            break

        ca = math.cos(cluster_angle)
        sa = math.sin(cluster_angle)
        rotated_centers = []
        for x, y, q, r in all_centers:
            rx = x * ca - y * sa
            ry = x * sa + y * ca
            rotated_centers.append((rx, ry, q, r))

        center_optimal = []
        for i in range(19):
            x, y, _, _ = rotated_centers[i]
            L_h, theta_h = optimal_theta_for_hexagon(x * SCALE, y * SCALE)
            center_optimal.append((L_h, theta_h))

        # Try all subsets of size 12 from 19 centers
        sorted_indices = sorted(range(19), key=lambda i: center_optimal[i][0])
        
        for subset in combinations(range(19), 12):
            if time.time() > deadline - 25.0:
                break
            L = max(center_optimal[i][0] for i in subset)
            if L < best_L:
                best_L = L
                best_subset = subset
                best_thetas = [center_optimal[i][1] for i in subset]
                best_cluster_angle = cluster_angle

    if best_subset is None:
        best_cluster_angle = 0.0
        rotated_centers = all_centers
        center_optimal = []
        for i in range(19):
            x, y, _, _ = rotated_centers[i]
            L_h, theta_h = optimal_theta_for_hexagon(x * SCALE, y * SCALE)
            center_optimal.append((L_h, theta_h))
        best_subset = tuple(range(12))
        best_thetas = [center_optimal[i][1] for i in best_subset]
        best_L = max(center_optimal[i][0] for i in best_subset)

    # Joint refinement using continuous optimization
    def smooth_separation_penalty(params, beta=50.0):
        penalty = 0.0
        for i in range(N):
            xi, yi, thi = params[3*i], params[3*i+1], params[3*i+2]
            vi = get_hex_vertices(xi, yi, thi)
            for j in range(i+1, N):
                xj, yj, thj = params[3*j], params[3*j+1], params[3*j+2]
                vj = get_hex_vertices(xj, yj, thj)
                min_dist = float('inf')
                for v in vi:
                    if point_in_polygon(v, vj):
                        d = point_polygon_distance(v, vj)
                        min_dist = min(min_dist, d)
                for v in vj:
                    if point_in_polygon(v, vi):
                        d = point_polygon_distance(v, vi)
                        min_dist = min(min_dist, d)
                if min_dist == float('inf'):
                    n1 = len(vi)
                    n2 = len(vj)
                    for ii in range(n1):
                        p1 = vi[ii]
                        p2 = vi[(ii+1)%n1]
                        for jj in range(n2):
                            p3 = vj[jj]
                            p4 = vj[(jj+1)%n2]
                            dx = p4[0] - p3[0]
                            dy = p4[1] - p3[1]
                            if dx == 0 and dy == 0:
                                d = math.hypot(p1[0]-p3[0], p1[1]-p3[1])
                            else:
                                t = ((p1[0]-p3[0])*dx + (p1[1]-p3[1])*dy) / (dx*dx + dy*dy)
                                t = max(0, min(1, t))
                                proj_x = p3[0] + t*dx
                                proj_y = p3[1] + t*dy
                                d = math.hypot(p1[0]-proj_x, p1[1]-proj_y)
                            min_dist = min(min_dist, d)
                            dx = p2[0] - p1[0]
                            dy = p2[1] - p1[1]
                            if dx == 0 and dy == 0:
                                d = math.hypot(p3[0]-p1[0], p3[1]-p1[1])
                            else:
                                t = ((p3[0]-p1[0])*dx + (p3[1]-p1[1])*dy) / (dx*dx + dy*dy)
                                t = max(0, min(1, t))
                                proj_x = p1[0] + t*dx
                                proj_y = p1[1] + t*dy
                                d = math.hypot(p3[0]-proj_x, p3[1]-proj_y)
                            min_dist = min(min_dist, d)
                if min_dist < 0.0:
                    penalty += 50.0 * min_dist * min_dist
                elif min_dist < 0.01:
                    penalty += 0.1 * (0.01 - min_dist) ** 2
        return penalty

    def objective(params, penalty_weight=1.0):
        if time.time() > deadline:
            return 1e10
        all_vertices = []
        for i in range(N):
            x = params[3*i]
            y = params[3*i+1]
            th = params[3*i+2]
            verts = get_hex_vertices(x, y, th)
            all_vertices.extend(verts)
        L_req = get_min_L(all_vertices)
        penalty = smooth_separation_penalty(params)
        return L_req + penalty_weight * penalty

    # Build initial params from combinatorial solution
    ca = math.cos(best_cluster_angle)
    sa = math.sin(best_cluster_angle)
    init_params = []
    for idx, i in enumerate(best_subset):
        x, y, _, _ = all_centers[i]
        rx = (x * ca - y * sa) * SCALE
        ry = (x * sa + y * ca) * SCALE
        init_params.extend([rx, ry, best_thetas[idx] * 180.0 / math.pi])

    # Phase 1: Nelder-Mead polish
    best_params = np.array(init_params)
    best_score = objective(best_params)
    
    if time.time() < deadline - 20.0:
        try:
            res = minimize(objective, best_params, method='Nelder-Mead', 
                           options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Phase 2: L-BFGS-B
    if time.time() < deadline - 10.0:
        try:
            res = minimize(lambda p: objective(p, 5.0), best_params, method='L-BFGS-B', 
                           options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 3000})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Phase 3: Fine Nelder-Mead
    if time.time() < deadline - 5.0:
        try:
            res = minimize(lambda p: objective(p, 10.0), best_params, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Evaluate final result
    positions = []
    all_vertices = []
    for i in range(N):
        x = best_params[3*i]
        y = best_params[3*i+1]
        th = best_params[3*i+2]
        positions.append([x, y, th])
        verts = get_hex_vertices(x, y, th)
        all_vertices.extend(verts)
        
    L_final = get_min_L(all_vertices)
    
    # Verify disjointness
    is_valid = True
    verts_list = [get_hex_vertices(pos[0], pos[1], pos[2]) for pos in positions]
    for i in range(N):
        for j in range(i+1, N):
            if not polygons_disjoint(verts_list[i], verts_list[j]):
                is_valid = False
                break
        if not is_valid:
            break
    
    if not is_valid:
        # Fallback to combinatorial solution
        inner = []
        ca = math.cos(best_cluster_angle)
        sa = math.sin(best_cluster_angle)
        for idx, i in enumerate(best_subset):
            x, y, _, _ = all_centers[i]
            rx = (x * ca - y * sa) * SCALE
            ry = (x * sa + y * ca) * SCALE
            inner.append([rx, ry, best_thetas[idx] * 180.0 / math.pi])
        
        fb_vertices = []
        for p in inner:
            v = get_hex_vertices(p[0], p[1], p[2])
            fb_vertices.extend(v)
        L_final = get_min_L(fb_vertices)
        return inner, (0.0, 0.0), L_final, 0.0

    return positions, (0.0, 0.0), L_final, 0.0
