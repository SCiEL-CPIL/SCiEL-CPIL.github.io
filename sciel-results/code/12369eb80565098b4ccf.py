import numpy as np
from scipy.optimize import minimize
import time
import math
import random

N = 12
SQRT3 = np.sqrt(3.0)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill

    # ---------- Geometry helpers ----------
    
    def hex_vertices(cx, cy, side, angle_rad):
        angles = np.linspace(0, 2 * np.pi, 7)[:-1] + angle_rad
        xs = cx + side * np.cos(angles)
        ys = cy + side * np.sin(angles)
        return np.column_stack((xs, ys))

    def hexagon_vertices(cx, cy, side, angle_deg):
        return hex_vertices(cx, cy, side, np.radians(angle_deg))

    def point_in_polygon(point, polygon):
        x, y = point
        n = len(polygon)
        for i in range(n):
            x1, y1 = polygon[i]
            x2, y2 = polygon[(i + 1) % n]
            cross = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
            if cross < -1e-9:
                return False
        return True

    def polygons_intersect(poly1, poly2):
        def project_polygon(axis, poly):
            projections = np.dot(poly, axis)
            return np.min(projections), np.max(projections)

        def get_edges(poly):
            edges = []
            n = len(poly)
            for i in range(n):
                p1 = poly[i]
                p2 = poly[(i + 1) % n]
                edge = p2 - p1
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm > 1e-9:
                    normal /= norm
                edges.append(normal)
            return edges

        axes = get_edges(poly1) + get_edges(poly2)
        
        for axis in axes:
            min1, max1 = project_polygon(axis, poly1)
            min2, max2 = project_polygon(axis, poly2)
            if max1 < min2 or max2 < min1:
                return False
        return True

    def distance_point_to_segment(p, a, b):
        ab = b - a
        ap = p - a
        ab_len_sq = np.dot(ab, ab)
        if ab_len_sq < 1e-9:
            return np.linalg.norm(p - a)
        t = np.dot(ap, ab) / ab_len_sq
        t = np.clip(t, 0, 1)
        proj = a + t * ab
        return np.linalg.norm(p - proj)

    def min_distance_to_polygon_boundary(point, polygon):
        min_d = np.inf
        n = len(polygon)
        for i in range(n):
            p1 = polygon[i]
            p2 = polygon[(i + 1) % n]
            d = distance_point_to_segment(point, p1, p2)
            if d < min_d:
                min_d = d
        return min_d

    def min_distance_between_polygons(poly1, poly2):
        if polygons_intersect(poly1, poly2):
            min_d = np.inf
            for v in poly1:
                d = min_distance_to_polygon_boundary(v, poly2)
                if d < min_d:
                    min_d = d
            return -min_d * 10.0
        
        min_d = np.inf
        for v in poly1:
            d = min_distance_to_polygon_boundary(v, poly2)
            if d < min_d:
                min_d = d
        for v in poly2:
            d = min_distance_to_polygon_boundary(v, poly1)
            if d < min_d:
                min_d = d
        return min_d

    def check_validity(inner, outer_center, outer_side, outer_angle):
        if outer_side <= 0:
            return False, np.inf

        outer_verts = hex_vertices(outer_center, outer_side, np.radians(outer_angle))
        n_inner = len(inner)
        inner_verts = []
        
        for i in range(n_inner):
            x, y, ang = inner[i]
            v = hex_vertices((x, y), 1.0, np.radians(ang))
            inner_verts.append(v)
            
        for i in range(n_inner):
            v = inner_verts[i]
            for pt in v:
                if not point_in_polygon(pt, outer_verts):
                    return False, np.inf

        for i in range(n_inner):
            for j in range(i + 1, n_inner):
                if polygons_intersect(inner_verts[i], inner_verts[j]):
                    return False, np.inf
        return True, 0.0

    # ---------- Fast geometry from inspiration ----------
    
    def hexagons_disjoint_fast(h1, h2):
        """Fast check for disjointness using SAT."""
        dx = h1[0] - h2[0]
        dy = h1[1] - h2[1]
        dist_sq = dx*dx + dy*dy
        
        # If dist > 2.0, definitely disjoint
        if dist_sq > 4.0:
            return True
        
        v1 = hexagon_vertices(h1[0], h1[1], 1.0, h1[2])
        v2 = hexagon_vertices(h2[0], h2[1], 1.0, h2[2])
        
        for i in range(6):
            e1 = v1[(i+1) % 6] - v1[i]
            n1 = np.array([-e1[1], e1[0]])
            nn1 = np.linalg.norm(n1)
            if nn1 > 1e-12:
                n1 = n1 / nn1
                proj1 = np.dot(v1, n1)
                proj2 = np.dot(v2, n1)
                if np.max(proj1) < np.min(proj2) - 1e-9 or np.max(proj2) < np.min(proj1) - 1e-9:
                    return True
            
            e2 = v2[(i+1) % 6] - v2[i]
            n2 = np.array([-e2[1], e2[0]])
            nn2 = np.linalg.norm(n2)
            if nn2 > 1e-12:
                n2 = n2 / nn2
                proj1 = np.dot(v1, n2)
                proj2 = np.dot(v2, n2)
                if np.max(proj1) < np.min(proj2) - 1e-9 or np.max(proj2) < np.min(proj1) - 1e-9:
                    return True
        return False

    def compute_min_side_fast(inner, center, angle):
        """Vectorized computation of minimum outer side length."""
        sqrt3 = np.sqrt(3)
        max_extent = 0.0
        
        all_verts = np.vstack([hexagon_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2]) for i in range(N)])
        offsets = all_verts - np.array(center)
        
        for k in range(6):
            normal_angle = np.radians(angle) + np.radians(30) + k * np.pi / 3
            normal = np.array([np.cos(normal_angle), np.sin(normal_angle)])
            extents = np.dot(offsets, normal)
            max_e = np.max(extents)
            if max_e > max_extent:
                max_extent = max_e
        
        return max_extent * 2.0 / sqrt3

    # ---------- Penalty function for Nelder-Mead ----------
    
    def penalty_function(params):
        outer_side = params[0]
        outer_cx, outer_cy = params[1:3]
        outer_angle = params[3]
        
        inner_params = params[4:]
        inner = []
        for i in range(N):
            x = inner_params[3*i]
            y = inner_params[3*i+1]
            a = inner_params[3*i+2]
            inner.append((x, y, a))
        
        outer_center = (outer_cx, outer_cy)
        
        inner_verts = []
        for i in range(N):
            x, y, a = inner[i]
            v = hex_vertices((x, y), 1.0, np.radians(a))
            inner_verts.append(v)
            
        outer_verts = hex_vertices(outer_center, outer_side, np.radians(outer_angle))
        
        total_penalty = 0.0
        
        for v in inner_verts:
            for pt in v:
                if not point_in_polygon(pt, outer_verts):
                    dist = min_distance_to_polygon_boundary(pt, outer_verts)
                    total_penalty += dist * dist * 100.0

        for i in range(N):
            for j in range(i + 1, N):
                min_dist = min_distance_between_polygons(inner_verts[i], inner_verts[j])
                if min_dist < 0:
                    total_penalty += (-min_dist) * 100.0
                elif min_dist < 1e-3:
                    total_penalty += (1e-3 - min_dist) * 10.0

        return outer_side + total_penalty

    # ---------- Soft-body differentiable energy for L-BFGS-B ----------
    
    def soft_body_energy(params):
        """
        Smooth energy for gradient-based optimization.
        Uses smooth approximations of overlap and containment.
        """
        outer_side = params[0]
        outer_cx, outer_cy = params[1:3]
        outer_angle = params[3]
        
        inner_params = params[4:]
        inner = []
        for i in range(N):
            x = inner_params[3*i]
            y = inner_params[3*i+1]
            a = inner_params[3*i+2]
            inner.append((x, y, a))
        
        outer_center = np.array([outer_cx, outer_cy])
        
        # Compute smooth containment penalty
        outer_verts = hex_vertices(outer_center, outer_side, np.radians(outer_angle))
        
        # Smooth containment: for each inner vertex, compute signed distance to outer polygon
        # Using a smooth max approximation
        containment_penalty = 0.0
        for i in range(N):
            x, y, a = inner[i]
            v = hex_vertices((x, y), 1.0, np.radians(a))
            for pt in v:
                # Compute distance to each edge, take max (violation)
                violations = []
                for j in range(6):
                    x1, y1 = outer_verts[j]
                    x2, y2 = outer_verts[(j + 1) % 6]
                    # Signed distance: positive if inside
                    ex = x2 - x1
                    ey = y2 - y1
                    edge_len = np.sqrt(ex*ex + ey*ey)
                    if edge_len < 1e-9:
                        continue
                    # Normal pointing inward
                    nx = -ey / edge_len
                    ny = ex / edge_len
                    # Distance from point to edge line
                    d = (pt[0] - x1) * nx + (pt[1] - y1) * ny
                    violations.append(-d)  # Negative if inside
                if violations:
                    # Smooth max of violations (using log-sum-exp approximation)
                    max_v = max(violations)
                    if max_v > 0:
                        # Smooth approximation
                        smooth_max = max_v + 0.1 * np.log(np.sum([np.exp(v / 0.1) for v in violations]))
                        containment_penalty += smooth_max * smooth_max * 50.0
        
        # Smooth pairwise overlap penalty
        overlap_penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                x1, y1, a1 = inner[i]
                x2, y2, a2 = inner[j]
                v1 = hex_vertices((x1, y1), 1.0, np.radians(a1))
                v2 = hex_vertices((x2, y2), 1.0, np.radians(a2))
                
                # Compute minimum distance between polygons using smooth approximation
                # Use all vertex-vertex and vertex-edge distances
                min_dist_sq = np.inf
                
                # Vertex-vertex
                for p1 in v1:
                    for p2 in v2:
                        d_sq = (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
                        if d_sq < min_dist_sq:
                            min_dist_sq = d_sq
                
                # Vertex-edge
                for p in v1:
                    for k in range(6):
                        a = v2[k]
                        b = v2[(k+1) % 6]
                        ab = b - a
                        ap = p - a
                        ab_len_sq = np.dot(ab, ab)
                        if ab_len_sq < 1e-9:
                            d_sq = np.dot(ap, ap)
                        else:
                            t = np.dot(ap, ab) / ab_len_sq
                            t = np.clip(t, 0, 1)
                            proj = a + t * ab
                            d_sq = np.dot(p - proj, p - proj)
                        if d_sq < min_dist_sq:
                            min_dist_sq = d_sq
                
                for p in v2:
                    for k in range(6):
                        a = v1[k]
                        b = v1[(k+1) % 6]
                        ab = b - a
                        ap = p - a
                        ab_len_sq = np.dot(ab, ab)
                        if ab_len_sq < 1e-9:
                            d_sq = np.dot(ap, ap)
                        else:
                            t = np.dot(ap, ab) / ab_len_sq
                            t = np.clip(t, 0, 1)
                            proj = a + t * ab
                            d_sq = np.dot(p - proj, p - proj)
                        if d_sq < min_dist_sq:
                            min_dist_sq = d_sq
                
                min_dist = np.sqrt(min_dist_sq)
                
                # If overlapping, penalty
                if min_dist < 0.1:
                    # Smooth overlap penalty
                    overlap_penalty += (0.1 - min_dist) ** 2 * 200.0
        
        # Objective: outer side + penalties
        return outer_side + containment_penalty + overlap_penalty

    def soft_body_energy_grad(params):
        """
        Numerical gradient for soft_body_energy.
        """
        eps = 1e-5
        grad = np.zeros_like(params)
        base = soft_body_energy(params)
        
        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += eps
            f_plus = soft_body_energy(params_plus)
            params_minus = params.copy()
            params_minus[i] -= eps
            f_minus = soft_body_energy(params_minus)
            grad[i] = (f_plus - f_minus) / (2 * eps)
        
        return grad

    # ---------- Initializations ----------
    
    def init_grid():
        params = np.zeros(4 + 3*N)
        params[0] = 10.0
        params[1] = 0.0
        params[2] = 0.0
        params[3] = 0.0
        
        for i in range(N):
            col = i % 4
            row = i // 4
            ix = col * 3.5
            iy = row * 3.0
            params[4 + 3*i] = ix
            params[4 + 3*i+1] = iy
            params[4 + 3*i+2] = 0.0
        return params

    def init_hex_cluster():
        params = np.zeros(4 + 3*N)
        params[0] = 10.0
        params[1] = 0.0
        params[2] = 0.0
        params[3] = 0.0
        
        idx = 0
        params[4 + 3*idx] = 0
        params[4 + 3*idx+1] = 0
        params[4 + 3*idx+2] = 0
        
        for k in range(6):
            idx = k + 1
            ang = np.radians(k * 60)
            dist = 2.0
            params[4 + 3*idx] = dist * np.cos(ang)
            params[4 + 3*idx+1] = dist * np.sin(ang)
            params[4 + 3*idx+2] = 0.0
            
        for k in range(5):
            idx = k + 7
            ang = np.radians(k * 72 + 30)
            dist = 4.0
            params[4 + 3*idx] = dist * np.cos(ang)
            params[4 + 3*idx+1] = dist * np.sin(ang)
            params[4 + 3*idx+2] = 0.0
            
        return params

    def init_honeycomb():
        """Honeycomb-like packing."""
        params = np.zeros(4 + 3*N)
        params[0] = 10.0
        params[1] = 0.0
        params[2] = 0.0
        params[3] = 0.0
        
        # 3 rows of 4
        for i in range(N):
            row = i // 4
            col = i % 4
            x = col * 3.0 + (row % 2) * 1.5
            y = row * 2.6
            params[4 + 3*i] = x
            params[4 + 3*i+1] = y
            params[4 + 3*i+2] = 0.0
        return params

    initializations = [init_grid(), init_hex_cluster(), init_honeycomb()]
    
    best_score = np.inf
    best_params = None
    
    # Phase 1: Nelder-Mead with multiple starts
    for init_params in initializations:
        if time.time() > deadline - 30.0:
            break
            
        try:
            res = minimize(penalty_function, init_params, method='Nelder-Mead', 
                           options={'maxiter': 300, 'xatol': 1e-4, 'fatol': 1e-4})
            
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
                
        except Exception:
            pass

    # Phase 2: L-BFGS-B with soft-body energy from best Nelder-Mead result
    if best_params is not None and time.time() < deadline - 10.0:
        try:
            # Convert best_params to soft-body format (same format)
            lbfgs_start = best_params.copy()
            
            res2 = minimize(soft_body_energy, lbfgs_start, method='L-BFGS-B',
                           jac=soft_body_energy_grad,
                           options={'maxiter': 100, 'gtol': 1e-4, 'ftol': 1e-6})
            
            if res2.fun < best_score:
                best_score = res2.fun
                best_params = res2.x.copy()
                
        except Exception:
            pass

    # Phase 3: Final Nelder-Mead polish from L-BFGS-B result
    if best_params is not None and time.time() < deadline - 5.0:
        try:
            res3 = minimize(penalty_function, best_params, method='Nelder-Mead', 
                           options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-5})
            
            if res3.fun < best_score:
                best_score = res3.fun
                best_params = res3.x.copy()
                
        except Exception:
            pass

    if best_params is None:
        inner = []
        for i in range(N):
            col = i % 4
            row = i // 4
            x = col * 3.5 - 5.0
            y = row * 3.0 - 1.5
            inner.append((x, y, 0))
        return inner, (0, 0), 10.0, 0.0

    outer_side = best_params[0]
    outer_cx = best_params[1]
    outer_cy = best_params[2]
    outer_angle = best_params[3]
    
    inner = []
    for i in range(N):
        x = best_params[4 + 3*i]
        y = best_params[4 + 3*i+1]
        a = best_params[4 + 3*i+2]
        inner.append((x, y, a))
        
    outer_center = (outer_cx, outer_cy)
    
    # Final validity check
    valid, _ = check_validity(inner, outer_center, outer_side, outer_angle)
    
    if not valid:
        # Try to fix by using compute_min_side_fast
        side_fixed = compute_min_side_fast(inner, outer_center, outer_angle)
        if side_fixed < outer_side * 1.1:
            outer_side = side_fixed
            valid, _ = check_validity(inner, outer_center, outer_side, outer_angle)
    
    if not valid:
        # Fallback to a large valid configuration
        inner = []
        for i in range(N):
            col = i % 4
            row = i // 4
            x = col * 3.5 - 5.0
            y = row * 3.0 - 1.5
            inner.append((x, y, 0))
        outer_side = 10.0
        outer_center = (0, 0)
        outer_angle = 0.0
        
    return inner, outer_center, outer_side, outer_angle
