"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    def hex_vertices(cx, cy, angle_deg, side=1.0):
        """Return 6 vertices of a regular hexagon centered at (cx,cy) rotated by angle_deg."""
        angle_rad = math.radians(angle_deg)
        verts = np.zeros((6, 2))
        for i in range(6):
            a = angle_rad + i * math.pi / 3
            verts[i, 0] = cx + side * math.cos(a)
            verts[i, 1] = cy + side * math.sin(a)
        return verts

    def polygons_disjoint(v1, v2, tol=1e-9):
        """
        Check if two convex polygons are disjoint (non-overlapping) using SAT.
        Returns True if disjoint, False if overlapping.
        """
        for poly in [v1, v2]:
            for i in range(6):
                ax, ay = poly[i]
                bx, by = poly[(i + 1) % 6]
                ex, ey = bx - ax, by - ay
                nx, ny = -ey, ex
                proj1 = (v1 - v1[0]) @ np.array([nx, ny])
                proj2 = (v2 - v1[0]) @ np.array([nx, ny])
                if np.max(proj1) < np.min(proj2) - tol or np.max(proj2) < np.min(proj1) - tol:
                    return True
        return False

    def point_in_convex_polygon(px, py, poly):
        """Check if point (px,py) is inside convex polygon (CCW ordered vertices)."""
        n = len(poly)
        for i in range(n):
            ax, ay = poly[i]
            bx, by = poly[(i + 1) % n]
            cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
            if cross < -1e-9:
                return False
        return True

    def compute_outer_side(inner_data, outer_cx, outer_cy, outer_angle_deg):
        """
        Compute the minimum side length of a regular hexagon centered at (outer_cx, outer_cy)
        with rotation outer_angle_deg that contains all inner hexagons.
        """
        outer_center = np.array([outer_cx, outer_cy])
        outer_angle_rad = math.radians(outer_angle_deg)
        
        max_proj = 0.0
        for cx, cy, ang in inner_data:
            verts = hex_vertices(cx, cy, ang)
            rel_verts = verts - outer_center
            for k in range(6):
                normal_angle = outer_angle_rad + k * math.pi / 3 + math.pi / 6
                nx = math.cos(normal_angle)
                ny = math.sin(normal_angle)
                proj = rel_verts @ np.array([nx, ny])
                max_proj = max(max_proj, np.max(proj))
        
        return 2.0 * max_proj / math.sqrt(3)

    def check_validity(inner_data, outer_cx, outer_cy, outer_angle_deg, side, margin=1e-4):
        """Check if all inner hexagons are inside the outer hexagon and non-overlapping."""
        verts_list = [hex_vertices(cx, cy, ang) for cx, cy, ang in inner_data]
        for i in range(N):
            for j in range(i + 1, N):
                if not polygons_disjoint(verts_list[i], verts_list[j]):
                    return False
        
        # Check containment
        outer_verts = hex_vertices(outer_cx, outer_cy, outer_angle_deg, side)
        for i in range(N):
            for px, py in verts_list[i]:
                if not point_in_convex_polygon(px, py, outer_verts):
                    return False
        return True

    def build_honeycomb():
        """Build a honeycomb-like arrangement of 12 hexagons (4x3 grid)."""
        s = 1.0
        dx = 2.0 * s
        dy = math.sqrt(3) * s
        offset = s
        
        centers = []
        for r in range(4):
            for c in range(3):
                x = c * dx
                y = r * dy
                if r % 2 == 1:
                    x += offset
                centers.append((x, y))
        
        cx_mean = sum(c[0] for c in centers) / N
        cy_mean = sum(c[1] for c in centers) / N
        adjusted = [(x - cx_mean, y - cy_mean) for x, y in centers]
        
        inner_data = [(x, y, 0.0) for x, y in adjusted]
        return inner_data, 0.0, 0.0, 0.0

    def build_ring():
        """Build a ring of 12 hexagons around a central point."""
        inner_data = []
        for i in range(12):
            angle = i * math.pi / 6
            r = 2.5
            x = r * math.cos(angle)
            y = r * math.sin(angle)
            ang_deg = math.degrees(angle) + 90
            inner_data.append((x, y, ang_deg))
        return inner_data, 0.0, 0.0, 0.0

    def build_two_rings():
        """Build two concentric rings: 6 inner, 6 outer."""
        inner_data = []
        for i in range(6):
            angle = i * math.pi / 3
            r = 1.8
            x = r * math.cos(angle)
            y = r * math.sin(angle)
            ang_deg = math.degrees(angle) + 90
            inner_data.append((x, y, ang_deg))
        for i in range(6):
            angle = i * math.pi / 3 + math.pi / 6
            r = 3.2
            x = r * math.cos(angle)
            y = r * math.sin(angle)
            ang_deg = math.degrees(angle) + 90
            inner_data.append((x, y, ang_deg))
        return inner_data, 0.0, 0.0, 0.0

    def build_rectangular():
        """Build a rectangular 3x4 arrangement."""
        inner_data = []
        for r in range(4):
            for c in range(3):
                x = c * 2.1
                y = r * 1.75
                inner_data.append((x - 2.1, y - 3.5, 0.0))
        return inner_data, 0.0, 0.0, 0.0

    def build_hexagonal_cluster():
        """Build a central hexagon surrounded by 6, then 5 more around."""
        inner_data = []
        # Central
        inner_data.append((0.0, 0.0, 0.0))
        # First ring (6)
        for i in range(6):
            angle = i * math.pi / 3
            r = 2.0
            x = r * math.cos(angle)
            y = r * math.sin(angle)
            inner_data.append((x, y, 0.0))
        # Second ring (5) - skip one position to make 12 total
        for i in range(6):
            if i == 3: continue
            angle = i * math.pi / 3 + math.pi / 6
            r = 3.5
            x = r * math.cos(angle)
            y = r * math.sin(angle)
            inner_data.append((x, y, 0.0))
        
        # Center
        cx_mean = sum(d[0] for d in inner_data) / N
        cy_mean = sum(d[1] for d in inner_data) / N
        inner_data = [(x - cx_mean, y - cy_mean, a) for x, y, a in inner_data]
        return inner_data, 0.0, 0.0, 0.0

    def objective(params, inner_data_list, outer_cx, outer_cy, outer_angle_deg):
        """
        Objective function.
        params: [x0, y0, a0, ..., x11, y11, a11]
        Returns side length + penalty for overlaps.
        """
        inner_data = []
        for i in range(N):
            cx = params[3 * i]
            cy = params[3 * i + 1]
            ang = params[3 * i + 2]
            inner_data.append((cx, cy, ang))
        
        side = compute_outer_side(inner_data, outer_cx, outer_cy, outer_angle_deg)
        
        penalty = 0.0
        verts_list = [hex_vertices(cx, cy, ang) for cx, cy, ang in inner_data]
        for i in range(N):
            for j in range(i + 1, N):
                if not polygons_disjoint(verts_list[i], verts_list[j]):
                    penalty += 10.0
        
        return side + penalty

    def optimize_topology(inner_data, ocx, ocy, oang, max_iter=2000):
        """Run Nelder-Mead optimization on a given topology."""
        x0 = np.zeros(3 * N)
        for i in range(N):
            x0[3 * i] = inner_data[i][0]
            x0[3 * i + 1] = inner_data[i][1]
            x0[3 * i + 2] = inner_data[i][2]
        
        try:
            res = minimize(
                lambda p: objective(p, inner_data, ocx, ocy, oang),
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': max_iter,
                    'maxfev': max_iter * 2,
                    'xatol': 1e-4,
                    'fatol': 1e-4,
                    'adaptive': True
                }
            )
            
            inner_res = []
            for i in range(N):
                inner_res.append((res.x[3 * i], res.x[3 * i + 1], res.x[3 * i + 2]))
            
            side_res = compute_outer_side(inner_res, ocx, ocy, oang)
            
            # Check validity
            if check_validity(inner_res, ocx, ocy, oang, side_res * 1.001):
                return inner_res, ocx, ocy, oang, side_res
            else:
                return None
        except Exception:
            return None

    def polish_outer(inner_data, ocx, ocy, oang, current_side):
        """Refine outer hexagon parameters to minimize side length."""
        def outer_obj(params):
            ocx, ocy, oang = params
            return compute_outer_side(inner_data, ocx, ocy, oang)
        
        x0 = np.array([ocx, ocy, oang])
        try:
            res = minimize(
                outer_obj,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-5}
            )
            new_ocx, new_ocy, new_oang = res.x
            new_side = compute_outer_side(inner_data, new_ocx, new_ocy, new_oang)
            
            if check_validity(inner_data, new_ocx, new_ocy, new_oang, new_side * 1.001):
                return new_ocx, new_ocy, new_oang, new_side
            else:
                return ocx, ocy, oang, current_side
        except Exception:
            return ocx, ocy, oang, current_side

    # Main logic
    topologies = [
        build_honeycomb,
        build_ring,
        build_two_rings,
        build_rectangular,
        build_hexagonal_cluster,
    ]

    best_side = float('inf')
    best_result = None

    for top_idx, build_fn in enumerate(topologies):
        if time.time() > deadline - 10:
            break
        
        inner_data, ocx, ocy, oang = build_fn()
        
        # Initial side estimate
        side_est = compute_outer_side(inner_data, ocx, ocy, oang)
        
        # Check if initial topology is valid
        if check_validity(inner_data, ocx, ocy, oang, side_est * 1.001):
            if side_est < best_side:
                best_side = side_est
                best_result = (inner_data, ocx, ocy, oang)
        
        # Run optimization
        result = optimize_topology(inner_data, ocx, ocy, oang)
        if result is not None:
            inner_res, res_ocx, res_ocy, res_oang, side_res = result
            if side_res < best_side:
                best_side = side_res
                best_result = (inner_res, res_ocx, res_ocy, res_oang)

    # If we have a result, try to polish the outer hexagon
    if best_result is not None:
        inner_res, ocx, ocy, oang = best_result
        new_ocx, new_ocy, new_oang, new_side = polish_outer(inner_res, ocx, ocy, oang, best_side)
        if new_side < best_side:
            best_side = new_side
            best_result = (inner_res, new_ocx, new_ocy, new_oang)

    # Fallback if no valid result found
    if best_result is None:
        inner_data, ocx, ocy, oang = build_honeycomb()
        side_est = compute_outer_side(inner_data, ocx, ocy, oang) * 1.1
        best_result = (inner_data, ocx, ocy, oang)
        best_side = side_est

    inner_res, ocx, ocy, oang = best_result
    
    # Construct return value
    inner = []
    for cx, cy, ang in inner_res:
        inner.append([float(cx), float(cy), float(ang)])
    
    # Add small safety margin to side length
    final_side = best_side * 1.005 + 1e-4
    
    return inner, (float(ocx), float(ocy)), float(final_side), float(oang)
# EVOLVE-BLOCK-END
