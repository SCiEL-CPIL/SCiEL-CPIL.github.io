"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s limit

    def hex_vertices(cx, cy, angle_deg):
        """Return 6 vertices of a unit regular hexagon centered at (cx,cy) rotated by angle_deg."""
        angle_rad = math.radians(angle_deg)
        verts = np.zeros((6, 2))
        for i in range(6):
            a = angle_rad + i * math.pi / 3
            verts[i, 0] = cx + math.cos(a)
            verts[i, 1] = cy + math.sin(a)
        return verts

    def polygons_disjoint(v1, v2):
        """
        Check if two convex polygons are disjoint (non-overlapping) using SAT.
        Returns True if disjoint, False if overlapping.
        """
        for poly in [v1, v2]:
            for i in range(6):
                ax, ay = poly[i]
                bx, by = poly[(i + 1) % 6]
                # Edge vector
                ex, ey = bx - ax, by - ay
                # Normal (perpendicular)
                nx, ny = -ey, ex
                # Project both polygons onto this normal
                proj1 = np.array([dot(px - ax, py - ay, nx, ny) for px, py in v1])
                proj2 = np.array([dot(px - ax, py - ay, nx, ny) for px, py in v2])
                if np.max(proj1) < np.min(proj2) or np.max(proj2) < np.min(proj1):
                    return True  # Separating axis found
        return False  # No separating axis: polygons overlap

    def dot(x1, y1, x2, y2):
        return x1 * x2 + y1 * y2

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
                normal_angle = outer_angle_rad + k * math.pi / 3
                nx = math.cos(normal_angle)
                ny = math.sin(normal_angle)
                proj = rel_verts @ np.array([nx, ny])
                max_proj = max(max_proj, np.max(proj))
        
        # For a regular hexagon with side s, the apothem is s*sqrt(3)/2
        # max_proj is the maximum distance from center to any vertex of inner hexagons
        # projected onto the normals. The outer hexagon must have apothem >= max_proj
        # apothem = s * sqrt(3) / 2 => s = 2 * max_proj / sqrt(3)
        return 2.0 * max_proj / math.sqrt(3)

    def check_validity(inner_data, outer_cx, outer_cy, outer_angle_deg, side):
        """Check if all inner hexagons are inside the outer hexagon and non-overlapping."""
        # Check non-overlap
        verts_list = [hex_vertices(cx, cy, ang) for cx, cy, ang in inner_data]
        for i in range(N):
            for j in range(i + 1, N):
                if not polygons_disjoint(verts_list[i], verts_list[j]):
                    return False
        
        # Check containment
        outer_verts = hex_vertices(outer_cx, outer_cy, outer_angle_deg)
        # Scale outer hexagon to the given side length
        # The hex_vertices function returns side=1, so we need to scale
        scale = side
        outer_verts = outer_center + scale * (outer_verts - outer_center) if False else \
                      np.array([[outer_cx + scale * math.cos(math.radians(outer_angle_deg) + i * math.pi/3),
                                 outer_cy + scale * math.sin(math.radians(outer_angle_deg) + i * math.pi/3)] for i in range(6)])
        
        for i in range(N):
            for px, py in verts_list[i]:
                if not point_in_convex_polygon(px, py, outer_verts):
                    return False
        return True

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

    def objective(params):
        """
        Objective function for optimization.
        params: [x0, y0, a0, x1, y1, a1, ..., x11, y11, a11, outer_cx, outer_cy, outer_angle]
        """
        inner_data = []
        for i in range(N):
            cx = params[3 * i]
            cy = params[3 * i + 1]
            ang = params[3 * i + 2]
            inner_data.append((cx, cy, ang))
        
        outer_cx = params[36]
        outer_cy = params[37]
        outer_angle_deg = params[38]
        
        side = compute_outer_side(inner_data, outer_cx, outer_cy, outer_angle_deg)
        
        # Overlap penalty
        penalty = 0.0
        verts_list = [hex_vertices(cx, cy, ang) for cx, cy, ang in inner_data]
        for i in range(N):
            for j in range(i + 1, N):
                if not polygons_disjoint(verts_list[i], verts_list[j]):
                    # Compute overlap penalty using minimum translation vector
                    # Approximate: count as penalty of 1.0 per overlapping pair
                    penalty += 1.0
        
        return side + 100.0 * penalty

    def build_honeycomb():
        """Build a honeycomb-like arrangement of 12 hexagons."""
        s = 1.0
        dx = 2.0 * s
        dy = math.sqrt(3) * s
        offset = s
        
        centers = []
        # 4 rows of 3
        for r in range(4):
            for c in range(3):
                x = c * dx
                y = r * dy
                if r % 2 == 1:
                    x += offset
                centers.append((x, y))
        
        # Center at origin
        cx_mean = sum(c[0] for c in centers) / N
        cy_mean = sum(c[1] for c in centers) / N
        adjusted = [(x - cx_mean, y - cy_mean) for x, y in centers]
        
        inner_data = [(x, y, 0.0) for x, y in adjusted]
        return inner_data, 0.0, 0.0, 0.0

    def build_ring():
        """Build a ring of 12 hexagons around a central point."""
        inner_data = []
        # Place 12 hexagons in a ring
        for i in range(12):
            angle = i * math.pi / 6
            # Distance from center to allow hexagons to fit
            r = 2.5
            x = r * math.cos(angle)
            y = r * math.sin(angle)
            # Rotate each hexagon to face outward
            ang_deg = math.degrees(angle) + 90
            inner_data.append((x, y, ang_deg))
        return inner_data, 0.0, 0.0, 0.0

    def build_grid_3x4():
        """Build a 3x4 grid arrangement."""
        s = 1.0
        # For flat-top hexagons, spacing is different
        # Let's use pointy-top (angle 0)
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

    def build_two_rings():
        """Build two concentric rings: 6 inner, 6 outer."""
        inner_data = []
        # Inner ring: 6 hexagons
        for i in range(6):
            angle = i * math.pi / 3
            r = 1.8
            x = r * math.cos(angle)
            y = r * math.sin(angle)
            ang_deg = math.degrees(angle) + 90
            inner_data.append((x, y, ang_deg))
        # Outer ring: 6 hexagons
        for i in range(6):
            angle = i * math.pi / 3 + math.pi / 6
            r = 3.2
            x = r * math.cos(angle)
            y = r * math.sin(angle)
            ang_deg = math.degrees(angle) + 90
            inner_data.append((x, y, ang_deg))
        return inner_data, 0.0, 0.0, 0.0

    def build_rectangular():
        """Build a rectangular 3x4 arrangement with optimized spacing."""
        inner_data = []
        # 3 columns, 4 rows
        for r in range(4):
            for c in range(3):
                x = c * 2.1
                y = r * 1.75
                inner_data.append((x - 2.1, y - 3.5, 0.0))
        return inner_data, 0.0, 0.0, 0.0

    # Try multiple initial topologies
    topologies = [
        build_honeycomb,
        build_ring,
        build_grid_3x4,
        build_two_rings,
        build_rectangular,
    ]

    best_side = float('inf')
    best_result = None

    for top_idx, build_fn in enumerate(topologies):
        if time.time() > deadline - 10:
            break
        
        inner_data, ocx, ocy, oang = build_fn()
        
        # Initial side estimate
        side_est = compute_outer_side(inner_data, ocx, ocy, oang)
        
        # Set up initial parameters
        x0 = np.zeros(39)
        for i in range(N):
            x0[3 * i] = inner_data[i][0]
            x0[3 * i + 1] = inner_data[i][1]
            x0[3 * i + 2] = inner_data[i][2]
        x0[36] = ocx
        x0[37] = ocy
        x0[38] = oang
        
        # Run optimization
        try:
            res = minimize(
                objective,
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 5000,
                    'maxfev': 10000,
                    'xatol': 1e-5,
                    'fatol': 1e-5,
                    'adaptive': True
                }
            )
            
            # Extract result
            inner_res = []
            for i in range(N):
                inner_res.append((res.x[3 * i], res.x[3 * i + 1], res.x[3 * i + 2]))
            
            res_ocx = res.x[36]
            res_ocy = res.x[37]
            res_oang = res.x[38]
            
            side_res = compute_outer_side(inner_res, res_ocx, res_ocy, res_oang)
            
            # Check validity
            if check_validity(inner_res, res_ocx, res_ocy, res_oang, side_res * 1.001):
                if side_res < best_side:
                    best_side = side_res
                    best_result = (inner_res, res_ocx, res_ocy, res_oang)
        except Exception:
            pass
        
        # Also try the unoptimized topology
        if check_validity(inner_data, ocx, ocy, oang, side_est * 1.001):
            if side_est < best_side:
                best_side = side_est
                best_result = (inner_data, ocx, ocy, oang)

    # If no valid result found, use a safe fallback
    if best_result is None:
        # Use a large hexagon with hexagons placed in a grid
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
