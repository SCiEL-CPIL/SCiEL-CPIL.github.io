"""Structural change: Simulated Annealing with direct convex-hull containment,
using a different representation (positions + angles) and a different search
method (SA with multi-start and local polish)."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s limit

    def hexagon_vertices(cx, cy, side, angle_rad):
        """Return 6 vertices of a regular hexagon."""
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def convex_hull(points):
        """Compute convex hull of a set of 2D points using Graham scan."""
        points = np.array(points)
        if len(points) <= 3:
            return points
        # Find point with lowest y (and lowest x if tie)
        pivot = np.argmin(points[:, 1] + 1e-15 * points[:, 0])
        p0 = points[pivot]
        rest = np.delete(points, pivot, axis=0)
        rest = rest[np.lexsort((rest[:, 0], rest[:, 1]))]
        
        # Sort by polar angle
        def polar_angle(p):
            return math.atan2(p[1] - p0[1], p[0] - p0[0])
        
        rest = rest[np.argsort([polar_angle(p) for p in rest])]
        
        hull = [p0, rest[0]]
        for i in range(1, len(rest)):
            while len(hull) >= 2:
                a = hull[-2]
                b = hull[-1]
                c = rest[i]
                cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
                if cross <= 0:
                    hull.pop()
                else:
                    break
            hull.append(rest[i])
        
        return np.array(hull)

    def min_containing_hex_side(hull_pts, outer_angle_rad):
        """Minimum side length of a regular hexagon (with given rotation) 
        centered at origin that contains all hull points."""
        theta = outer_angle_rad
        max_required = 0.0
        for v in hull_pts:
            dx, dy = v[0], v[1]
            for k in range(6):
                edge_angle = theta + np.radians(k * 60)
                nx = np.cos(edge_angle)
                ny = np.sin(edge_angle)
                dist = dx * nx + dy * ny
                if dist > 0:
                    required_S = dist * 2.0 / np.sqrt(3.0)
                    if required_S > max_required:
                        max_required = required_S
        return max_required

    def min_containing_hex(hull_pts):
        """Find minimum side over all outer hexagon rotations."""
        # Try 12 angles (symmetry of hexagon means 60° period, but we try finer)
        best_side = np.inf
        best_angle = 0.0
        for a in np.linspace(0, np.pi / 6, 13):
            s = min_containing_hex_side(hull_pts, a)
            if s < best_side:
                best_side = s
                best_angle = a
        return best_side, best_angle

    def sat_disjoint(poly1, poly2, tol=1e-9):
        """Check if two convex polygons are disjoint using SAT."""
        for poly in [poly1, poly2]:
            for i in range(len(poly)):
                v1 = poly[i]
                v2 = poly[(i + 1) % len(poly)]
                edge = v2 - v1
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm < 1e-10:
                    continue
                normal = normal / norm
                proj1 = np.dot(poly1, normal)
                proj2 = np.dot(poly2, normal)
                if np.max(proj1) < np.min(proj2) - tol or np.max(proj2) < np.min(poly1) - tol:
                    return True
        return False

    def check_all_disjoint(inner_hexes, tol=1e-8):
        """Check if all inner hexagons are pairwise disjoint."""
        for i in range(N):
            for j in range(i + 1, N):
                if not sat_disjoint(inner_hexes[i], inner_hexes[j], tol):
                    return False
        return True

    def objective(params):
        """Objective: outer hexagon side length + penalty for overlaps."""
        if time.time() > deadline:
            return 1e6
        inner_x = params[0:12]
        inner_y = params[12:24]
        inner_a = params[24:36]
        
        inner_hexes = []
        all_verts = []
        for i in range(N):
            verts = hexagon_vertices(inner_x[i], inner_y[i], 1.0, inner_a[i])
            inner_hexes.append(verts)
            all_verts.append(verts)
        
        all_verts = np.vstack(all_verts)
        hull = convex_hull(all_verts)
        
        side, _ = min_containing_hex(hull)
        
        # Penalty for overlaps
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                if not sat_disjoint(inner_hexes[i], inner_hexes[j], -0.01):
                    # Overlapping: compute minimum distance violation
                    # Use a simple proxy: if not disjoint with negative tolerance
                    penalty += 10.0
        return side + penalty

    def make_initial_honeycomb(scale=1.0, rot_offset=0.0):
        """Create a honeycomb-like initial configuration."""
        sqrt3 = math.sqrt(3)
        hx, hy, ha = [], [], []
        for r in range(3):
            for c in range(4):
                x = c * sqrt3 * 1.05 * scale
                y = r * 1.55 * scale
                if r % 2 == 1:
                    x += sqrt3 * 1.05 * scale / 2
                hx.append(x)
                hy.append(y)
                ha.append(rot_offset)
        hx = np.array(hx)
        hy = np.array(hy)
        ha = np.array(ha)
        hx -= np.mean(hx)
        hy -= np.mean(hy)
        return np.concatenate([hx, hy, ha])

    def make_initial_ring():
        """Create a ring-like initial configuration."""
        cx, cy, ca = [], [], []
        for i in range(6):
            angle = 2 * np.pi * i / 6
            r = 2.0
            cx.append(r * np.cos(angle))
            cy.append(r * np.sin(angle))
            ca.append(angle)
        for i in range(6):
            angle = 2 * np.pi * i / 6 + np.pi / 6
            r = 3.5
            cx.append(r * np.cos(angle))
            cy.append(r * np.sin(angle))
            ca.append(angle)
        cx = np.array(cx)
        cy = np.array(cy)
        ca = np.array(ca)
        cx -= np.mean(cx)
        cy -= np.mean(cy)
        return np.concatenate([cx, cy, ca])

    # Generate diverse initial configurations
    np.random.seed(42)
    configs = []
    
    # Honeycomb variants
    configs.append(make_initial_honeycomb(1.0, 0.0))
    configs.append(make_initial_honeycomb(1.0, np.pi / 6))
    configs.append(make_initial_honeycomb(1.05, 0.0))
    configs.append(make_initial_honeycomb(1.1, np.pi / 12))
    
    # Ring
    configs.append(make_initial_ring())
    
    # Random
    for _ in range(3):
        rx = np.random.uniform(-4, 4, 12)
        ry = np.random.uniform(-4, 4, 12)
        ra = np.random.uniform(-np.pi, np.pi, 12)
        rx -= np.mean(rx)
        ry -= np.mean(ry)
        configs.append(np.concatenate([rx, ry, ra]))

    # Evaluate all configs
    best_params = configs[0].copy()
    best_obj = objective(best_params)
    for cfg in configs:
        obj = objective(cfg)
        if obj < best_obj:
            best_obj = obj
            best_params = cfg.copy()

    # Simulated Annealing
    current_params = best_params.copy()
    current_obj = best_obj
    T = 0.5  # Initial temperature
    T_min = 0.001
    cooling = 0.9995
    max_iters = 50000
    accepted = 0
    improved = 0

    for it in range(max_iters):
        if time.time() > deadline - 10:
            break
        if T < T_min:
            break

        # Perturb
        step = 0.1 * (T / 0.5 + 0.1)
        new_params = current_params + np.random.normal(0, step, len(current_params))
        
        # Keep angles in reasonable range
        for i in range(12):
            new_params[24 + i] = np.arctan2(np.sin(new_params[24 + i]), np.cos(new_params[24 + i]))

        new_obj = objective(new_params)

        if new_obj < current_obj:
            current_params = new_params
            current_obj = new_obj
            accepted += 1
            improved += 1
            if current_obj < best_obj:
                best_obj = current_obj
                best_params = current_params.copy()
        else:
            delta = new_obj - current_obj
            if delta < 0 or np.random.random() < math.exp(-delta / T):
                current_params = new_params
                current_obj = new_obj
                accepted += 1

        T *= cooling

        # Periodic local polish
        if it % 5000 == 0 and it > 0 and time.time() < deadline - 20:
            try:
                bounds = [(-6, 6)] * 12 + [(-6, 6)] * 12 + [(-np.pi, np.pi)] * 12
                result = minimize(
                    objective, current_params, method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6}
                )
                if result.fun < current_obj:
                    current_params = result.x.copy()
                    current_obj = result.fun
                    if current_obj < best_obj:
                        best_obj = current_obj
                        best_params = current_params.copy()
            except Exception:
                pass

    # Final local polish from best
    if time.time() < deadline - 15:
        try:
            bounds = [(-6, 6)] * 12 + [(-6, 6)] * 12 + [(-np.pi, np.pi)] * 12
            result = minimize(
                objective, best_params, method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
            )
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x.copy()
        except Exception:
            pass

    # Nelder-Mead polish
    if time.time() < deadline - 10:
        try:
            bounds = [(-6, 6)] * 12 + [(-6, 6)] * 12 + [(-np.pi, np.pi)] * 12
            result = minimize(
                objective, best_params, method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True}
            )
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x.copy()
        except Exception:
            pass

    # Extract final parameters
    params = best_params
    inner_x = params[0:12]
    inner_y = params[12:24]
    inner_a = params[24:36]

    inner_hexes = []
    all_verts = []
    for i in range(N):
        verts = hexagon_vertices(inner_x[i], inner_y[i], 1.0, inner_a[i])
        inner_hexes.append(verts)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    hull = convex_hull(all_verts)

    # Validate disjointness
    valid = check_all_disjoint(inner_hexes)

    if not valid:
        # Try to fix by scaling up
        for scale in [1.05, 1.1, 1.15, 1.2]:
            fixed_params = params.copy()
            for i in range(12):
                fixed_params[i] *= scale
                fixed_params[i + 12] *= scale
            fx = fixed_params[0:12]
            fy = fixed_params[12:24]
            fa = fixed_params[24:36]
            fhexes = []
            for i in range(N):
                verts = hexagon_vertices(fx[i], fy[i], 1.0, fa[i])
                fhexes.append(verts)
            if check_all_disjoint(fhexes):
                params = fixed_params
                inner_x = fx
                inner_y = fy
                inner_a = fa
                inner_hexes = fhexes
                all_verts = np.vstack(fhexes)
                hull = convex_hull(all_verts)
                valid = True
                break

    if not valid:
        # Fallback: widely spaced
        inner = [[3.5 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 35.0, 0.0

    # Compute exact outer side
    side, angle = min_containing_hex(hull)
    # Center is at origin for the hull computation
    outer_cx, outer_cy = 0.0, 0.0

    inner = [[inner_x[i], inner_y[i], math.degrees(inner_a[i])] for i in range(N)]
    return inner, (outer_cx, outer_cy), side, math.degrees(angle)
# EVOLVE-BLOCK-END
