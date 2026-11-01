"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    def hexagon_vertices(cx, cy, side, angle_rad):
        """Vertices of a regular hexagon (flat-top when angle=0)."""
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def point_to_edge_distance(p, e1, e2):
        """Distance from point p to edge e1-e2 (positive if outside)."""
        d = e2 - e1
        L2 = np.dot(d, d)
        if L2 < 1e-12:
            return np.linalg.norm(p - e1)
        t = np.dot(p - e1, d) / L2
        t = max(0.0, min(1.0, t))
        closest = e1 + t * d
        return np.linalg.norm(p - closest)

    def min_vertex_edge_distance(hex1_verts, hex2_verts):
        """Minimum distance from vertices of hex1 to edges of hex2 and vice versa.
        Negative if overlapping."""
        min_dist = np.inf
        # Check vertices of hex1 against edges of hex2
        for v in hex1_verts:
            for i in range(6):
                e1 = hex2_verts[i]
                e2 = hex2_verts[(i + 1) % 6]
                d = point_to_edge_distance(v, e1, e2)
                if d < min_dist:
                    min_dist = d
        # Check vertices of hex2 against edges of hex1
        for v in hex2_verts:
            for i in range(6):
                e1 = hex1_verts[i]
                e2 = hex1_verts[(i + 1) % 6]
                d = point_to_edge_distance(v, e1, e2)
                if d < min_dist:
                    min_dist = d
        return min_dist

    def polygons_disjoint_sat(poly1, poly2, tol=1e-9):
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
                
                min1, max1 = np.min(proj1), np.max(proj1)
                min2, max2 = np.min(proj2), np.max(proj2)
                
                if max1 < min2 - tol or max2 < min1 - tol:
                    return True
        return False

    def compute_outer_side(inner_hexes, outer_cx, outer_cy, outer_angle_rad):
        """Compute minimum outer hexagon side length to contain all inner hexagons."""
        max_required = 0.0
        theta = outer_angle_rad
        for hex_verts in inner_hexes:
            for v in hex_verts:
                dx = v[0] - outer_cx
                dy = v[1] - outer_cy
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

    def make_hexes(params):
        """Build inner hexagon vertices from params."""
        inner_x = params[0:12]
        inner_y = params[12:24]
        inner_a = params[24:36]
        outer_cx = params[36]
        outer_cy = params[37]
        outer_a = params[38]
        inner_hexes = []
        for i in range(N):
            verts = hexagon_vertices(inner_x[i], inner_y[i], 1.0, inner_a[i])
            inner_hexes.append(verts)
        return inner_hexes, outer_cx, outer_cy, outer_a

    def objective(params):
        if time.time() > deadline:
            return 1e6
        
        inner_hexes, outer_cx, outer_cy, outer_a = make_hexes(params)
        
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dist = min_vertex_edge_distance(inner_hexes[i], inner_hexes[j])
                if dist < 0:
                    # Smooth penalty for overlap
                    penalty += 100.0 * dist * dist
        
        side = compute_outer_side(inner_hexes, outer_cx, outer_cy, outer_a)
        return side + penalty

    # Generate diverse initial configurations
    np.random.seed(42)
    configs = []
    sqrt3 = np.sqrt(3)

    # Config 1: Honeycomb lattice (3x4 staggered) - well-separated
    hx, hy, ha = [], [], []
    for r in range(3):
        for c in range(4):
            x = c * sqrt3 * 1.05
            y = r * 1.55
            if r % 2 == 1:
                x += sqrt3 * 1.05 / 2
            hx.append(x); hy.append(y); ha.append(0.0)
    hx = np.array(hx); hy = np.array(hy); ha = np.array(ha)
    hx -= np.mean(hx); hy -= np.mean(hy)
    configs.append(np.concatenate([hx, hy, ha, [0.0, 0.0, 0.0]]))

    # Config 2: Honeycomb with 30-degree rotation
    hx2, hy2, ha2 = [], [], []
    for r in range(3):
        for c in range(4):
            x = c * sqrt3 * 1.05
            y = r * 1.55
            if r % 2 == 1:
                x += sqrt3 * 1.05 / 2
            hx2.append(x); hy2.append(y); ha2.append(np.pi / 6)
    hx2 = np.array(hx2); hy2 = np.array(hy2); ha2 = np.array(ha2)
    hx2 -= np.mean(hx2); hy2 -= np.mean(hy2)
    configs.append(np.concatenate([hx2, hy2, ha2, [0.0, 0.0, np.pi / 6]]))

    # Config 3: Compact circular arrangement
    cx, cy, ca = [], [], []
    for i in range(12):
        angle = 2 * np.pi * i / 12
        r = 2.8
        cx.append(r * np.cos(angle))
        cy.append(r * np.sin(angle))
        ca.append(angle)
    cx = np.array(cx); cy = np.array(cy); ca = np.array(ca)
    cx -= np.mean(cx); cy -= np.mean(cy)
    configs.append(np.concatenate([cx, cy, ca, [0.0, 0.0, 0.0]]))

    # Config 4: Two rows of 6 with spacing
    rx, ry, ra = [], [], []
    for i in range(6):
        rx.append(i * sqrt3 * 1.05 - 2.5 * sqrt3 * 1.05)
        ry.append(0.0)
        ra.append(0.0)
        rx.append(i * sqrt3 * 1.05 - 2.5 * sqrt3 * 1.05)
        ry.append(1.55)
        ra.append(0.0)
    rx = np.array(rx); ry = np.array(ry); ra = np.array(ra)
    rx -= np.mean(rx); ry -= np.mean(ry)
    configs.append(np.concatenate([rx, ry, ra, [0.0, 0.0, 0.0]]))

    # Config 5: Random perturbation of honeycomb
    rx5, ry5, ra5 = [], [], []
    for r in range(3):
        for c in range(4):
            x = c * sqrt3 * 1.05 + np.random.uniform(-0.2, 0.2)
            y = r * 1.55 + np.random.uniform(-0.2, 0.2)
            if r % 2 == 1:
                x += sqrt3 * 1.05 / 2
            rx5.append(x); ry5.append(y); ra5.append(np.random.uniform(-0.1, 0.1))
    rx5 = np.array(rx5); ry5 = np.array(ry5); ra5 = np.array(ra5)
    rx5 -= np.mean(rx5); ry5 -= np.mean(ry5)
    configs.append(np.concatenate([rx5, ry5, ra5, [0.0, 0.0, 0.0]]))

    # Evaluate all configs and pick best
    best_params = configs[0].copy()
    best_obj = objective(best_params)
    for cfg in configs:
        obj = objective(cfg)
        if obj < best_obj:
            best_obj = obj
            best_params = cfg.copy()

    # Phase 1: Differential Evolution for global search
    bounds = [(-6, 6)] * 12 + [(-6, 6)] * 12 + [(-np.pi, np.pi)] * 12 + [(-1, 1), (-1, 1), (-np.pi / 6, np.pi / 6)]
    
    de_result = None
    try:
        de_result = differential_evolution(
            objective, bounds,
            seed=42, maxiter=300, popsize=40,
            tol=1e-7, mut=(0.4, 1.0), recomb=0.8,
            polish=False, init='sohal'
        )
        if de_result.fun < best_obj:
            best_obj = de_result.fun
            best_params = de_result.x
    except Exception:
        pass

    # Phase 2: Local refinement with Nelder-Mead from best DE result
    if time.time() < deadline - 20:
        try:
            result = minimize(
                objective, best_params, method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True}
            )
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x
        except Exception:
            pass

    # Phase 3: Additional local refinements from perturbed best
    if time.time() < deadline - 15:
        for trial in range(5):
            if time.time() > deadline - 10:
                break
            perturbed = best_params + np.random.normal(0, 0.05, len(best_params))
            try:
                result = minimize(
                    objective, perturbed, method='Nelder-Mead',
                    options={'maxiter': 1500, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True}
                )
                if result.fun < best_obj:
                    best_obj = result.fun
                    best_params = result.x
            except Exception:
                pass

    # Extract solution
    params = best_params
    inner_x = params[0:12]
    inner_y = params[12:24]
    inner_a = params[24:36]
    outer_cx = params[36]
    outer_cy = params[37]
    outer_a = params[38]
    
    inner = [[inner_x[i], inner_y[i], np.degrees(inner_a[i])] for i in range(N)]
    
    # Verify validity
    inner_hexes = []
    for i in range(N):
        verts = hexagon_vertices(inner_x[i], inner_y[i], 1.0, inner_a[i])
        inner_hexes.append(verts)
    
    valid = True
    for i in range(N):
        for j in range(i + 1, N):
            if not polygons_disjoint_sat(inner_hexes[i], inner_hexes[j], 1e-7):
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to a known valid configuration with wider spacing
        inner = [[3.5 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 35.0, 0.0
    
    # Recompute exact side length without penalty
    exact_side = compute_outer_side(inner_hexes, outer_cx, outer_cy, outer_a)
    
    return inner, (outer_cx, outer_cy), exact_side, np.degrees(outer_a)
# EVOLVE-BLOCK-END
