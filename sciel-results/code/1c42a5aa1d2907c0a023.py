"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s limit

    def hexagon_vertices(cx, cy, side, angle_rad):
        """Vertices of a regular hexagon (flat-top when angle=0)."""
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

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

    def min_vertex_edge_distance(hex1_verts, hex2_verts):
        """Minimum distance from vertices of hex1 to edges of hex2 and vice versa.
        Negative if overlapping."""
        min_dist = np.inf
        for v in hex1_verts:
            for i in range(6):
                e1 = hex2_verts[i]
                e2 = hex2_verts[(i + 1) % 6]
                d = e2 - e1
                L2 = np.dot(d, d)
                if L2 < 1e-12:
                    dist = np.linalg.norm(v - e1)
                else:
                    t = np.dot(v - e1, d) / L2
                    t = max(0.0, min(1.0, t))
                    closest = e1 + t * d
                    dist = np.linalg.norm(v - closest)
                if dist < min_dist:
                    min_dist = dist
        for v in hex2_verts:
            for i in range(6):
                e1 = hex1_verts[i]
                e2 = hex1_verts[(i + 1) % 6]
                d = e2 - e1
                L2 = np.dot(d, d)
                if L2 < 1e-12:
                    dist = np.linalg.norm(v - e1)
                else:
                    t = np.dot(v - e1, d) / L2
                    t = max(0.0, min(1.0, t))
                    closest = e1 + t * d
                    dist = np.linalg.norm(v - closest)
                if dist < min_dist:
                    min_dist = dist
        return min_dist

    def smooth_penalty(hexes, strength=50.0, eps=0.1):
        """Smooth L1-like penalty for overlaps between hexagons."""
        penalty = 0.0
        for i in range(len(hexes)):
            for j in range(i + 1, len(hexes)):
                d = min_vertex_edge_distance(hexes[i], hexes[j])
                if d < 0:
                    # Smooth L1: for d < 0, penalty grows quadratically
                    penalty += strength * d * d
        return penalty

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
        """Objective: outer side length + smooth overlap penalty."""
        if time.time() > deadline:
            return 1e6
        inner_hexes, outer_cx, outer_cy, outer_a = make_hexes(params)
        side = compute_outer_side(inner_hexes, outer_cx, outer_cy, outer_a)
        pen = smooth_penalty(inner_hexes)
        return side + pen

    def objective_strict(params):
        """Strict objective: returns large value if any overlap, else side length."""
        if time.time() > deadline:
            return 1e6
        inner_hexes, outer_cx, outer_cy, outer_a = make_hexes(params)
        # Check for any overlap
        for i in range(N):
            for j in range(i + 1, N):
                d = min_vertex_edge_distance(inner_hexes[i], inner_hexes[j])
                if d < -1e-7:
                    return 1e6
        return compute_outer_side(inner_hexes, outer_cx, outer_cy, outer_a)

    sqrt3 = np.sqrt(3)

    # Generate diverse initial configurations
    configs = []

    # Config 1: Honeycomb lattice (4x3 staggered)
    hx, hy, ha = [], [], []
    for r in range(3):
        for c in range(4):
            x = c * sqrt3 * 1.0
            y = r * 1.5
            if r % 2 == 1:
                x += sqrt3 * 1.0 / 2
            hx.append(x); hy.append(y); ha.append(0.0)
    hx = np.array(hx); hy = np.array(hy); ha = np.array(ha)
    hx -= np.mean(hx); hy -= np.mean(hy)
    configs.append(np.concatenate([hx, hy, ha, [0.0, 0.0, 0.0]]))

    # Config 2: Two rows of 6
    rx, ry, ra = [], [], []
    for i in range(6):
        rx.append(i * sqrt3 - 2.5 * sqrt3)
        ry.append(0.0)
        ra.append(0.0)
    for i in range(6):
        rx.append((i + 0.5) * sqrt3 - 2.5 * sqrt3)
        ry.append(1.5)
        ra.append(0.0)
    rx = np.array(rx); ry = np.array(ry); ra = np.array(ra)
    rx -= np.mean(rx); ry -= np.mean(ry)
    configs.append(np.concatenate([rx, ry, ra, [0.0, 0.0, 0.0]]))

    # Config 3: 3 rows of 4
    rx3, ry3, ra3 = [], [], []
    for r in range(3):
        for i in range(4):
            rx3.append(i * sqrt3 - 1.5 * sqrt3)
            ry3.append(r * 1.5 - 1.5)
            ra3.append(0.0)
    rx3 = np.array(rx3); ry3 = np.array(ry3); ra3 = np.array(ra3)
    rx3 -= np.mean(rx3); ry3 -= np.mean(ry3)
    configs.append(np.concatenate([rx3, ry3, ra3, [0.0, 0.0, 0.0]]))

    # Config 4: Circular arrangement
    cx, cy, ca = [], [], []
    for i in range(12):
        angle = 2 * np.pi * i / 12
        r = 3.0
        cx.append(r * np.cos(angle))
        cy.append(r * np.sin(angle))
        ca.append(angle)
    cx = np.array(cx); cy = np.array(cy); ca = np.array(ca)
    cx -= np.mean(cx); cy -= np.mean(cy)
    configs.append(np.concatenate([cx, cy, ca, [0.0, 0.0, 0.0]]))

    # Config 5: Compact honeycomb with 30-degree rotation
    hx5, hy5, ha5 = [], [], []
    for r in range(3):
        for c in range(4):
            x = c * sqrt3 * 1.0
            y = r * 1.5
            if r % 2 == 1:
                x += sqrt3 * 1.0 / 2
            hx5.append(x); hy5.append(y); ha5.append(np.pi / 6)
    hx5 = np.array(hx5); hy5 = np.array(hy5); ha5 = np.array(ha5)
    hx5 -= np.mean(hx5); hy5 -= np.mean(hy5)
    configs.append(np.concatenate([hx5, hy5, ha5, [0.0, 0.0, np.pi / 6]]))

    # Evaluate all configs and pick best
    best_params = configs[0].copy()
    best_obj = objective(best_params)
    for cfg in configs:
        obj = objective(cfg)
        if obj < best_obj:
            best_obj = obj
            best_params = cfg.copy()

    # Phase 1: Differential Evolution for global search
    bounds = [(-5, 5)] * 12 + [(-5, 5)] * 12 + [(-np.pi, np.pi)] * 12 + [(-1, 1), (-1, 1), (-np.pi / 6, np.pi / 6)]

    de_result = None
    try:
        remaining = deadline - time.time()
        if remaining > 30:
            de_result = differential_evolution(
                objective, bounds,
                maxiter=30, popsize=15,
                tol=1e-5, mutation=(0.5, 1.0),
                recombination=0.7, seed=42,
                polish=False,
                update_interval=1
            )
            if de_result.fun < best_obj:
                best_obj = de_result.fun
                best_params = de_result.x.copy()
    except Exception:
        pass

    # Phase 2: Local refinement with L-BFGS-B using smooth objective
    if time.time() < deadline - 10:
        try:
            res = minimize(
                objective, best_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
            )
            if res.fun < best_obj:
                best_obj = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Phase 3: Another L-BFGS-B pass with tighter tolerance if time permits
    if time.time() < deadline - 5:
        try:
            res = minimize(
                objective, best_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 30}
            )
            if res.fun < best_obj:
                best_obj = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Extract best solution
    params = best_params
    inner_x = params[0:12]
    inner_y = params[12:24]
    inner_a = params[24:36]
    outer_cx = params[36]
    outer_cy = params[37]
    outer_a = params[38]

    # Verify the solution is valid (no overlaps)
    inner_hexes = []
    for i in range(N):
        verts = hexagon_vertices(inner_x[i], inner_y[i], 1.0, inner_a[i])
        inner_hexes.append(verts)

    valid = True
    for i in range(N):
        for j in range(i + 1, N):
            d = min_vertex_edge_distance(inner_hexes[i], inner_hexes[j])
            if d < -1e-6:
                valid = False
                break
        if not valid:
            break

    if not valid:
        # Try to fix by using a larger outer hexagon or falling back
        # Use the best valid config from initial set
        for cfg in configs:
            chexes, ccx, ccy, ca = make_hexes(cfg)
            cvalid = True
            for i in range(N):
                for j in range(i + 1, N):
                    d = min_vertex_edge_distance(chexes[i], chexes[j])
                    if d < -1e-6:
                        cvalid = False
                        break
                if not cvalid:
                    break
            if cvalid:
                params = cfg
                break

    params = best_params
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

    # Final check
    valid = True
    for i in range(N):
        for j in range(i + 1, N):
            d = min_vertex_edge_distance(inner_hexes[i], inner_hexes[j])
            if d < -1e-6:
                valid = False
                break
        if not valid:
            break

    if not valid:
        # Fallback: use initial honeycomb with large outer
        hx, hy, ha = [], [], []
        for r in range(3):
            for c in range(4):
                x = c * sqrt3 * 1.5
                y = r * 2.0
                if r % 2 == 1:
                    x += sqrt3 * 1.5 / 2
                hx.append(x); hy.append(y); ha.append(0.0)
        inner = [[hx[i], hy[i], np.degrees(ha[i])] for i in range(N)]
        return inner, (0.0, 0.0), 20.0, 0.0

    outer_side = compute_outer_side(inner_hexes, outer_cx, outer_cy, outer_a)

    inner = [[inner_x[i], inner_y[i], np.degrees(inner_a[i])] for i in range(N)]
    return inner, (outer_cx, outer_cy), outer_side, np.degrees(outer_a)
# EVOLVE-BLOCK-END
