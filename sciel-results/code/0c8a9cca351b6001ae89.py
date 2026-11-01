"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

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

    def sat_separation_distance(hex1_verts, hex2_verts):
        """Compute the minimum separation distance using SAT.
        Returns negative if overlapping, positive if separated."""
        min_gap = np.inf
        
        for poly in [hex1_verts, hex2_verts]:
            for i in range(6):
                v1 = poly[i]
                v2 = poly[(i + 1) % 6]
                edge = v2 - v1
                # Normal pointing outward from poly
                normal = np.array([edge[1], -edge[0]])
                norm = np.linalg.norm(normal)
                if norm < 1e-10:
                    continue
                normal = normal / norm
                
                proj1 = np.dot(hex1_verts, normal)
                proj2 = np.dot(hex2_verts, normal)
                
                # Determine which poly is on which side
                center1 = np.mean(hex1_verts, axis=0)
                center2 = np.mean(hex2_verts, axis=0)
                side1 = np.dot(center1 - center2, normal)
                
                if side1 > 0:
                    # hex1 is on positive side, need max of hex1 >= min of hex2
                    gap = np.min(proj2) - np.max(proj1)
                else:
                    # hex1 is on negative side, need min of hex1 <= max of hex2
                    gap = np.min(proj1) - np.max(proj2)
                
                if gap < min_gap:
                    min_gap = gap
        
        return min_gap

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

    def objective(params):
        if time.time() > deadline:
            return 1e6
        
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
        
        # Smooth penalty for overlap using SAT separation
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                sep = sat_separation_distance(inner_hexes[i], inner_hexes[j])
                if sep < 0:
                    # Use smooth quadratic penalty
                    penalty += 50.0 * sep * sep
        
        side = compute_outer_side(inner_hexes, outer_cx, outer_cy, outer_a)
        return side + penalty

    def objective_with_grad(params):
        """Objective function for gradient-based optimization."""
        return objective(params)

    # Generate diverse initial configurations
    np.random.seed(42)
    configs = []
    sqrt3 = np.sqrt(3)

    # Config 1: Honeycomb lattice (3x4 staggered)
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

    # Phase 1: L-BFGS-B optimization with multiple restarts
    bounds = [(-6, 6)] * 12 + [(-6, 6)] * 12 + [(-np.pi, np.pi)] * 12 + [(-1, 1), (-1, 1), (-np.pi / 6, np.pi / 6)]
    
    # Try L-BFGS-B from best config
    if time.time() < deadline - 30:
        try:
            result = minimize(
                objective, best_params, method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 5000, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
            )
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x
        except Exception:
            pass

    # Phase 2: Nelder-Mead refinement
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

    # Phase 3: Multiple local refinements from perturbed best
    if time.time() < deadline - 15:
        for trial in range(8):
            if time.time() > deadline - 10:
                break
            perturbation_scale = 0.03 + 0.02 * trial
            perturbed = best_params + np.random.normal(0, perturbation_scale, len(best_params))
            try:
                result = minimize(
                    objective, perturbed, method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True}
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
    
    # Verify validity using SAT
    inner_hexes = []
    for i in range(N):
        verts = hexagon_vertices(inner_x[i], inner_y[i], 1.0, inner_a[i])
        inner_hexes.append(verts)
    
    valid = True
    for i in range(N):
        for j in range(i + 1, N):
            sep = sat_separation_distance(inner_hexes[i], inner_hexes[j])
            if sep < -1e-6:  # Allow tiny numerical tolerance
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
