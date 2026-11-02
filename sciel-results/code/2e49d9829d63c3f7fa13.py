import numpy as np
from scipy.optimize import minimize
import time

N = 12
SQRT3 = np.sqrt(3)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def hex_vertices(cx, cy, side, angle_rad):
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        x = cx + side * np.cos(angles)
        y = cy + side * np.sin(angles)
        return np.column_stack([x, y])

    def get_normals(verts):
        edges = np.roll(verts, -1, axis=0) - verts
        normals = np.column_stack([edges[:, 1], -edges[:, 0]])
        norms = np.linalg.norm(normals, axis=1)
        norms = np.where(norms < 1e-12, 1.0, norms)
        normals = normals / norms[:, np.newaxis]
        return normals

    def penetration_depth_fast(v1, v2, normals):
        p1 = v1 @ normals.T
        p2 = v2 @ normals.T
        min1 = p1.min(axis=0)
        max1 = p1.max(axis=0)
        min2 = p2.min(axis=0)
        max2 = p2.max(axis=0)
        overlaps = np.minimum(max1, max2) - np.maximum(min1, min2)
        if np.min(overlaps) <= 1e-9:
            return 0.0
        return np.min(overlaps)

    def inside_violation_fast(inner_verts, outer_verts, outer_normals):
        p_inner = inner_verts @ outer_normals.T
        p_outer_max = (outer_verts @ outer_normals.T).max(axis=0)
        p_inner_max = p_inner.max(axis=0)
        viol = p_inner_max - p_outer_max
        return max(0.0, np.max(viol))

    def objective(params, alpha_rad, theta_rad, unit_hex, inner_normals, outer_normals_cache):
        if time.time() > deadline:
            return 1e10

        positions = params[:2 * N].reshape(N, 2)
        s = params[2 * N]

        if s < 2.0:
            return 1e6

        outer_verts = hex_vertices(0, 0, s, theta_rad)
        
        # Update outer normals if s or theta changed? 
        # In this formulation, s is a variable, theta is fixed per run.
        # But outer_normals depends on theta only? No, normals are direction, independent of size.
        # Wait, get_normals returns unit normals. So they are independent of side length.
        # So outer_normals is constant for a fixed theta.
        # We can pass it in.
        
        total_penalty = 0.0
        inner_verts_list = []

        for i in range(N):
            verts = positions[i] + unit_hex
            inner_verts_list.append(verts)
            viol = inside_violation_fast(verts, outer_verts, outer_normals_cache)
            total_penalty += viol * 1000.0

        for i in range(N):
            for j in range(i + 1, N):
                pd = penetration_depth_fast(inner_verts_list[i], inner_verts_list[j], inner_normals)
                total_penalty += pd * 1000.0

        return s + total_penalty

    def make_honeycomb(rows, cols, flat_top):
        pos = []
        if flat_top:
            # Flat-top: angle 30 deg.
            # Horizontal spacing: sqrt(3)
            # Vertical spacing: 1.5
            # Shift: sqrt(3)/2
            dx = SQRT3
            dy = 1.5
            shift = SQRT3 / 2
            for r in range(rows):
                for c in range(cols):
                    x = c * dx
                    if r % 2 == 1:
                        x += shift
                    y = r * dy
                    pos.append([x, y])
        else:
            # Pointy-top: angle 0 deg.
            # Vertical spacing: sqrt(3)
            # Horizontal spacing: 1.5
            # Shift: 1.5/2 = 0.75
            dx = 1.5
            dy = SQRT3
            shift = 0.75
            for r in range(rows):
                for c in range(cols):
                    y = c * dy
                    if r % 2 == 1:
                        y += shift
                    x = r * dx
                    pos.append([x, y])
        
        pos = np.array(pos)
        center = pos.mean(axis=0)
        pos = pos - center
        return pos

    inits = []
    # 3x4 Flat-Top (Wide)
    pos_34_ft = make_honeycomb(3, 4, True)
    inits.append((pos_34_ft, np.radians(30), np.radians(0))) # Inner 30, Outer 0
    
    # 4x3 Pointy-Top (Tall)
    pos_43_pt = make_honeycomb(4, 3, False)
    inits.append((pos_43_pt, np.radians(0), np.radians(30))) # Inner 0, Outer 30

    best_s = np.inf
    best_params = None
    best_alpha = 0
    best_theta = 0

    for pos_init, alpha_init, theta_init in inits:
        if time.time() > deadline - 5:
            break
            
        # Estimate initial s
        # Bounding box of init
        x_min, x_max = pos_init[:, 0].min(), pos_init[:, 0].max()
        y_min, y_max = pos_init[:, 1].min(), pos_init[:, 1].max()
        # Add 1 for hex radius
        w = (x_max - x_min) + 2
        h = (y_max - y_min) + 2
        # Approximate side length needed
        init_s = max(w, h) * 0.6 + 1.0 
        # Safer:
        init_s = 4.0

        unit_hex = hex_vertices(0, 0, 1, alpha_init)
        inner_normals = get_normals(unit_hex)
        outer_normals = get_normals(hex_vertices(0, 0, 1, theta_init))
        
        x0 = np.concatenate([pos_init.flatten(), [init_s]])

        try:
            res = minimize(
                objective,
                x0,
                args=(alpha_init, theta_init, unit_hex, inner_normals, outer_normals),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-7, 'fatol': 1e-7, 'adaptive': True}
            )

            pos = res.x[:2 * N].reshape(N, 2)
            s = res.x[2 * N]

            # Verify
            outer_verts = hex_vertices(0, 0, s, theta_init)
            valid = True
            inner_verts_list = []
            for i in range(N):
                verts = pos[i] + unit_hex
                inner_verts_list.append(verts)
                if inside_violation_fast(verts, outer_verts, outer_normals) > 1e-4:
                    valid = False
                    break

            if valid:
                for i in range(N):
                    for j in range(i + 1, N):
                        if penetration_depth_fast(inner_verts_list[i], inner_verts_list[j], inner_normals) > 1e-4:
                            valid = False
                            break
                    if not valid:
                        break

            if valid and s < best_s:
                best_s = s
                best_params = res.x.copy()
                best_alpha = alpha_init
                best_theta = theta_init
                
        except Exception:
            pass

    if best_params is None:
        # Fallback
        best_s = 5.0
        best_params = np.concatenate([inits[0][0].flatten(), [best_s]])
        best_alpha = inits[0][1]
        best_theta = inits[0][2]

    final_pos = best_params[:2 * N].reshape(N, 2)
    final_s = best_params[2 * N]

    inner = [[final_pos[i, 0], final_pos[i, 1], np.degrees(best_alpha)] for i in range(N)]
    outer_center = (0.0, 0.0)
    outer_side = final_s
    outer_angle_deg = np.degrees(best_theta)

    return inner, outer_center, outer_side, outer_angle_deg
