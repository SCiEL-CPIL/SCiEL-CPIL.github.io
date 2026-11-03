import numpy as np
from scipy.optimize import differential_evolution, minimize
import time
import math

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 148.0

    # --- Geometry Helpers ---

    def hex_vertices(cx, cy, side, angle_rad):
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        x = cx + side * np.cos(angles)
        y = cy + side * np.sin(angles)
        return np.column_stack([x, y])

    def get_normals(verts):
        edges = np.roll(verts, -1, axis=0) - verts
        normals = np.column_stack([edges[:, 1], -edges[:, 0]])
        norms = np.linalg.norm(normals, axis=1)
        normals = normals / (norms[:, np.newaxis] + 1e-15)
        return normals

    def sat_overlap(v1, v2):
        normals1 = get_normals(v1)
        normals2 = get_normals(v2)
        normals = np.vstack([normals1, normals2])

        p1 = v1 @ normals.T
        p2 = v2 @ normals.T

        min1 = p1.min(axis=0)
        max1 = p1.max(axis=0)
        min2 = p2.min(axis=0)
        max2 = p2.max(axis=0)

        overlaps = np.minimum(max1, max2) - np.maximum(min1, min2)
        return float(np.min(overlaps))

    def smooth_max(values, temperature=0.01):
        if len(values) == 0:
            return 0.0
        v_max = np.max(values)
        if v_max == -np.inf:
            return -np.inf
        shifted = (values - v_max) / temperature
        shifted = np.clip(shifted, -50, 50)
        result = v_max + temperature * np.log(np.sum(np.exp(shifted)))
        return result

    def get_hexes_symmetry(params):
        # 6-fold symmetry: params = [r1, theta1, alpha1, r2, theta2, alpha2]
        r1, th1, al1, r2, th2, al2 = params
        hexes = []
        for k in range(6):
            angle = np.radians(60 * k)
            x1 = r1 * np.cos(th1 + angle)
            y1 = r1 * np.sin(th1 + angle)
            a1 = al1 + angle
            hexes.append(hex_vertices(x1, y1, 1.0, a1))
            x2 = r2 * np.cos(th2 + angle)
            y2 = r2 * np.sin(th2 + angle)
            a2 = al2 + angle
            hexes.append(hex_vertices(x2, y2, 1.0, a2))
        return hexes

    def compute_min_side_fixed(hexes, outer_angle_rad=0.0):
        normal_angles = outer_angle_rad + np.radians([30, 90, 150, 210, 270, 330])
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        max_proj = 0.0
        for h in hexes:
            proj = h @ normals.T
            max_proj = max(max_proj, float(np.max(proj)))
        return max_proj * 2.0 / np.sqrt(3.0)

    def compute_min_side_smooth(hexes, temperature=0.01, outer_angle_rad=0.0):
        normal_angles = outer_angle_rad + np.radians([30, 90, 150, 210, 270, 330])
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        all_projs = []
        for h in hexes:
            proj = h @ normals.T
            all_projs.append(proj)
        all_projs = np.concatenate(all_projs, axis=0)
        smooth_maxes = np.array([smooth_max(all_projs[:, i], temperature) for i in range(6)])
        max_smooth = smooth_max(smooth_maxes, temperature)
        return max_smooth * 2.0 / np.sqrt(3.0)

    def compute_objective_symmetry(params, temperature=0.01, penalty_weight=50.0, outer_angle_rad=0.0):
        if time.time() > deadline:
            return 1e10
        hexes = get_hexes_symmetry(params)
        min_s = compute_min_side_smooth(hexes, temperature, outer_angle_rad)
        penalty = 0.0
        for i in range(12):
            for j in range(i + 1, 12):
                min1 = hexes[i].min(axis=0)
                max1 = hexes[i].max(axis=0)
                min2 = hexes[j].min(axis=0)
                max2 = hexes[j].max(axis=0)
                if min1[0] > max2[0] or max1[0] < min2[0] or min1[1] > max2[1] or max1[1] < min2[1]:
                    continue
                overlap = sat_overlap(hexes[i], hexes[j])
                if overlap > 0:
                    penalty += penalty_weight * (np.log1p(np.exp(overlap * 10)) / 10)
        return min_s + penalty

    def compute_objective_free(params, temperature=0.01, penalty_weight=50.0, outer_angle_rad=0.0):
        """Free params: 12 hexagons, 3 params each (x, y, angle). Total 36 params."""
        if time.time() > deadline:
            return 1e10
        hexes = []
        for i in range(12):
            x, y, a = params[3*i], params[3*i+1], params[3*i+2]
            hexes.append(hex_vertices(x, y, 1.0, a))
        
        min_s = compute_min_side_smooth(hexes, temperature, outer_angle_rad)
        penalty = 0.0
        for i in range(12):
            for j in range(i + 1, 12):
                min1 = hexes[i].min(axis=0)
                max1 = hexes[i].max(axis=0)
                min2 = hexes[j].min(axis=0)
                max2 = hexes[j].max(axis=0)
                if min1[0] > max2[0] or max1[0] < min2[0] or min1[1] > max2[1] or max1[1] < min2[1]:
                    continue
                overlap = sat_overlap(hexes[i], hexes[j])
                if overlap > 0:
                    penalty += penalty_weight * (np.log1p(np.exp(overlap * 10)) / 10)
        return min_s + penalty

    def compute_objective_exact_symmetry(params, outer_angle_rad=0.0):
        hexes = get_hexes_symmetry(params)
        min_s = compute_min_side_fixed(hexes, outer_angle_rad)
        penalty = 0.0
        for i in range(12):
            for j in range(i + 1, 12):
                min1 = hexes[i].min(axis=0)
                max1 = hexes[i].max(axis=0)
                min2 = hexes[j].min(axis=0)
                max2 = hexes[j].max(axis=0)
                if min1[0] > max2[0] or max1[0] < min2[0] or min1[1] > max2[1] or max1[1] < min2[1]:
                    continue
                overlap = sat_overlap(hexes[i], hexes[j])
                if overlap > 0:
                    penalty += overlap * 100.0
        return min_s + penalty

    def compute_objective_exact_free(params, outer_angle_rad=0.0):
        hexes = []
        for i in range(12):
            x, y, a = params[3*i], params[3*i+1], params[3*i+2]
            hexes.append(hex_vertices(x, y, 1.0, a))
        min_s = compute_min_side_fixed(hexes, outer_angle_rad)
        penalty = 0.0
        for i in range(12):
            for j in range(i + 1, 12):
                min1 = hexes[i].min(axis=0)
                max1 = hexes[i].max(axis=0)
                min2 = hexes[j].min(axis=0)
                max2 = hexes[j].max(axis=0)
                if min1[0] > max2[0] or max1[0] < min2[0] or min1[1] > max2[1] or max1[1] < min2[1]:
                    continue
                overlap = sat_overlap(hexes[i], hexes[j])
                if overlap > 0:
                    penalty += overlap * 100.0
        return min_s + penalty

    def optimize_outer_rotation(hexes):
        best_angle = 0.0
        best_s = np.inf
        gr = (np.sqrt(5) - 1) / 2
        a, b = 0.0, 60.0
        x1 = b - gr * (b - a)
        x2 = a + gr * (b - a)
        f1 = compute_min_side_fixed(hexes, np.radians(x1))
        f2 = compute_min_side_fixed(hexes, np.radians(x2))
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            if f1 < f2:
                b = x2
                x2 = x1
                f2 = f1
                x1 = b - gr * (b - a)
                f1 = compute_min_side_fixed(hexes, np.radians(x1))
            else:
                a = x1
                x1 = x2
                f1 = f2
                x2 = a + gr * (b - a)
                f2 = compute_min_side_fixed(hexes, np.radians(x2))
        candidates = [0.0, 15.0, 30.0, 45.0, (a + b) / 2, a + gr * (b - a) / 2]
        for ang in candidates:
            s = compute_min_side_fixed(hexes, np.radians(ang))
            if s < best_s:
                best_s = s
                best_angle = ang
        return best_s, best_angle

    def is_valid(hexes, tol=1e-9):
        for i in range(12):
            for j in range(i + 1, 12):
                min1 = hexes[i].min(axis=0)
                max1 = hexes[i].max(axis=0)
                min2 = hexes[j].min(axis=0)
                max2 = hexes[j].max(axis=0)
                if min1[0] > max2[0] or max1[0] < min2[0] or min1[1] > max2[1] or max1[1] < min2[1]:
                    continue
                overlap = sat_overlap(hexes[i], hexes[j])
                if overlap > tol:
                    return False
        return True

    # --- Symmetry Optimization ---
    bounds_sym = [
        (0.5, 5.0), (0.0, 2*np.pi), (0.0, np.pi/3),
        (0.5, 5.0), (0.0, 2*np.pi), (0.0, np.pi/3),
    ]

    best_s = np.inf
    best_params_sym = None
    best_params_free = None
    best_outer_angle_deg = 0.0

    # Phase 1: Symmetry global search
    if time.time() < deadline - 90:
        try:
            res = differential_evolution(
                lambda p: compute_objective_symmetry(p, temperature=0.05, penalty_weight=20.0),
                bounds_sym,
                maxiter=200,
                popsize=35,
                tol=1e-5,
                seed=42,
                polish=False
            )
            if res.fun < best_s:
                best_s = res.fun
                best_params_sym = res.x.copy()
        except Exception:
            pass

    # Phase 2: Symmetry refinement
    if best_params_sym is not None and time.time() < deadline - 60:
        try:
            res2 = differential_evolution(
                lambda p: compute_objective_symmetry(p, temperature=0.01, penalty_weight=50.0),
                bounds_sym,
                x0=best_params_sym,
                maxiter=150,
                popsize=25,
                tol=1e-7,
                seed=43,
                polish=False
            )
            if res2.fun < best_s:
                best_s = res2.fun
                best_params_sym = res2.x.copy()
        except Exception:
            pass

    # Phase 3: Symmetry local polish
    if best_params_sym is not None and time.time() < deadline - 40:
        try:
            res_local = minimize(
                lambda p: compute_objective_symmetry(p, temperature=0.005, penalty_weight=100.0),
                best_params_sym,
                method='L-BFGS-B',
                bounds=bounds_sym,
                options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-16}
            )
            if res_local.fun < best_s:
                best_s = res_local.fun
                best_params_sym = res_local.x.copy()
        except Exception:
            pass

    # Phase 4: Convert to free params and try free optimization
    if best_params_sym is not None and time.time() < deadline - 30:
        hexes = get_hexes_symmetry(best_params_sym)
        outer_s, outer_angle_deg = optimize_outer_rotation(hexes)
        if is_valid(hexes):
            best_s = outer_s
            best_outer_angle_deg = outer_angle_deg

        # Initialize free params from symmetry solution
        free_params = np.zeros(36)
        for i, h in enumerate(hexes):
            cx = h[:, 0].mean()
            cy = h[:, 1].mean()
            # Estimate angle from first vertex relative to center
            dx = h[0, 0] - cx
            dy = h[0, 1] - cy
            angle = math.atan2(dy, dx)
            free_params[3*i] = cx
            free_params[3*i+1] = cy
            free_params[3*i+2] = angle

        # Free optimization with L-BFGS-B
        if time.time() < deadline - 15:
            try:
                bounds_free = [(x-0.5, x+0.5) for x in free_params]  # rough bounds
                # Better bounds: x,y in [-5,5], angle in [0, 2pi]
                bounds_free = []
                for i in range(12):
                    bounds_free.append((free_params[3*i] - 2.0, free_params[3*i] + 2.0))
                    bounds_free.append((free_params[3*i+1] - 2.0, free_params[3*i+1] + 2.0))
                    bounds_free.append((-np.pi, np.pi))

                outer_angle_rad = np.radians(best_outer_angle_deg)
                res_free = minimize(
                    lambda p: compute_objective_free(p, temperature=0.002, penalty_weight=200.0, outer_angle_rad=outer_angle_rad),
                    free_params,
                    method='L-BFGS-B',
                    bounds=bounds_free,
                    options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-16}
                )
                if res_free.fun < best_s:
                    best_params_free = res_free.x.copy()
                    hexes_free = [hex_vertices(best_params_free[3*i], best_params_free[3*i+1], 1.0, best_params_free[3*i+2]) for i in range(12)]
                    if is_valid(hexes_free):
                        outer_s2, outer_angle_deg2 = optimize_outer_rotation(hexes_free)
                        best_s = outer_s2
                        best_outer_angle_deg = outer_angle_deg2
            except Exception:
                pass

    # Fallback: if symmetry didn't work well, try a direct free search
    if best_params_free is None and time.time() < deadline - 20:
        # Use a known good symmetric configuration as starting point
        r1, th1, al1, r2, th2, al2 = best_params_sym if best_params_sym is not None else [2.0, 0, 0, 4.0, 0, 0]
        hexes = get_hexes_symmetry([r1, th1, al1, r2, th2, al2])
        free_params = np.zeros(36)
        for i, h in enumerate(hexes):
            cx = h[:, 0].mean()
            cy = h[:, 1].mean()
            dx = h[0, 0] - cx
            dy = h[0, 1] - cy
            angle = math.atan2(dy, dx)
            free_params[3*i] = cx
            free_params[3*i+1] = cy
            free_params[3*i+2] = angle

        try:
            bounds_free = []
            for i in range(12):
                bounds_free.append((free_params[3*i] - 3.0, free_params[3*i] + 3.0))
                bounds_free.append((free_params[3*i+1] - 3.0, free_params[3*i+1] + 3.0))
                bounds_free.append((-np.pi, np.pi))

            res_free = minimize(
                lambda p: compute_objective_free(p, temperature=0.01, penalty_weight=100.0),
                free_params,
                method='L-BFGS-B',
                bounds=bounds_free,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-14}
            )
            hexes_free = [hex_vertices(res_free.x[3*i], res_free.x[3*i+1], 1.0, res_free.x[3*i+2]) for i in range(12)]
            if is_valid(hexes_free):
                outer_s, outer_angle_deg = optimize_outer_rotation(hexes_free)
                if outer_s < best_s:
                    best_s = outer_s
                    best_params_free = res_free.x.copy()
                    best_outer_angle_deg = outer_angle_deg
        except Exception:
            pass

    # Final evaluation
    if best_params_free is not None:
        hexes = [hex_vertices(best_params_free[3*i], best_params_free[3*i+1], 1.0, best_params_free[3*i+2]) for i in range(12)]
        if is_valid(hexes):
            final_s, final_angle = optimize_outer_rotation(hexes)
            best_s = final_s
            best_outer_angle_deg = final_angle
            inner = [[best_params_free[3*i], best_params_free[3*i+1], np.degrees(best_params_free[3*i+2])] for i in range(12)]
        else:
            # Fall back to symmetry
            hexes = get_hexes_symmetry(best_params_sym)
            final_s, final_angle = optimize_outer_rotation(hexes)
            best_s = final_s
            best_outer_angle_deg = final_angle
            r1, th1, al1, r2, th2, al2 = best_params_sym
            inner = []
            for k in range(6):
                angle = np.radians(60 * k)
                x1 = r1 * np.cos(th1 + angle)
                y1 = r1 * np.sin(th1 + angle)
                a1 = al1 + angle
                inner.append([x1, y1, np.degrees(a1)])
                x2 = r2 * np.cos(th2 + angle)
                y2 = r2 * np.sin(th2 + angle)
                a2 = al2 + angle
                inner.append([x2, y2, np.degrees(a2)])
    elif best_params_sym is not None:
        hexes = get_hexes_symmetry(best_params_sym)
        final_s, final_angle = optimize_outer_rotation(hexes)
        best_s = final_s
        best_outer_angle_deg = final_angle
        r1, th1, al1, r2, th2, al2 = best_params_sym
        inner = []
        for k in range(6):
            angle = np.radians(60 * k)
            x1 = r1 * np.cos(th1 + angle)
            y1 = r1 * np.sin(th1 + angle)
            a1 = al1 + angle
            inner.append([x1, y1, np.degrees(a1)])
            x2 = r2 * np.cos(th2 + angle)
            y2 = r2 * np.sin(th2 + angle)
            a2 = al2 + angle
            inner.append([x2, y2, np.degrees(a2)])
    else:
        # Last resort: random
        np.random.seed(42)
        inner = []
        for i in range(12):
            x = np.random.uniform(-4, 4)
            y = np.random.uniform(-4, 4)
            a = np.random.uniform(0, 360)
            inner.append([x, y, a])
        hexes = [hex_vertices(inner[i][0], inner[i][1], 1.0, np.radians(inner[i][2])) for i in range(12)]
        final_s, final_angle = optimize_outer_rotation(hexes)
        best_s = final_s
        best_outer_angle_deg = final_angle

    outer_center = (0.0, 0.0)
    return inner, outer_center, best_s, best_outer_angle_deg
