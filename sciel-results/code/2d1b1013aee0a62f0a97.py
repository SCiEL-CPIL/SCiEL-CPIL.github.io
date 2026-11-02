import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
import math

N = 12

def construct():
    t0 = time.time()
    DEADLINE = t0 + 145.0

    def hex_vertices(cx, cy, side, ang):
        a = ang + np.deg2rad(np.arange(0, 360, 60))
        return np.column_stack([cx + side * np.cos(a), cy + side * np.sin(a)])

    def get_normals(verts):
        edges = np.roll(verts, -1, axis=0) - verts
        normals = np.column_stack([edges[:, 1], -edges[:, 0]])
        norms = np.linalg.norm(normals, axis=1)
        normals = normals / (norms[:, np.newaxis] + 1e-15)
        return normals

    def sat_overlap(v1, v2, normals1=None, normals2=None):
        if normals1 is None:
            normals1 = get_normals(v1)
        if normals2 is None:
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

    def inside_violation(inner_verts, outer_verts, outer_normals=None):
        if outer_normals is None:
            outer_normals = get_normals(outer_verts)
        p_inner = inner_verts @ outer_normals.T
        p_outer_max = (outer_verts @ outer_normals.T).max(axis=0)
        p_inner_max = p_inner.max(axis=0)
        viol = p_inner_max - p_outer_max
        return float(max(0.0, np.max(viol)))

    def compute_min_side(hexes, outer_angle=0.0):
        """Compute minimum side length of enclosing regular hexagon."""
        # A regular hexagon with side s has inradius = s * sqrt(3)/2
        # The distance from center to each edge is the inradius
        # For 6 edge normals at angles theta_k = outer_angle + k*pi/3 + pi/6
        # We need: max over all vertices of (n_k . v) <= inradius for all k
        # So inradius >= max_k max_v (n_k . v)
        # And s = 2 * inradius / sqrt(3)
        
        normal_angles = outer_angle + np.deg2rad(np.arange(30, 390, 60))
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        max_proj = 0.0
        for h in hexes:
            proj = h @ normals.T
            max_proj = max(max_proj, np.max(proj))
        
        inradius = max_proj
        s = 2.0 * inradius / np.sqrt(3.0)
        return s

    def is_valid_layout(hexes, s, outer_angle=0.0, tol=1e-4):
        outer = hex_vertices(0.0, 0.0, s, outer_angle)
        outer_normals = get_normals(outer)
        for h in hexes:
            if inside_violation(h, outer, outer_normals) > tol:
                return False
        # Precompute normals for each hexagon
        hex_normals = [get_normals(h) for h in hexes]
        for i in range(len(hexes)):
            for j in range(i + 1, len(hexes)):
                min1 = hexes[i].min(axis=0)
                max1 = hexes[i].max(axis=0)
                min2 = hexes[j].min(axis=0)
                max2 = hexes[j].max(axis=0)
                if min1[0] > max2[0] or max1[0] < min2[0] or min1[1] > max2[1] or max1[1] < min2[1]:
                    continue
                overlap = sat_overlap(hexes[i], hexes[j], hex_normals[i], hex_normals[j])
                if overlap > tol:
                    return False
        return True

    def soft_max_penalty(overlaps, beta=50.0):
        clipped = np.maximum(overlaps, 0.0)
        if clipped.max() < 1e-12:
            return 0.0
        shifted = clipped - clipped.max()
        log_sum = np.log(np.sum(np.exp(beta * shifted)) + 1e-15) / beta
        return clipped.max() + log_sum

    def generate_hexes_symmetric(params):
        x1, y1, a1, x2, y2, a2 = params
        base1 = hex_vertices(x1, y1, 1.0, a1)
        base2 = hex_vertices(x2, y2, 1.0, a2)
        hexes = []
        for k in range(6):
            angle = np.deg2rad(60 * k)
            c = np.cos(angle)
            s_ = np.sin(angle)
            rot = np.array([[c, -s_], [s_, c]])
            h1 = base1 @ rot.T
            h2 = base2 @ rot.T
            hexes.append(h1)
            hexes.append(h2)
        return hexes

    def objective_symmetric(params, outer_angle=0.0):
        if time.time() > DEADLINE:
            return 1e10
        hexes = generate_hexes_symmetric(params)
        s_min = compute_min_side(hexes, outer_angle)
        
        # Precompute normals for SAT
        hex_normals = [get_normals(h) for h in hexes]
        
        overlaps = []
        for i in range(12):
            for j in range(i + 1, 12):
                min1 = hexes[i].min(axis=0)
                max1 = hexes[i].max(axis=0)
                min2 = hexes[j].min(axis=0)
                max2 = hexes[j].max(axis=0)
                if min1[0] > max2[0] or max1[0] < min2[0] or min1[1] > max2[1] or max1[1] < min2[1]:
                    continue
                overlap = sat_overlap(hexes[i], hexes[j], hex_normals[i], hex_normals[j])
                overlaps.append(overlap)
        if overlaps:
            penalty = soft_max_penalty(np.array(overlaps), beta=50.0) * 1000.0
        else:
            penalty = 0.0
        return s_min + penalty

    def objective_full(params, outer_angle=0.0):
        if time.time() > DEADLINE:
            return 1e10
        pos = params[:24].reshape(12, 2)
        angles = params[24:36]
        hexes = []
        for i in range(12):
            hexes.append(hex_vertices(pos[i, 0], pos[i, 1], 1.0, angles[i]))
        s_min = compute_min_side(hexes, outer_angle)
        
        # Precompute normals for SAT
        hex_normals = [get_normals(h) for h in hexes]
        
        overlaps = []
        for i in range(12):
            for j in range(i + 1, 12):
                min1 = hexes[i].min(axis=0)
                max1 = hexes[i].max(axis=0)
                min2 = hexes[j].min(axis=0)
                max2 = hexes[j].max(axis=0)
                if min1[0] > max2[0] or max1[0] < min2[0] or min1[1] > max2[1] or max1[1] < min2[1]:
                    continue
                overlap = sat_overlap(hexes[i], hexes[j], hex_normals[i], hex_normals[j])
                overlaps.append(overlap)
        if overlaps:
            penalty = soft_max_penalty(np.array(overlaps), beta=50.0) * 1000.0
        else:
            penalty = 0.0
        return s_min + penalty

    def honeycomb_init():
        """Generate a honeycomb-like initial configuration."""
        # 1 center + 6 in first ring + 5 in second ring (partial)
        d1 = np.sqrt(3)  # distance to first ring centers
        d2 = 2 * np.sqrt(3)  # distance to second ring centers
        
        pos = [[0.0, 0.0]]
        for k in range(6):
            a = np.deg2rad(60 * k)
            pos.append([d1 * np.cos(a), d1 * np.sin(a)])
        for k in range(5):
            a = np.deg2rad(60 * k)
            pos.append([d2 * np.cos(a), d2 * np.sin(a)])
        
        pos = np.array(pos)
        angles = np.zeros(12)
        return pos, angles

    best_s = np.inf
    best_hexes = None
    best_outer_angle = 0.0
    best_params_sym = None
    best_params_full = None
    best_pos_full = None
    best_angles_full = None

    outer_angles = [0.0, np.deg2rad(30)]

    for oa in outer_angles:
        if time.time() > DEADLINE - 15:
            break

        bounds_sym = [
            (-4.0, 4.0), (-4.0, 4.0), (0.0, np.pi/3),
            (-4.0, 4.0), (-4.0, 4.0), (0.0, np.pi/3)
        ]

        # Try differential evolution
        try:
            res_de = differential_evolution(
                objective_symmetric,
                bounds_sym,
                args=(oa,),
                maxiter=100,
                popsize=12,
                tol=1e-5,
                seed=42,
                polish=True
            )
            if res_de.fun < best_s:
                best_s = res_de.fun
                best_params_sym = res_de.x
                best_hexes = generate_hexes_symmetric(res_de.x)
                best_outer_angle = oa
        except Exception:
            pass

        # Local refinement from DE result
        if best_params_sym is not None:
            try:
                res_local = minimize(
                    objective_symmetric,
                    best_params_sym,
                    args=(oa,),
                    method='L-BFGS-B',
                    bounds=bounds_sym,
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-14}
                )
                if res_local.fun < best_s:
                    best_s = res_local.fun
                    best_params_sym = res_local.x
                    best_hexes = generate_hexes_symmetric(res_local.x)
                    best_outer_angle = oa
            except Exception:
                pass

        # Multiple random restarts for local optimization
        rng = np.random.default_rng(12345 + int(oa * 1000))
        for _ in range(8):
            if time.time() > DEADLINE - 20:
                break
            p = np.array([
                rng.uniform(0.5, 3.5), rng.uniform(-1.0, 1.0), rng.uniform(0, np.pi/3),
                rng.uniform(1.5, 4.0), rng.uniform(-1.0, 1.0), rng.uniform(0, np.pi/3)
            ])
            try:
                res_local = minimize(
                    objective_symmetric,
                    p,
                    args=(oa,),
                    method='L-BFGS-B',
                    bounds=bounds_sym,
                    options={'maxiter': 150, 'ftol': 1e-14}
                )
                if res_local.fun < best_s:
                    best_s = res_local.fun
                    best_params_sym = res_local.x
                    best_hexes = generate_hexes_symmetric(res_local.x)
                    best_outer_angle = oa
            except Exception:
                pass

    # Break symmetry: optimize all 12 hexagons independently
    if best_hexes is not None and time.time() < DEADLINE - 30:
        if best_params_sym is not None:
            pos = np.array([h.mean(axis=0) for h in best_hexes])
            a1, a2 = best_params_sym[2], best_params_sym[5]
            angles = np.zeros(12)
            for k in range(6):
                ang_k = np.deg2rad(60 * k)
                angles[2*k] = a1 + ang_k
                angles[2*k+1] = a2 + ang_k
        else:
            pos, angles = honeycomb_init()
            
        p_full = np.concatenate([pos.ravel(), angles])
        bounds_full = [(-5, 5)] * 24 + [(0, np.pi)] * 12
        
        try:
            res_full = minimize(
                objective_full,
                p_full,
                args=(best_outer_angle,),
                method='L-BFGS-B',
                bounds=bounds_full,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-14}
            )
            if res_full.fun < best_s:
                pos_f = res_full.x[:24].reshape(12, 2)
                ang_f = res_full.x[24:36]
                hexes_f = [hex_vertices(pos_f[i, 0], pos_f[i, 1], 1.0, ang_f[i]) for i in range(12)]
                s_f = compute_min_side(hexes_f, best_outer_angle)
                if is_valid_layout(hexes_f, s_f, best_outer_angle):
                    best_s = s_f
                    best_hexes = hexes_f
                    best_params_full = res_full.x
                    best_pos_full = pos_f
                    best_angles_full = ang_f
        except Exception:
            pass

    # Additional polish on full parameters
    if best_params_full is not None and time.time() < DEADLINE - 15:
        try:
            res_polish = minimize(
                objective_full,
                best_params_full,
                args=(best_outer_angle,),
                method='L-BFGS-B',
                bounds=[(-5, 5)] * 24 + [(0, np.pi)] * 12,
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-15}
            )
            if res_polish.fun < best_s:
                pos_f = res_polish.x[:24].reshape(12, 2)
                ang_f = res_polish.x[24:36]
                hexes_f = [hex_vertices(pos_f[i, 0], pos_f[i, 1], 1.0, ang_f[i]) for i in range(12)]
                s_f = compute_min_side(hexes_f, best_outer_angle)
                if is_valid_layout(hexes_f, s_f, best_outer_angle):
                    best_s = s_f
                    best_hexes = hexes_f
                    best_params_full = res_polish.x
                    best_pos_full = pos_f
                    best_angles_full = ang_f
        except Exception:
            pass

    # Final polish on symmetric if still best
    if best_hexes is not None and best_params_sym is not None and time.time() < DEADLINE - 10:
        try:
            res_polish_sym = minimize(
                objective_symmetric,
                best_params_sym,
                args=(best_outer_angle,),
                method='L-BFGS-B',
                bounds=[(-4.0, 4.0), (-4.0, 4.0), (0.0, np.pi/3),
                       (-4.0, 4.0), (-4.0, 4.0), (0.0, np.pi/3)],
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-15}
            )
            if res_polish_sym.fun < best_s:
                best_params_sym = res_polish_sym.x
                best_hexes = generate_hexes_symmetric(res_polish_sym.x)
                best_s = compute_min_side(best_hexes, best_outer_angle)
        except Exception:
            pass

    # Fallback if no valid solution found
    if best_hexes is None:
        pos, angles = honeycomb_init()
        best_hexes = [hex_vertices(pos[i, 0], pos[i, 1], 1.0, angles[i]) for i in range(12)]
        best_s = compute_min_side(best_hexes, 0.0)
        best_outer_angle = 0.0

    s_final = compute_min_side(best_hexes, best_outer_angle)
    if not is_valid_layout(best_hexes, s_final, best_outer_angle, tol=1e-3):
        s_final *= 1.02
        if not is_valid_layout(best_hexes, s_final, best_outer_angle, tol=1e-3):
            s_final *= 1.05
        if not is_valid_layout(best_hexes, s_final, best_outer_angle, tol=1e-3):
            s_final = 4.5

    inner = []
    for h in best_hexes:
        cx, cy = h.mean(axis=0)
        v0 = h[0] - np.array([cx, cy])
        ang = np.arctan2(v0[1], v0[0])
        inner.append([float(cx), float(cy), float(np.rad2deg(ang))])

    outer_center = (0.0, 0.0)
    outer_side = float(s_final)
    outer_angle_deg = float(np.rad2deg(best_outer_angle))

    return inner, outer_center, outer_side, outer_angle_deg
