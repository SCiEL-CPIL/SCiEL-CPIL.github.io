import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def construct():
    t0 = time.time()
    DEADLINE = t0 + 140.0

    def hex_vertices(cx, cy, side, ang):
        a = ang + np.deg2rad(np.arange(0, 360, 60))
        return np.column_stack([cx + side * np.cos(a), cy + side * np.sin(a)])

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
        min_overlap = np.min(overlaps)
        
        return float(min_overlap)

    def inside_violation(inner_verts, outer_verts):
        outer_normals = get_normals(outer_verts)
        p_inner = inner_verts @ outer_normals.T
        p_outer_max = (outer_verts @ outer_normals.T).max(axis=0)
        p_inner_max = p_inner.max(axis=0)
        viol = p_inner_max - p_outer_max
        return float(max(0.0, np.max(viol)))

    def compute_min_side(hexes, outer_angle=0.0):
        normal_angles = outer_angle + np.deg2rad(np.arange(30, 360, 60))
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        max_proj = 0.0
        for h in hexes:
            proj = h @ normals.T
            max_proj = max(max_proj, np.max(proj))
            
        s = max_proj * 2.0 / np.sqrt(3.0)
        return s

    def is_valid_layout(hexes, s, outer_angle=0.0, tol=1e-4):
        outer = hex_vertices(0.0, 0.0, s, outer_angle)
        
        for h in hexes:
            if inside_violation(h, outer) > tol:
                return False
        
        for i in range(len(hexes)):
            for j in range(i + 1, len(hexes)):
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

    # ---------- Symmetric Objective ----------
    def generate_hexes_symmetric(params):
        """
        params: [x1, y1, a1, x2, y2, a2]
        Generates 12 hexagons using 6-fold symmetry.
        """
        x1, y1, a1, x2, y2, a2 = params
        hexes = []
        for k in range(6):
            angle = np.deg2rad(60 * k)
            c = np.cos(angle)
            s_ = np.sin(angle)
            
            # Rotate position
            x1r = x1 * c - y1 * s_
            y1r = x1 * s_ + y1 * c
            a1r = a1 + angle
            
            x2r = x2 * c - y2 * s_
            y2r = x2 * s_ + y2 * c
            a2r = a2 + angle
            
            hexes.append(hex_vertices(x1r, y1r, 1.0, a1r))
            hexes.append(hex_vertices(x2r, y2r, 1.0, a2r))
        return hexes

    def objective_symmetric(params, outer_angle=0.0):
        if time.time() > DEADLINE:
            return 1e10
        hexes = generate_hexes_symmetric(params)
        s_min = compute_min_side(hexes, outer_angle)
        
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
                    
        return s_min + penalty

    # ---------- 3-fold Symmetry Objective (Structural Alternative) ----------
    def generate_hexes_3fold(params):
        """
        params: [x1, y1, a1, x2, y2, a2, x3, y3, a3, x4, y4, a4]
        Generates 12 hexagons using 3-fold symmetry (4 orbits of 3).
        """
        hexes = []
        for i in range(4):
            x, y, a = params[3*i], params[3*i+1], params[3*i+2]
            for k in range(3):
                angle = np.deg2rad(120 * k)
                c = np.cos(angle)
                s_ = np.sin(angle)
                xr = x * c - y * s_
                yr = x * s_ + y * c
                ar = a + angle
                hexes.append(hex_vertices(xr, yr, 1.0, ar))
        return hexes

    def objective_3fold(params, outer_angle=0.0):
        if time.time() > DEADLINE:
            return 1e10
        hexes = generate_hexes_3fold(params)
        s_min = compute_min_side(hexes, outer_angle)
        
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
                    
        return s_min + penalty

    best_s = np.inf
    best_hexes = None
    best_outer_angle = 0.0
    best_params_sym = None
    best_params_3fold = None

    outer_angles = [0.0, np.deg2rad(30)]

    for oa in outer_angles:
        if time.time() > DEADLINE - 10:
            break

        # --- 1. 6-fold Symmetric Global Search ---
        bounds_sym = [
            (-4.0, 4.0), (-4.0, 4.0), (0.0, np.pi/3),
            (-4.0, 4.0), (-4.0, 4.0), (0.0, np.pi/3)
        ]
        
        try:
            res_de = differential_evolution(
                objective_symmetric,
                bounds_sym,
                args=(oa,),
                maxiter=150,
                popsize=15,
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

        # --- 2. 3-fold Symmetric Global Search ---
        bounds_3fold = [
            (-4.0, 4.0), (-4.0, 4.0), (0.0, np.pi/3),
            (-4.0, 4.0), (-4.0, 4.0), (0.0, np.pi/3),
            (-4.0, 4.0), (-4.0, 4.0), (0.0, np.pi/3),
            (-4.0, 4.0), (-4.0, 4.0), (0.0, np.pi/3)
        ]
        
        try:
            res_de_3 = differential_evolution(
                objective_3fold,
                bounds_3fold,
                args=(oa,),
                maxiter=100,
                popsize=15,
                tol=1e-5,
                seed=123,
                polish=True
            )
            
            if res_de_3.fun < best_s:
                best_s = res_de_3.fun
                best_params_3fold = res_de_3.x
                best_hexes = generate_hexes_3fold(res_de_3.x)
                best_outer_angle = oa
        except Exception:
            pass

        # --- 3. Local Refinement for 6-fold ---
        if best_params_sym is not None and time.time() < DEADLINE - 20:
            try:
                res_local = minimize(
                    objective_symmetric,
                    best_params_sym,
                    args=(oa,),
                    method='L-BFGS-B',
                    bounds=bounds_sym,
                    options={'maxiter': 200, 'ftol': 1e-12}
                )
                if res_local.fun < best_s:
                    best_s = res_local.fun
                    best_params_sym = res_local.x
                    best_hexes = generate_hexes_symmetric(res_local.x)
                    best_outer_angle = oa
            except Exception:
                pass

        # --- 4. Local Refinement for 3-fold ---
        if best_params_3fold is not None and time.time() < DEADLINE - 20:
            try:
                res_local_3 = minimize(
                    objective_3fold,
                    best_params_3fold,
                    args=(oa,),
                    method='L-BFGS-B',
                    bounds=bounds_3fold,
                    options={'maxiter': 200, 'ftol': 1e-12}
                )
                if res_local_3.fun < best_s:
                    best_s = res_local_3.fun
                    best_params_3fold = res_local_3.x
                    best_hexes = generate_hexes_3fold(res_local_3.x)
                    best_outer_angle = oa
            except Exception:
                pass

        # --- 5. Multi-start Random Symmetric Search ---
        rng = np.random.default_rng(12345)
        for _ in range(3):
            if time.time() > DEADLINE - 15:
                break
            p = np.array([
                rng.uniform(1, 3), rng.uniform(-0.5, 0.5), rng.uniform(0, np.pi/3),
                rng.uniform(2, 4), rng.uniform(-0.5, 0.5), rng.uniform(0, np.pi/3)
            ])
            try:
                res_local = minimize(
                    objective_symmetric,
                    p,
                    args=(oa,),
                    method='L-BFGS-B',
                    bounds=bounds_sym,
                    options={'maxiter': 100, 'ftol': 1e-12}
                )
                if res_local.fun < best_s:
                    best_s = res_local.fun
                    best_params_sym = res_local.x
                    best_hexes = generate_hexes_symmetric(res_local.x)
                    best_outer_angle = oa
            except Exception:
                pass

    # --- 6. Full Non-Symmetric Refinement ---
    def objective_full(params, outer_angle=0.0):
        if time.time() > DEADLINE:
            return 1e10
        pos = params[:24].reshape(12, 2)
        angles = params[24:36]
        
        hexes = []
        for i in range(12):
            hexes.append(hex_vertices(pos[i, 0], pos[i, 1], 1.0, angles[i]))
        
        s_min = compute_min_side(hexes, outer_angle)
        
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
                    
        return s_min + penalty

    if best_hexes is not None and time.time() < DEADLINE - 20:
        pos = np.array([h.mean(axis=0) for h in best_hexes])
        
        # Estimate angles
        angles = np.zeros(12)
        # If we have symmetric params, we can try to recover angles, but L-BFGS will fix it.
        # Just initialize with 0 or current estimates.
        for i, h in enumerate(best_hexes):
            v0 = h[0] - pos[i]
            ang = np.arctan2(v0[1], v0[0])
            angles[i] = ang

        p_full = np.concatenate([pos.ravel(), angles])
        bounds_full = [(-5, 5)] * 24 + [(0, np.pi)] * 12
        
        try:
            res_full = minimize(
                objective_full,
                p_full,
                args=(best_outer_angle,),
                method='L-BFGS-B',
                bounds=bounds_full,
                options={'maxiter': 300, 'ftol': 1e-12}
            )
            if res_full.fun < best_s:
                pos_f = res_full.x[:24].reshape(12, 2)
                ang_f = res_full.x[24:36]
                hexes_f = [hex_vertices(pos_f[i, 0], pos_f[i, 1], 1.0, ang_f[i]) for i in range(12)]
                s_f = compute_min_side(hexes_f, best_outer_angle)
                
                if is_valid_layout(hexes_f, s_f, best_outer_angle):
                    best_s = s_f
                    best_hexes = hexes_f
        except Exception:
            pass

    # --- 7. Final Validation and Formatting ---
    if best_hexes is None:
        # Fallback: Flower pattern
        d1 = np.sqrt(3)
        d2 = 2 * np.sqrt(3)
        pos = [[0.0, 0.0]]
        for k in range(6):
            a = np.deg2rad(60 * k)
            pos.append([d1 * np.cos(a), d1 * np.sin(a)])
        for k in range(5):
            a = np.deg2rad(60 * k)
            pos.append([d2 * np.cos(a), d2 * np.sin(a)])
        pos = np.array(pos)
        angles = np.zeros(12)
        best_hexes = [hex_vertices(pos[i, 0], pos[i, 1], 1.0, angles[i]) for i in range(12)]
        best_s = compute_min_side(best_hexes, 0.0)
        best_outer_angle = 0.0

    s_final = compute_min_side(best_hexes, best_outer_angle)
    if not is_valid_layout(best_hexes, s_final, best_outer_angle, tol=1e-3):
        s_final *= 1.01
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
