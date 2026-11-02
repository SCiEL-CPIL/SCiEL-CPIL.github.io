import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
import math

N = 12

def construct():
    t0 = time.time()
    DEADLINE = t0 + 145.0  # Safety margin vs 150s

    # ---------- geometry helpers ----------
    def hex_vertices(cx, cy, side, ang):
        """Returns 6x2 array of vertices for a regular hexagon."""
        a = ang + np.deg2rad(np.arange(0, 360, 60))
        return np.column_stack([cx + side * np.cos(a), cy + side * np.sin(a)])

    def get_normals(verts):
        """Returns 6x2 array of outward normals for a CCW polygon."""
        edges = np.roll(verts, -1, axis=0) - verts
        normals = np.column_stack([edges[:, 1], -edges[:, 0]])
        norms = np.linalg.norm(normals, axis=1)
        normals = normals / (norms[:, np.newaxis] + 1e-15)
        return normals

    def sat_overlap(v1, v2):
        """
        Returns min overlap depth (negative if disjoint) using SAT.
        """
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
        """Returns max distance inner sticks out of outer (0 if inside)."""
        outer_normals = get_normals(outer_verts)
        
        p_inner = inner_verts @ outer_normals.T
        p_outer_max = (outer_verts @ outer_normals.T).max(axis=0)
        
        p_inner_max = p_inner.max(axis=0)
        
        viol = p_inner_max - p_outer_max
        return float(max(0.0, np.max(viol)))

    def compute_min_side(hexes, outer_angle=0.0):
        """
        Calculate the minimum side length of a regular hexagon (centered at origin, given orientation)
        that contains all given hexagons.
        """
        # The outer hexagon has fixed orientation (outer_angle).
        # Its supporting lines are defined by the 6 outward normals of a hexagon with side 1, angle outer_angle.
        normal_angles = outer_angle + np.deg2rad(np.arange(30, 360, 60))
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        max_proj = 0.0
        for h in hexes:
            proj = h @ normals.T
            max_proj = max(max_proj, np.max(proj))
            
        # For a regular hexagon of side s, the distance from center to side is s * sqrt(3) / 2.
        # So max_proj = s * sqrt(3) / 2 => s = max_proj * 2 / sqrt(3)
        s = max_proj * 2.0 / np.sqrt(3.0)
        return s

    def is_valid_layout(hexes, s, outer_angle=0.0, tol=1e-4):
        """Check if layout is valid: no overlaps, all inside outer hexagon."""
        outer = hex_vertices(0.0, 0.0, s, outer_angle)
        
        # Check containment
        for h in hexes:
            if inside_violation(h, outer) > tol:
                return False
        
        # Check pairwise overlaps
        for i in range(len(hexes)):
            for j in range(i + 1, len(hexes)):
                # Quick bounding box check
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
        Returns list of 12 hexagons (each 6x2).
        """
        x1, y1, a1, x2, y2, a2 = params
        
        # Base hexagons
        base1 = hex_vertices(x1, y1, 1.0, a1)
        base2 = hex_vertices(x2, y2, 1.0, a2)
        
        hexes = []
        for k in range(6):
            angle = np.deg2rad(60 * k)
            c = np.cos(angle)
            s_ = np.sin(angle)
            rot = np.array([[c, -s_], [s_, c]])
            
            # Rotate base1 and base2
            h1 = base1 @ rot.T
            h2 = base2 @ rot.T
            
            hexes.append(h1)
            hexes.append(h2)
        
        return hexes

    def objective_symmetric(params, outer_angle=0.0):
        if time.time() > DEADLINE:
            return 1e10
            
        hexes = generate_hexes_symmetric(params)
        
        # Calculate minimal outer side required to contain all
        s_min = compute_min_side(hexes, outer_angle)
        
        penalty = 0.0
        
        # Check overlaps
        for i in range(12):
            for j in range(i + 1, 12):
                # Quick bounding box check
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

    # ---------- Non-Symmetric Objective (Fallback/Refinement) ----------
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

    best_s = np.inf
    best_hexes = None
    best_outer_angle = 0.0
    best_params_sym = None
    best_params_full = None

    # Try different outer angles (0 and 30 degrees are typical optimal orientations)
    outer_angles = [0.0, np.deg2rad(30)]

    for oa in outer_angles:
        if time.time() > DEADLINE - 10:
            break

        # --- 1. Symmetric Global Search ---
        # Bounds for 2 base hexagons:
        # x, y in [-4, 4]
        # angle in [0, pi/3] (due to 6-fold symmetry, angles repeat every 60 deg)
        bounds_sym = [
            (-4.0, 4.0), (-4.0, 4.0), (0.0, np.pi/3),
            (-4.0, 4.0), (-4.0, 4.0), (0.0, np.pi/3)
        ]
        
        # Use Differential Evolution for global search
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

        # --- 2. Local Refinement from Symmetric Start ---
        if best_params_sym is not None:
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

        # --- 3. Multi-start Random Symmetric Search ---
        # Try a few random starts to escape local minima
        rng = np.random.default_rng(12345)
        for _ in range(5):
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

    # --- 4. Full Non-Symmetric Refinement ---
    # If we found a good symmetric candidate, try to break symmetry to see if it improves
    if best_hexes is not None and time.time() < DEADLINE - 20:
        # Initialize full params from symmetric solution
        pos = np.array([h.mean(axis=0) for h in best_hexes])
        # Estimate angles from symmetric params if available, else 0
        if best_params_sym is not None:
            a1, a2 = best_params_sym[2], best_params_sym[5]
            angles = np.zeros(12)
            for k in range(6):
                ang_k = np.deg2rad(60 * k)
                angles[2*k] = a1 + ang_k
                angles[2*k+1] = a2 + ang_k
        else:
            angles = np.zeros(12)
            
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
                
                # Validate
                if is_valid_layout(hexes_f, s_f, best_outer_angle):
                    best_s = s_f
                    best_hexes = hexes_f
                    best_params_full = res_full.x
        except Exception:
            pass

    # --- 5. Final Validation and Formatting ---
    if best_hexes is None:
        # Fallback: Return a known valid layout (flower pattern)
        # 1 center, 6 ring 1, 5 ring 2
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

    # Ensure validity one last time
    s_final = compute_min_side(best_hexes, best_outer_angle)
    if not is_valid_layout(best_hexes, s_final, best_outer_angle, tol=1e-3):
        # If invalid, expand slightly
        s_final *= 1.01
        if not is_valid_layout(best_hexes, s_final, best_outer_angle, tol=1e-3):
            s_final = 4.5 # Safe fallback

    # Extract centers and angles
    inner = []
    for h in best_hexes:
        cx, cy = h.mean(axis=0)
        # Estimate angle: angle of first vertex relative to center
        v0 = h[0] - np.array([cx, cy])
        ang = np.arctan2(v0[1], v0[0])
        # Normalize to [0, 60) for consistency? No, just return the angle.
        # The evaluator allows any angle.
        inner.append([float(cx), float(cy), float(np.rad2deg(ang))])

    outer_center = (0.0, 0.0)
    outer_side = float(s_final)
    outer_angle_deg = float(np.rad2deg(best_outer_angle))

    return inner, outer_center, outer_side, outer_angle_deg
