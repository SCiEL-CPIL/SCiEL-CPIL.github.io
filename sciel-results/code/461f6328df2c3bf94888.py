"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 11
START_TIME = time.time()
TIME_LIMIT = 140.0  # Safety margin for 150s limit

# EVOLVE-BLOCK-START
def construct():
    def hex_vertices(cx, cy, angle, s=1.0):
        """Return the 6 vertices of a regular hexagon."""
        angles = angle + np.arange(6) * np.pi / 3
        return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

    def get_axes(poly):
        """Get the 6 outward normals of a hexagon."""
        v = poly
        edges = np.roll(v, -1, axis=0) - v
        normals = np.column_stack([-edges[:, 1], edges[:, 0]])
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / norms
        return normals

    def compute_penalties(params):
        """
        Compute S and smooth penalties for containment and overlap.
        params layout:
        [0:3]   outer center x, y, angle
        [3]     outer side length S
        [4:4+3N] inner hexagons (cx, cy, angle)
        """
        ocx, ocy, oangle = params[0], params[1], params[2]
        S = params[3]
        
        if S < 0.5: S = 0.5
        
        outer_normals = np.column_stack([
            np.cos(oangle + np.pi/6 + np.arange(6)*np.pi/3),
            np.sin(oangle + np.pi/6 + np.arange(6)*np.pi/3)
        ])
        
        apothem = np.sqrt(3) / 2.0
        limit = S * apothem
        
        contain_violations = []
        
        inner_centers = params[4:4+3*N:3]
        inner_ys = params[5:4+3*N:3]
        inner_angles = params[6:4+3*N:3]
        
        inner_verts_list = []
        for i in range(N):
            cx, cy, ang = inner_centers[i], inner_ys[i], inner_angles[i]
            verts = hex_vertices(cx, cy, ang)
            inner_verts_list.append(verts)
            
        for i in range(N):
            verts = inner_verts_list[i]
            rel_verts = verts - np.array([ocx, ocy])
            projs = rel_verts @ outer_normals.T
            viol = projs - limit
            contain_violations.append(viol.flatten())
            
        contain_violations = np.concatenate(contain_violations)
        
        beta = 10.0
        max_v = np.max(contain_violations)
        if max_v > 0:
            log_sum_exp = max_v + np.log(np.sum(np.exp(beta * (contain_violations - max_v))))
            smooth_max_v = log_sum_exp / beta
            P_contain = np.log1p(np.exp(beta * smooth_max_v)) / beta
        else:
            P_contain = 0.0
            if max_v > -5.0:
                log_sum_exp = max_v + np.log(np.sum(np.exp(beta * (contain_violations - max_v))))
                smooth_max_v = log_sum_exp / beta
                P_contain = np.log1p(np.exp(beta * smooth_max_v)) / beta

        overlap_penalties = 0.0
        
        for i in range(N):
            v1 = inner_verts_list[i]
            n1 = get_axes(v1)
            for j in range(i + 1, N):
                v2 = inner_verts_list[j]
                n2 = get_axes(v2)
                
                axes = np.vstack([n1, n2])
                
                proj1 = v1 @ axes.T
                proj2 = v2 @ axes.T
                
                min1 = np.min(proj1, axis=1)
                max1 = np.max(proj1, axis=1)
                min2 = np.min(proj2, axis=1)
                max2 = np.max(proj2, axis=1)
                
                gap_forward = min2 - max1
                gap_backward = min1 - max2
                gaps = np.maximum(gap_forward, gap_backward)
                
                min_gap = np.min(gaps)
                
                if min_gap < 0:
                    neg_gaps = -gaps
                    max_neg_gap = np.max(neg_gaps)
                    log_sum_exp = max_neg_gap + np.log(np.sum(np.exp(beta * (neg_gaps - max_neg_gap))))
                    smooth_max_neg_gap = log_sum_exp / beta
                    smooth_min_gap = -smooth_max_neg_gap
                    
                    val = -smooth_min_gap
                    if val > 0:
                        P_pair = np.log1p(np.exp(beta * val)) / beta
                    else:
                        P_pair = 0.0
                    overlap_penalties += P_pair
        
        w_c = 100.0
        w_o = 100.0
        
        return S + w_c * P_contain + w_o * overlap_penalties

    def get_initial_params():
        p = np.zeros(3 + 1 + 3 * N)
        p[0:3] = [0, 0, 0]
        p[3] = 3.5
        
        p[4:7] = [0, 0, 0]
        
        idx = 1
        for k in range(6):
            theta = k * np.pi / 3
            cx = np.sqrt(3) * np.cos(theta)
            cy = np.sqrt(3) * np.sin(theta)
            p[4 + 3*idx : 4 + 3*idx + 3] = [cx, cy, 0]
            idx += 1
            
        angles_r2 = [0, 120, 240, 180]
        for deg in angles_r2:
            theta = np.radians(deg)
            cx = 2 * np.sqrt(3) * np.cos(theta)
            cy = 2 * np.sqrt(3) * np.sin(theta)
            p[4 + 3*idx : 4 + 3*idx + 3] = [cx, cy, 0]
            idx += 1
            
        return p

    best_params = get_initial_params()
    best_score = compute_penalties(best_params)
    
    np.random.seed(42)
    
    for i in range(50):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
            
        p = best_params.copy()
        if i > 0:
            noise_inner = np.random.normal(0, 0.5, size=3 * N)
            p[4:] += noise_inner
            
        try:
            res = minimize(compute_penalties, p, method='Powell', 
                          options={'maxiter': 200, 'maxfev': 2000})
            
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass

    ocx, ocy, oangle = best_params[0], best_params[1], best_params[2]
    S_final = best_params[3]
    
    inner = []
    for i in range(N):
        cx = best_params[4 + 3 * i]
        cy = best_params[5 + 3 * i]
        angle = best_params[6 + 3 * i]
        inner.append([cx, cy, np.degrees(angle)])
    
    def exact_check(inner, ocx, ocy, oangle, S):
        outer_normals = np.column_stack([
            np.cos(oangle + np.pi/6 + np.arange(6)*np.pi/3),
            np.sin(oangle + np.pi/6 + np.arange(6)*np.pi/3)
        ])
        limit = S * np.sqrt(3) / 2.0
        
        for i in range(N):
            cx, cy, ang_deg = inner[i]
            ang = np.radians(ang_deg)
            verts = hex_vertices(cx, cy, ang)
            rel_verts = verts - np.array([ocx, ocy])
            projs = rel_verts @ outer_normals.T
            if np.any(projs > limit + 1e-9):
                return False
                
        verts_list = []
        for i in range(N):
            cx, cy, ang_deg = inner[i]
            ang = np.radians(ang_deg)
            verts = hex_vertices(cx, cy, ang)
            verts_list.append(verts)
            
        for i in range(N):
            v1 = verts_list[i]
            n1 = get_axes(v1)
            for j in range(i + 1, N):
                v2 = verts_list[j]
                n2 = get_axes(v2)
                axes = np.vstack([n1, n2])
                proj1 = v1 @ axes.T
                proj2 = v2 @ axes.T
                min1 = np.min(proj1, axis=1)
                max1 = np.max(proj1, axis=1)
                min2 = np.min(proj2, axis=1)
                max2 = np.max(proj2, axis=1)
                gap_forward = min2 - max1
                gap_backward = min1 - max2
                gaps = np.maximum(gap_forward, gap_backward)
                if np.all(gaps < -1e-9):
                    return False
        return True

    if not exact_check(inner, ocx, ocy, oangle, S_final):
        S_final += 0.01

    return inner, (ocx, ocy), S_final, np.degrees(oangle)
# EVOLVE-BLOCK-END
