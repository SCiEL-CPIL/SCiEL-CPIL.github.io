import math
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar, linprog

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Define candidate point sets (axial coordinates)
    candidates = []

    axial_1 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ]
    candidates.append(axial_1)

    axial_2 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_2)

    axial_3 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1)
    ]
    candidates.append(axial_3)

    axial_4 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2)
    ]
    candidates.append(axial_4)

    axial_5 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_5)

    axial_6 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3)
    ]
    candidates.append(axial_6)

    axial_7 = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_7)

    axial_8 = [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)
    ]
    candidates.append(axial_8)

    axial_9 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (-2, 2), (-1, 3), (1, -2)
    ]
    candidates.append(axial_9)

    axial_10 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3)
    ]
    candidates.append(axial_10)

    axial_11 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (1, 3)
    ]
    candidates.append(axial_11)

    axial_12 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3),
        (0, 4)
    ]
    candidates.append(axial_12)

    axial_13 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (1, 2)
    ]
    candidates.append(axial_13)

    axial_14 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (1, 1), (-1, -1)
    ]
    candidates.append(axial_14)

    axial_15 = [
        (0, 0), (1, 0), (2, 0),
        (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1), (0, -1)
    ]
    candidates.append(axial_15)

    axial_16 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3), (3, 2)
    ]
    candidates.append(axial_16)

    axial_17 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (1, 2), (-1, 2), (0, 3)
    ]
    candidates.append(axial_17)

    axial_18 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2), (3, 2)
    ]
    candidates.append(axial_18)

    axial_19 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3), (3, 3)
    ]
    candidates.append(axial_19)

    axial_20 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (1, 1), (-1, 1)
    ]
    candidates.append(axial_20)

    axial_21 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (1, 2), (2, 2), (3, 2)
    ]
    candidates.append(axial_21)

    axial_22 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3)
    ]
    candidates.append(axial_22)

    axial_23 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_23)

    axial_24 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3),
        (0, 4)
    ]
    candidates.append(axial_24)

    axial_25 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_25)

    axial_26 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (1, 3), (2, 3)
    ]
    candidates.append(axial_26)

    axial_27 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (1, -2)
    ]
    candidates.append(axial_27)

    axial_28 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3), (2, 3),
        (1, 4), (2, 4)
    ]
    candidates.append(axial_28)

    axial_29 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, -1), (2, -1)
    ]
    candidates.append(axial_29)

    axial_30 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3),
        (1, -1)
    ]
    candidates.append(axial_30)

    axial_31 = [
        (0, 0), (1, 0), (2, 0),
        (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3),
        (1, -1), (2, -1)
    ]
    candidates.append(axial_31)

    axial_32 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (1, 2), (-1, 2), (0, -2)
    ]
    candidates.append(axial_32)

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def hex_vertices(cx, cy, side, angle):
        """Return vertices of a regular hexagon with center (cx,cy), side length, and rotation angle."""
        verts = []
        for k in range(6):
            ang = angle + k * math.pi / 3.0
            vx = cx + side * math.cos(ang)
            vy = cy + side * math.sin(ang)
            verts.append((vx, vy))
        return np.array(verts)

    def hex_normal_directions(angle):
        """Return the 6 outward normal directions of a hexagon rotated by angle."""
        normals = []
        for k in range(6):
            ang = angle + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))
        return np.array(normals)

    def min_distance_hex_hex(c1, a1, c2, a2):
        """
        Minimum distance between two unit hexagons with centers c1,c2 and angles a1,a2.
        Returns the minimum separation distance (0 if touching/overlapping).
        This is the radius of the Minkowski difference hexagon in the direction from c1 to c2.
        """
        # For two regular hexagons of side 1, the distance between centers when they just touch
        # depends on the relative orientation. The Minkowski sum of two unit hexagons is a hexagon
        # (or 12-gon) and the minimum distance is the support function of the difference.
        
        # A simpler approach: the minimum distance between two regular hexagons of side 1
        # with given orientations can be computed as the minimum over all 6+6 vertex-edge
        # combinations, but that's complex. 
        # 
        # For the LP formulation, we need linear constraints. If all hexagons have the same
        # orientation, the Minkowski difference is a regular hexagon and the constraint becomes
        # |n_k · (c_i - c_j)| <= apothem_diff for the 6 normal directions.
        #
        # For general orientations, the constraint is nonlinear. So we restrict to same orientation
        # for the LP approach.
        
        pass

    def calc_S_lp_same_orient(pts, beta):
        """
        LP for same-orientation case: all inner hexagons rotated by beta.
        Find center (cx,cy) and D such that all hexagons fit in outer hexagon.
        For same orientation, the constraint for each inner hex i at position p_i:
        For each of the 6 outer normals n_k (of the enclosing hexagon at angle beta):
          n_k · (p_i - c) + APOTHEM <= D  (far vertex of inner hex in direction n_k)
          n_k · (c - p_i) + APOTHEM <= D  (near vertex)
        This simplifies to: |n_k · (p_i - c)| + APOTHEM <= D
        So: -D + n_k·c <= -n_k·p_i + APOTHEM and -D - n_k·c <= n_k·p_i + APOTHEM
        Wait, let me redo.
        
        Inner hex i has center p_i, rotation beta. Its vertices are p_i + v_k where v_k are
        the 6 vertices of a unit hex at angle beta. The farthest vertex in direction n_k
        is p_i + max_j(n_k · v_j) where v_j are the 6 vertices.
        For a regular hex, max_j(n_k · v_j) = side * cos(0) = 1 if n_k aligns with a vertex direction.
        Actually, for a unit hex rotated by beta, the support function in direction n is:
        h(n) = max_k n · (side * cos(beta + k*pi/3), side * sin(beta + k*pi/3))
        This equals side if n is in the direction of a vertex, and side*cos(pi/6) = apothem if n is
        in the direction of a face normal.
        
        For the enclosing hexagon with rotation beta_outer, its normals are at angles
        beta_outer + pi/6 + k*pi/3.
        
        Let's assume beta_inner = beta_outer = beta. Then the normals of the outer hex are
        at the face-normal directions of the inner hexagons. So for each inner hex, the support
        in each outer normal direction n_k is exactly APOTHEM (the apothem of the unit hex).
        
        So the constraint becomes:
        n_k · (p_i - c) + APOTHEM <= D for all k, i
        n_k · (c - p_i) + APOTHEM <= D for all k, i
        
        Equivalently: |n_k · (p_i - c)| + APOTHEM <= D
        
        Variables: cx, cy, D
        Minimize D
        Subject to:
          For all k, i:
            n_kx*cx + n_ky*cy - D <= n_kx*p_ix + n_ky*p_iy + APOTHEM
            -n_kx*cx - n_ky*cy - D <= -n_kx*p_ix - n_ky*p_iy + APOTHEM
        """
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), None, None
        
        normals = hex_normal_directions(beta)
        
        c_obj = np.array([0.0, 0.0, 1.0])
        A_ub = []
        b_ub = []
        
        for k in range(6):
            nx, ny = normals[k]
            for px, py in pts:
                # n·(p - c) + APOTHEM <= D  =>  -nx*cx - ny*cy - D <= -n·p + APOTHEM
                A_ub.append([-nx, -ny, -1.0])
                b_ub.append(-(px * nx + py * ny) + APOTHEM)
                # n·(c - p) + APOTHEM <= D  =>  nx*cx + ny*cy - D <= n·p + APOTHEM
                A_ub.append([nx, ny, -1.0])
                b_ub.append(px * nx + py * ny + APOTHEM)
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        bounds = [(None, None), (None, None), (0, None)]
        
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                D = res.x[2]
                cx = res.x[0]
                cy = res.x[1]
                S = D / APOTHEM
                return S, (cx, cy), D
            else:
                return float('inf'), None, None
        except Exception:
            return float('inf'), None, None

    def calc_S_lp_general(pts, beta):
        """
        General LP for fixed beta (outer hex orientation), but inner hexagons can have
        arbitrary orientations. We use the vertex-based constraint:
        For each inner hex i with orientation theta_i, its 6 vertices must be inside
        the outer hex. The outer hex with center c and side S has the constraint:
        |n_k · (v - c)| <= S * APOTHEM for all 6 normals n_k.
        
        But theta_i is a variable, making this nonlinear. So this LP only works when
        orientations are fixed.
        
        Let's instead fix the inner orientations to 0 (or beta) and use the same-orient LP.
        """
        return calc_S_lp_same_orient(pts, beta)

    def calc_S_nm(pts, beta):
        """Nelder-Mead fallback for fixed beta (outer orientation), optimizing center."""
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), None, None
        
        normals = hex_normal_directions(beta)
        
        # For each inner hex at position p with rotation 0, the max projection in direction n_k
        # is max of n_k · (p + v_j) for the 6 vertices v_j of unit hex at rotation 0.
        # = n_k · p + h_inner(n_k) where h_inner is the support function of unit hex at rot 0.
        # For unit hex at rot 0, vertices are at angles 0, 60, 120, 180, 240, 300.
        # Support in direction angle phi: max_j cos(phi - j*pi/3) = cos(min_k |phi - k*pi/3|)
        # This is a piecewise cosine function.
        
        def support_unit_hex(angle, phi):
            """Support function of unit hex at rotation 'angle' in direction phi."""
            # Vertices at angle + k*pi/3
            vals = [angle + k * math.pi / 3.0 for k in range(6)]
            cos_vals = [math.cos(phi - v) for v in vals]
            return max(cos_vals)
        
        cx_center = (np.max(pts[:, 0]) + np.min(pts[:, 0])) / 2
        cy_center = (np.max(pts[:, 1]) + np.min(pts[:, 1])) / 2
        
        def D_of_center(cx, cy):
            max_proj = 0
            for k in range(6):
                ang = beta + math.pi / 6.0 + k * math.pi / 3.0
                nx, ny = math.cos(ang), math.sin(ang)
                for px, py in pts:
                    # Support of inner hex (rot 0) in direction ang
                    h_inner = support_unit_hex(0.0, ang)
                    proj = nx * (px - cx) + ny * (py - cy) + h_inner
                    max_proj = max(max_proj, proj)
                    proj2 = nx * (cx - px) + ny * (cy - py) + h_inner
                    max_proj = max(max_proj, proj2)
            return max_proj
        
        best_D = float('inf')
        best_c = (cx_center, cy_center)
        
        cx_range = np.max(pts[:, 0]) - np.min(pts[:, 0])
        cy_range = np.max(pts[:, 1]) - np.min(pts[:, 1])
        
        for dx in np.linspace(-cx_range*0.3, cx_range*0.3, 7):
            for dy in np.linspace(-cy_range*0.3, cy_range*0.3, 7):
                D = D_of_center(cx_center + dx, cy_center + dy)
                if D < best_D:
                    best_D = D
                    best_c = (cx_center + dx, cy_center + dy)
        
        try:
            res = minimize(D_of_center, best_c, method='Nelder-Mead',
                          options={'maxiter': 300, 'xatol': 1e-12, 'fatol': 1e-12})
            if res.fun < best_D:
                best_D = res.fun
                best_c = (res.x[0], res.x[1])
        except Exception:
            pass
        
        S = best_D / APOTHEM
        return S, best_c, best_D

    def calc_S(pts, beta):
        """Try LP first, fall back to NM."""
        S_lp, c_lp, D_lp = calc_S_lp_general(pts, beta)
        if S_lp < float('inf'):
            return S_lp, c_lp, D_lp
        return calc_S_nm(pts, beta)

    def optimize_shape(pts):
        best_S = float('inf')
        best_beta = 0.0
        best_center = None
        
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, None
        
        # Coarse grid search over beta (0 to 30 degrees, period of hex symmetry)
        for i in range(19):
            beta = i * (math.pi / 6) / 18
            S, center, D = calc_S(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = center
        
        # Refine beta
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                res = minimize_scalar(lambda b: calc_S(pts, b)[0], bounds=(0, math.pi / 6), method='bounded',
                                     options={'maxiter': 150, 'xatol': 1e-10})
                if res.fun < best_S:
                    best_S = res.fun
                    best_beta = res.x
                    _, best_center, _ = calc_S(pts, best_beta)
        except Exception:
            pass
        
        # Polish with alternating optimization
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                def objective(params):
                    cx, cy, beta = params
                    beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
                    normals = hex_normal_directions(beta)
                    max_proj = 0
                    for k in range(6):
                        ang = beta + math.pi / 6.0 + k * math.pi / 3.0
                        nx, ny = normals[k]
                        for px, py in pts:
                            h_inner = support_unit_hex(0.0, ang)
                            proj = nx * (px - cx) + ny * (py - cy) + h_inner
                            max_proj = max(max_proj, proj)
                            proj2 = nx * (cx - px) + ny * (cy - py) + h_inner
                            max_proj = max(max_proj, proj2)
                    return max_proj / APOTHEM
                
                cx, cy = best_center
                beta = best_beta
                
                for _ in range(4):
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    
                    def obj_c(params):
                        return objective([params[0], params[1], beta])
                    
                    res_c = minimize(obj_c, [cx, cy], method='Nelder-Mead',
                                     options={'maxiter': 400, 'xatol': 1e-10, 'fatol': 1e-10})
                    cx, cy = res_c.x
                    S_c = res_c.fun
                    
                    def obj_b(params):
                        return objective([cx, cy, params[0]])
                    
                    b_starts = np.linspace(0, math.pi/6, 15)
                    best_b = beta
                    best_S_b = S_c
                    for b0 in b_starts:
                        if abs(b0 - best_b) < 1e-3: continue
                        res_b = minimize_scalar(obj_b, bounds=(0, math.pi/6), method='bounded',
                                                options={'maxiter': 80, 'xatol': 1e-10})
                        if res_b.fun < best_S_b:
                            best_S_b = res_b.fun
                            best_b = res_b.x
                    beta = best_b
                    
                    if best_S_b < best_S:
                        best_S = best_S_b
                        best_center = (cx, cy)
                        best_beta = beta
                        
        except Exception:
            pass
            
        return best_S, best_beta, best_center

    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

    for axial in candidates:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        pts = axial_to_cartesian(axial)
        S, beta, center = optimize_shape(pts)
        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts
            best_global_beta = beta
            best_global_center = center

    if best_global_center is None or best_global_S == float('inf'):
        best_global_pts = axial_to_cartesian(candidates[0])
        best_global_center = (float(np.mean(best_global_pts[:, 0])), float(np.mean(best_global_pts[:, 1])))
        best_global_beta = 0.0
        max_proj = 0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            for px, py in best_global_pts:
                proj = nx * px + ny * py + 1.0
                max_proj = max(max_proj, proj)
                proj2 = nx * (-px) + ny * (-py) + 1.0
                max_proj = max(max_proj, proj2)
        best_global_S = max_proj / APOTHEM

    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)
    
    # Scale points slightly to avoid numerical issues at boundaries
    SCALE = 1.0 + 1e-12
    pts_scaled = []
    for x, y in best_global_pts:
        sx = cx + (x - cx) * SCALE
        sy = cy + (y - cy) * SCALE
        pts_scaled.append((sx, sy))
    
    pts_scaled = np.array(pts_scaled)
    
    max_proj = 0
    for k in range(6):
        ang = best_global_beta + math.pi / 6.0 + k * math.pi / 3.0
        nx, ny = math.cos(ang), math.sin(ang)
        for px, py in pts_scaled:
            h_inner = support_unit_hex(0.0, ang)
            proj = nx * (px - cx) + ny * (py - cy) + h_inner
            max_proj = max(max_proj, proj)
            proj2 = nx * (cx - px) + ny * (cy - py) + h_inner
            max_proj = max(max_proj, proj2)
    
    D_scaled = max_proj
    S_scaled = D_scaled / APOTHEM
    
    inner = []
    for x, y in pts_scaled:
        inner.append([float(x), float(y), float(angle_deg)])
        
    center = [float(cx), float(cy)]
    side = float(S_scaled)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation
