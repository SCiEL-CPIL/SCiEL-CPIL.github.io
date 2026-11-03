import math
import numpy as np
import time
from scipy.optimize import linprog, minimize, minimize_scalar
from itertools import product

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 145.0
SAFETY_MARGIN = 5.0
EPS = 1e-9

def _hex_verts(cx, cy, side, rot):
    """Return 6 vertices of a regular hexagon."""
    r = math.radians(rot)
    v = []
    for k in range(6):
        a = r + k * math.pi / 3.0
        v.append((cx + side * math.cos(a), cy + side * math.sin(a)))
    return v

def _seg_intersect(p1, p2, p3, p4):
    """Check if segments p1-p2 and p3-p4 intersect."""
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False

def _hex_overlap_precise(c1, s1, r1, c2, s2, r2, eps=1e-9):
    """Precise hexagon overlap check using SAT with all edges."""
    v1 = _hex_verts(c1[0], c1[1], s1, r1)
    v2 = _hex_verts(c2[0], c2[1], s2, r2)
    
    # Quick bounding box check
    x1_min = min(v[0] for v in v1)
    x1_max = max(v[0] for v in v1)
    y1_min = min(v[1] for v in v1)
    y1_max = max(v[1] for v in v1)
    x2_min = min(v[0] for v in v2)
    x2_max = max(v[0] for v in v2)
    y2_min = min(v[1] for v in v2)
    y2_max = max(v[1] for v in v2)
    if x1_max < x2_min - eps or x2_max < x1_min - eps or y1_max < y2_min - eps or y2_max < y1_min - eps:
        return False
    
    # SAT: check all 12 edge normals
    edges = []
    for k in range(6):
        for i in range(2):
            vlist = v1 if i == 0 else v2
            a = vlist[k]
            b = vlist[(k+1) % 6]
            ex = b[0] - a[0]
            ey = b[1] - a[1]
            elen = math.sqrt(ex*ex + ey*ey)
            if elen > 1e-12:
                edges.append((-ey/elen, ex/elen))
    
    for nx, ny in edges:
        proj1 = [v[0]*nx + v[1]*ny for v in v1]
        proj2 = [v[0]*nx + v[1]*ny for v in v2]
        if max(proj1) < min(proj2) - eps or max(proj2) < min(proj1) - eps:
            return False
    return True

def _hexagon_overlap(c1, c2, s1, s2, rot1, rot2):
    """Check if two hexagons overlap using separating axis theorem."""
    dx = c1[0] - c2[0]
    dy = c1[1] - c2[1]
    dist = math.sqrt(dx*dx + dy*dy)
    # Quick reject: if distance >= sum of circumradii, no overlap
    circum1 = s1
    circum2 = s2
    if dist >= circum1 + circum2:
        return False
    # For hexagons, the minimum distance for non-overlap depends on orientation
    # Use a conservative check: if dist < (s1+s2)*APOTHEM, definitely overlap
    # If dist >= s1 + s2, definitely no overlap
    if dist >= s1 + s2:
        return False
    # More precise: check all 12 axes (6 from each hexagon)
    axes = []
    for k in range(6):
        a1 = rot1 + k * math.pi / 3.0
        a2 = rot2 + k * math.pi / 3.0
        axes.append((math.cos(a1), math.sin(a1)))
        axes.append((math.cos(a2), math.sin(a2)))
    # Also check the axis between centers
    if dist > 1e-10:
        axes.append((dx/dist, dy/dist))
    
    for nx, ny in axes:
        # Project both hexagons onto this axis
        proj1 = 0
        for k in range(6):
            ang = rot1 + k * math.pi / 3.0
            vx = s1 * math.cos(ang)
            vy = s1 * math.sin(ang)
            p = vx * nx + vy * ny
            proj1 = max(proj1, p)
        proj1 += c1[0]*nx + c1[1]*ny
        proj1_min = c1[0]*nx + c1[1]*ny - proj1
        
        proj2 = 0
        for k in range(6):
            ang = rot2 + k * math.pi / 3.0
            vx = s2 * math.cos(ang)
            vy = s2 * math.sin(ang)
            p = vx * nx + vy * ny
            proj2 = max(proj2, p)
        proj2 += c2[0]*nx + c2[1]*ny
        proj2_min = c2[0]*nx + c2[1]*ny - proj2
        
        if proj1_min > proj2 + proj2 + 1e-9 or proj2_min > proj1 + proj1 + 1e-9:
            return False
    return True

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def hexagon_vertices(cx, cy, side, rotation_deg):
        rot = math.radians(rotation_deg)
        verts = []
        for k in range(6):
            angle = rot + k * math.pi / 3.0
            x = cx + side * math.cos(angle)
            y = cy + side * math.sin(angle)
            verts.append((x, y))
        return verts

    def point_in_hexagon(px, py, cx, cy, side, rotation_deg):
        rot = math.radians(rotation_deg)
        dx = px - cx
        dy = py - cy
        dx_local = dx * math.cos(rot) + dy * math.sin(rot)
        dy_local = -dx * math.sin(rot) + dy * math.cos(rot)
        apothem = side * APOTHEM
        for k in range(6):
            angle = k * math.pi / 3.0
            nx = math.cos(angle)
            ny = math.sin(angle)
            dot = dx_local * nx + dy_local * ny
            if dot > apothem + 1e-10:
                return False
        return True

    def hexagons_disjoint(c1, c2, s1, s2, rot1_deg, rot2_deg):
        dx = c1[0] - c2[0]
        dy = c1[1] - c2[1]
        dist = math.sqrt(dx*dx + dy*dy)
        return dist >= (s1 + s2) - 1e-9

    # Generate candidate configurations (axial coordinates for aligned hexagons)
    candidates = []
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(1,1),(-1,2),(-2,1),(2,-1)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,1),(3,1)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(1,2),(2,2),(3,2)])
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3)])
    candidates.append([(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(2,0),(-2,0),(0,2),(0,-2)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(1,1),(2,1),(3,1),(4,1),(5,1)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(-2,2),(-1,3),(1,-2)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3)])
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),(1,3)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(0,2),(1,2),(0,3),(1,3),(0,4)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(1,2)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,-1)])
    candidates.append([(0,0),(1,0),(2,0),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,1),(3,1),(0,-1)])
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(3,2)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(1,2),(-1,2),(0,3)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(1,2),(2,2),(3,2)])
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(3,3)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,1)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)])
    candidates.append([(0,0),(1,0),(0,1),(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(0,4)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),(1,3),(2,3)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(1,-2)])
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(0,3),(1,3),(2,3),(1,4),(2,4)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3),(2,3)])
    candidates.append([(0,0),(1,0),(2,0),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3),(0,-1),(3,1)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,-1)])
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(1,4),(2,4)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3)])
    candidates.append([(2,0),(-2,0),(0,2),(0,-2),(2,-2),(-2,2),(1,0),(-1,0),(0,1),(0,-1),(1,1)])
    candidates.append([(2,0),(-2,0),(0,2),(0,-2),(2,-2),(-2,2),(1,0),(-1,0),(1,1),(-1,1),(0,1)])
    candidates.append([(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(2,0),(-2,0),(0,2),(0,-2)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(4,0),(-1,1),(0,1),(1,1),(2,1),(3,1),(4,1)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(0,2),(1,2),(0,3),(1,3)])
    candidates.append([(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(2,0),(-2,0),(0,2),(0,-2),(2,-2)])
    
    # New family: hexagons arranged in a ring with 30-degree offset from outer
    # These are axial coords but we'll rotate them
    ring_candidates = [
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(-2,2),(-1,3),(1,-2)],
    ]
    for rc in ring_candidates:
        candidates.append(rc)

    def calc_S_for_beta(pts, beta):
        """LP for minimum enclosing hexagon with given rotation (vectorized)."""
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), None, None

        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])

        n_pts = len(pts)
        n_constraints = 12 * n_pts
        A_ub = np.zeros((n_constraints, 3))
        b_ub = np.zeros(n_constraints)

        for k in range(6):
            nx, ny = normals[k]
            projs = pts[:, 0] * nx + pts[:, 1] * ny
            start = 2 * k * n_pts
            A_ub[start:start+n_pts, 0] = -nx
            A_ub[start:start+n_pts, 1] = -ny
            A_ub[start:start+n_pts, 2] = -1.0
            b_ub[start:start+n_pts] = -projs
            A_ub[start+n_pts:start+2*n_pts, 0] = nx
            A_ub[start+n_pts:start+2*n_pts, 1] = ny
            A_ub[start+n_pts:start+2*n_pts, 2] = -1.0
            b_ub[start+n_pts:start+2*n_pts] = projs

        c_obj = np.array([0.0, 0.0, 1.0])
        bounds = [(None, None), (None, None), (0, None)]

        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                D = res.x[2]
                cx = res.x[0]
                cy = res.x[1]
                S = (D + APOTHEM) / APOTHEM
                return S, (cx, cy), D
            else:
                return float('inf'), None, None
        except Exception:
            return float('inf'), None, None

    def exact_S(pts, cx, cy, beta):
        """Exact computation of S for given center and rotation."""
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        pts_centered = pts - np.array([cx, cy])
        projs = pts_centered @ normals.T
        D = np.max(np.abs(projs))
        return (D + APOTHEM) / APOTHEM

    def smooth_calc_S(pts, cx, cy, beta, temperature=0.05):
        """Smooth surrogate using log-sum-exp."""
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        pts_centered = pts - np.array([cx, cy])
        projs = pts_centered @ normals.T
        all_projs = np.concatenate([projs.ravel(), (-projs).ravel()])
        max_val = np.max(all_projs)
        if max_val > 200:
            return float('inf')
        shifted = (all_projs - max_val) / temperature
        shifted = np.clip(shifted, -50, 50)
        log_sum = np.log(np.sum(np.exp(shifted))) + max_val / temperature
        D = temperature * log_sum
        return (D + APOTHEM) / APOTHEM

    def optimize_orientation_only(pts):
        """1D scalar minimization over outer hexagon orientation beta."""
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, None

        best_S = float('inf')
        best_beta = 0.0
        best_center = None

        n_grid = 61
        for i in range(n_grid):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            beta = i * (math.pi / 6) / (n_grid - 1)
            S, center, D = calc_S_for_beta(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = center

        if best_center is None:
            return float('inf'), 0.0, None

        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                window = (math.pi / 6) / (n_grid - 1)
                lo = max(0.0, best_beta - 2 * window)
                hi = min(math.pi / 6, best_beta + 2 * window)
                
                def obj_1d(b):
                    S, _, _ = calc_S_for_beta(pts, b)
                    return S
                
                res = minimize_scalar(obj_1d, bounds=(lo, hi), method='bounded',
                                     options={'maxiter': 100, 'xatol': 1e-12})
                if res.fun < best_S:
                    best_S = res.fun
                    best_beta = res.x
                    _, best_center, _ = calc_S_for_beta(pts, best_beta)
        except Exception:
            pass

        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                window = (math.pi / 6) / (n_grid - 1) * 0.5
                lo = max(0.0, best_beta - window)
                hi = min(math.pi / 6, best_beta + window)
                
                def obj_1d(b):
                    S, _, _ = calc_S_for_beta(pts, b)
                    return S
                
                res = minimize_scalar(obj_1d, bounds=(lo, hi), method='bounded',
                                     options={'maxiter': 200, 'xatol': 1e-14})
                if res.fun < best_S:
                    best_S = res.fun
                    best_beta = res.x
                    _, best_center, _ = calc_S_for_beta(pts, best_beta)
        except Exception:
            pass

        return best_S, best_beta, best_center

    def optimize_shape(pts, allow_inner_rotation=False):
        """Hybrid optimization for fixed point configuration."""
        best_S = float('inf')
        best_beta = 0.0
        best_center = None

        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, None

        S1, beta1, center1 = optimize_orientation_only(pts)
        if center1 is not None:
            best_S = S1
            best_beta = beta1
            best_center = center1

        if best_center is None:
            return float('inf'), 0.0, None

        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                temperatures = [0.05, 0.03, 0.01, 0.005]
                for temp in temperatures:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break

                    def smooth_obj(params, t=temp):
                        cx, cy, beta = params
                        beta = ((beta % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                        return smooth_calc_S(pts, cx, cy, beta, temperature=t)

                    start_betas = [best_beta, best_beta + 0.01, best_beta - 0.01,
                                   best_beta + 0.03, best_beta - 0.03,
                                   best_beta + 0.05, best_beta - 0.05]
                    for sb in start_betas:
                        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                            break
                        init = [best_center[0], best_center[1], sb]
                        try:
                            res = minimize(smooth_obj, init, method='Nelder-Mead',
                                         options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-10})
                            if res.fun < 1e10:
                                cx_r, cy_r, beta_r = res.x
                                beta_r = ((beta_r % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                                S_exact = exact_S(pts, cx_r, cy_r, beta_r)
                                if S_exact < best_S:
                                    best_S = S_exact
                                    best_beta = beta_r
                                    best_center = (cx_r, cy_r)
                        except Exception:
                            pass
        except Exception:
            pass

        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                def exact_obj(params):
                    cx, cy, beta = params
                    beta = ((beta % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                    return exact_S(pts, cx, cy, beta)

                delta_vals = [0.0, 0.0005, -0.0005, 0.001, -0.001, 0.002, -0.002,
                              0.003, -0.003, 0.005, -0.005, 0.008, -0.008]
                polish_starts = []
                for db in delta_vals:
                    for dcx in [0.0, 0.0005, -0.0005, 0.001, -0.001, 0.002, -0.002]:
                        for dcy in [0.0, 0.0005, -0.0005, 0.001, -0.001, 0.002, -0.002]:
                            polish_starts.append([
                                best_center[0] + dcx,
                                best_center[1] + dcy,
                                best_beta + db
                            ])
                seen = set()
                unique_starts = []
                for p in polish_starts:
                    key = (round(p[0], 7), round(p[1], 7), round(p[2], 7))
                    if key not in seen:
                        seen.add(key)
                        unique_starts.append(p)

                # Sort by distance from current best to prioritize nearby starts
                unique_starts.sort(key=lambda p: (p[0]-best_center[0])**2 + (p[1]-best_center[1])**2 + (p[2]-best_beta)**2)

                for init in unique_starts:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    try:
                        res = minimize(exact_obj, init, method='Nelder-Mead',
                                     options={'maxiter': 5000, 'xatol': 1e-13, 'fatol': 1e-13})
                        if res.fun < best_S:
                            best_S = res.fun
                            best_center = (res.x[0], res.x[1])
                            best_beta = ((res.x[2] % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                    except Exception:
                        pass
        except Exception:
            pass

        # Final high-precision polish: use SLSQP with tighter bounds
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                def exact_obj_slsqp(params):
                    cx, cy, beta = params
                    beta = ((beta % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                    return exact_S(pts, cx, cy, beta)

                # Try a few final SLSQP refinements
                for db in [0.0, 0.001, -0.001]:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    init = [best_center[0], best_center[1], best_beta + db]
                    try:
                        res = minimize(exact_obj_slsqp, init, method='SLSQP',
                                     options={'maxiter': 1000, 'ftol': 1e-15})
                        if res.fun < best_S:
                            best_S = res.fun
                            best_center = (res.x[0], res.x[1])
                            best_beta = ((res.x[2] % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                    except Exception:
                        pass
        except Exception:
            pass

        return best_S, best_beta, best_center

    # Evaluate all candidates with lattice-based optimization
    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

    def quick_estimate(axial_list):
        pts = axial_to_cartesian(axial_list)
        angles = math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        cx, cy = np.mean(pts[:, 0]), np.mean(pts[:, 1])
        pts_c = pts - np.array([cx, cy])
        projs = pts_c @ normals.T
        D = np.max(np.abs(projs))
        return (D + APOTHEM) / APOTHEM

    scored_candidates = []
    for axial in candidates:
        est = quick_estimate(axial)
        scored_candidates.append((est, axial))
    scored_candidates.sort(key=lambda x: x[0])

    for idx, (est, axial) in enumerate(scored_candidates):
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
            projs = best_global_pts[:, 0] * nx + best_global_pts[:, 1] * ny
            max_proj = max(max_proj, np.max(projs), -np.min(projs))
        best_global_S = (max_proj + APOTHEM) / APOTHEM

    # === NEW: Try rotating individual inner hexagons to improve packing ===
    def try_inner_rotations(pts, beta, center, max_rot_trials=200):
        """Try small rotations of individual hexagons to improve outer S."""
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return None
        
        best_S = float('inf')
        best_pts = pts.copy()
        best_rotations = np.zeros(len(pts))
        best_beta = beta
        best_center = center
        
        n = len(pts)
        # Try rotating each hexagon by small angles
        angle_options = [0, 15, 30, 45, 60]  # degrees
        angle_rads = [math.radians(a) for a in angle_options]
        
        # Try single hexagon rotations
        for i in range(n):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            for ar in angle_rads:
                if ar == 0:
                    continue
                new_pts = pts.copy()
                # Rotate hexagon i around its center
                cx_i, cy_i = pts[i]
                # For a hexagon rotated by ar, its vertices change but center stays
                # We need to check if it still fits and doesn't overlap
                # For now, just try it and compute S
                new_rotations = np.zeros(n)
                new_rotations[i] = ar
                
                # Compute new S with this rotation
                # The hexagon's effective extent changes with rotation
                # For a unit hexagon rotated by ar, the max projection onto any axis
                # is the same as unrotated (by symmetry), so S shouldn't change
                # unless the hexagon moves. So this approach won't help directly.
                # Instead, we should move hexagons to exploit the rotation.
                pass
        
        return None

    # Skip the inner rotation approach as it's complex; instead, do a final polish
    # with a finer grid on the outer rotation for the best solution
    
    if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN and best_global_pts is not None:
        # Fine polish on outer rotation
        pts = best_global_pts
        cx, cy = best_global_center
        beta_coarse = best_global_beta
        
        # Fine grid around best_beta
        fine_grid = np.linspace(max(0, beta_coarse - 0.02), min(math.pi/6, beta_coarse + 0.02), 41)
        for fb in fine_grid:
            angles = fb + math.pi/6.0 + np.arange(6) * math.pi/3.0
            normals = np.column_stack([np.cos(angles), np.sin(angles)])
            n_pts = len(pts)
            n_constraints = 12 * n_pts
            A_ub = np.zeros((n_constraints, 3))
            b_ub = np.zeros(n_constraints)
            for k in range(6):
                nx, ny = normals[k]
                projs = pts[:, 0] * nx + pts[:, 1] * ny
                start = 2 * k * n_pts
                A_ub[start:start+n_pts, 0] = -nx
                A_ub[start:start+n_pts, 1] = -ny
                A_ub[start:start+n_pts, 2] = -1.0
                b_ub[start:start+n_pts] = -projs
                A_ub[start+n_pts:start+2*n_pts, 0] = nx
                A_ub[start+n_pts:start+2*n_pts, 1] = ny
                A_ub[start+n_pts:start+2*n_pts, 2] = -1.0
                b_ub[start+n_pts:start+2*n_pts] = projs
            c_obj = np.array([0.0, 0.0, 1.0])
            bounds = [(None, None), (None, None), (0, None)]
            try:
                res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
                if res.status == 0:
                    D = res.x[2]
                    S_fine = (D + APOTHEM) / APOTHEM
                    if S_fine < best_global_S:
                        best_global_S = S_fine
                        best_global_beta = fb
                        best_global_center = (res.x[0], res.x[1])
            except:
                pass

    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)

    # Check if we have per-hexagon rotations (from force-directed)
    # For now, all inner hexagons use the same rotation as before
    # The force-directed result may have different rotations, but we need to track them
    
    SCALE = 1.0 + 1e-14
    pts_scaled = best_global_pts.copy()
    for i in range(len(pts_scaled)):
        pts_scaled[i, 0] = cx + (pts_scaled[i, 0] - cx) * SCALE
        pts_scaled[i, 1] = cy + (pts_scaled[i, 1] - cy) * SCALE

    max_proj = 0
    for k in range(6):
        ang = best_global_beta + math.pi / 6.0 + k * math.pi / 3.0
        nx, ny = math.cos(ang), math.sin(ang)
        projs = (pts_scaled[:, 0] - cx) * nx + (pts_scaled[:, 1] - cy) * ny
        max_proj = max(max_proj, np.max(projs), -np.min(projs))

    D_scaled = max_proj
    S_scaled = (D_scaled + APOTHEM) / APOTHEM

    inner = []
    for x, y in pts_scaled:
        inner.append([float(x), float(y), float(angle_deg)])

    center = [float(cx), float(cy)]
    side = float(S_scaled)
    rotation = float(angle_deg)

    return inner, center, side, rotation
