import math
import numpy as np
import time
from scipy.optimize import linprog, minimize, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 8.0
EPS = 1e-9
_TIME_DEADLINE = START_TIME + TIME_LIMIT - SAFETY_MARGIN

def _hex_vertices(cx, cy, s, rot_deg):
    """Return 6 vertices of a regular hexagon."""
    a = math.radians(rot_deg)
    vs = []
    for k in range(6):
        ang = a + k * math.pi / 3.0
        vs.append((cx + s * math.cos(ang), cy + s * math.sin(ang)))
    return vs

def _point_in_hex(px, py, cx, cy, s, rot_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy) with side s."""
    dx, dy = px - cx, py - cy
    a = math.radians(rot_deg)
    ca, sa = math.cos(a), math.sin(a)
    lx, ly = ca * dx + sa * dy, -sa * dx + ca * dy
    # Hexagon with flat-top orientation: |ly| <= apothem, |lx| <= s, and combined constraint
    ap = math.sqrt(3) / 2.0 * s
    if abs(ly) > ap + 1e-9:
        return False
    if abs(lx) > s + 1e-9:
        return False
    # The hexagon is intersection of 6 half-planes
    # For flat-top hex: vertices at angles 0,60,120,180,240,300
    for k in range(6):
        ang = k * math.pi / 3.0
        nx, ny = math.cos(ang), math.sin(ang)
        if lx * nx + ly * ny > s * math.cos(math.pi/6) + 1e-9:
            return False
    return True

def _hexes_disjoint(v1, v2, tol=1e-9):
    """Check if two hexagons (lists of 6 vertices) are disjoint (no interior overlap)."""
    # Two convex polygons are disjoint if no edge of one intersects any edge of other
    # AND neither is contained in the other
    def seg_intersect(p1, p2, p3, p4):
        x1,y1 = p1; x2,y2 = p2; x3,y3 = p3; x4,y4 = p4
        d1x, d1y = x2-x1, y2-y1
        d2x, d2y = x4-x3, y4-y3
        denom = d1x*d2y - d1y*d2x
        if abs(denom) < 1e-15:
            return False
        t = ((x3-x1)*d2y - (y3-y1)*d2x) / denom
        u = ((x3-x1)*d1y - (y3-y1)*d1x) / denom
        return -tol <= t <= 1+tol and -tol <= u <= 1+tol

    for i in range(6):
        for j in range(6):
            if seg_intersect(v1[i], v1[(i+1)%6], v2[j], v2[(j+1)%6]):
                return False
    # Check containment: center of v1 in v2 or vice versa
    cx1 = sum(p[0] for p in v1)/6; cy1 = sum(p[1] for p in v1)/6
    cx2 = sum(p[0] for p in v2)/6; cy2 = sum(p[1] for p in v2)/6
    if _point_in_hex(cx1, cy1, cx2, cy2, 1.0, 0.0) or _point_in_hex(cx2, cy2, cx1, cy1, 1.0, 0.0):
        return False
    return True

def construct():
    if time.time() > _TIME_DEADLINE:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Define candidate point sets (axial coordinates)
    candidates = []

    # Standard hexagonal cluster (7 center + 6 ring + 4 outer)
    axial_1 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ]
    candidates.append(axial_1)

    # Compact diamond
    axial_diamond = [
        (0, 0), (1, 0), (-1, 0),
        (0, 1), (1, 1), (-1, 1),
        (0, 2), (1, 2),
        (-1, -1), (0, -1), (1, -1)
    ]
    candidates.append(axial_diamond)

    # T-shaped
    axial_t = [
        (0, 0), (1, 0), (-1, 0),
        (0, 1), (1, 1), (-1, 1),
        (0, 2), (1, 2), (-1, 2),
        (0, 3), (1, 3)
    ]
    candidates.append(axial_t)

    # L-shaped dense
    axial_l = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_l)

    # Compact 4x3 center
    axial_c43 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3)
    ]
    candidates.append(axial_c43)

    # Cross-shaped
    axial_2 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_2)

    # 3x3 block with two extras
    axial_3 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1)
    ]
    candidates.append(axial_3)

    # 4x2 + 3
    axial_4 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2)
    ]
    candidates.append(axial_4)

    # Staircase
    axial_5 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_5)

    # Compact 3x3 + 2
    axial_6 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3)
    ]
    candidates.append(axial_6)

    # Star with long arms
    axial_7 = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_7)

    # Elongated
    axial_8 = [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)
    ]
    candidates.append(axial_8)

    # Asymmetric star
    axial_9 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (-2, 2), (-1, 3), (1, -2)
    ]
    candidates.append(axial_9)

    # Compact with top row
    axial_10 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3)
    ]
    candidates.append(axial_10)

    # Wide base
    axial_11 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (1, 3)
    ]
    candidates.append(axial_11)

    # Tall narrow
    axial_12 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3),
        (0, 4)
    ]
    candidates.append(axial_12)

    # Star with asymmetric outer
    axial_13 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (1, 2)
    ]
    candidates.append(axial_13)

    # Star with diagonal extras
    axial_14 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (1, 1), (-1, -1)
    ]
    candidates.append(axial_14)

    # Dense cluster
    axial_15 = [
        (0, 0), (1, 0), (2, 0),
        (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1), (0, -1)
    ]
    candidates.append(axial_15)

    # Zigzag
    axial_16 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3), (3, 2)
    ]
    candidates.append(axial_16)

    # Star with top cluster
    axial_17 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (1, 2), (-1, 2), (0, 3)
    ]
    candidates.append(axial_17)

    # Trapezoid
    axial_18 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2), (3, 2)
    ]
    candidates.append(axial_18)

    # Wide top
    axial_19 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3), (3, 3)
    ]
    candidates.append(axial_19)

    # Star with inner diagonal
    axial_20 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (1, 1), (-1, 1)
    ]
    candidates.append(axial_20)

    # 4x3 minus corners
    axial_21 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (1, 2), (2, 2), (3, 2)
    ]
    candidates.append(axial_21)

    # Wide with top
    axial_22 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3)
    ]
    candidates.append(axial_22)

    # Symmetric star
    axial_23 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_23)

    # Very tall
    axial_24 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3),
        (0, 4)
    ]
    candidates.append(axial_24)

    # 3x3 + row
    axial_25 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_25)

    # Dense 4-wide
    axial_26 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (1, 3), (2, 3)
    ]
    candidates.append(axial_26)

    # Star with asymmetric bottom
    axial_27 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (1, -2)
    ]
    candidates.append(axial_27)

    # Tall with top row
    axial_28 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3), (2, 3),
        (1, 4), (2, 4)
    ]
    candidates.append(axial_28)

    # Additional diverse configurations
    axial_29 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3)
    ]
    candidates.append(axial_29)

    axial_30 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_30)

    axial_31 = [
        (0, 0), (1, 0), (2, 0),
        (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3),
        (0, -1), (3, 1)
    ]
    candidates.append(axial_31)

    axial_32 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2)
    ]
    candidates.append(axial_32)

    axial_33 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (1, 1), (-1, -1)
    ]
    candidates.append(axial_33)

    axial_34 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3),
        (1, 4), (2, 4)
    ]
    candidates.append(axial_34)

    axial_35 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3)
    ]
    candidates.append(axial_35)

    axial_36 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3)
    ]
    candidates.append(axial_36)

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def calc_S_for_beta(pts, beta):
        """Vectorized LP-based evaluation of minimum enclosing hexagon side length."""
        if time.time() > _TIME_DEADLINE:
            return float('inf'), None, None

        # Precompute normals
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])  # (6, 2)

        n_pts = len(pts)
        
        # Build LP: minimize D s.t. for each normal n_k and point p_i:
        #   n_k . (p_i - c) <= D  =>  -n_k . c - D <= -n_k . p_i
        #  -n_k . (p_i - c) <= D  =>   n_k . c - D <=  n_k . p_i
        # Variables: [cx, cy, D]

        # For each (normal, point) pair, we have two constraints
        # Total: 12 * n_pts constraints
        n_constraints = 12 * n_pts
        A_ub = np.zeros((n_constraints, 3))
        b_ub = np.zeros(n_constraints)

        # Vectorized construction
        # For each normal k, project all points
        # projs[i, k] = n_k . p_i
        projs = pts @ normals.T  # (n_pts, 6)
        
        # Constraint 1 for all (i, k): -nx*cx - ny*cy - D <= -projs[i,k]
        # Constraint 2 for all (i, k): nx*cx + ny*cy - D <= projs[i,k]
        
        # Flatten: for k in 0..5, i in 0..n_pts-1
        # A_ub row for constraint 1 (k, i): [-nx_k, -ny_k, -1], b = -projs[i,k]
        # A_ub row for constraint 2 (k, i): [nx_k, ny_k, -1], b = projs[i,k]
        
        # Build using broadcasting
        # normals shape (6, 2), projs shape (n_pts, 6)
        # We need to interleave: for each k, all i for constraint 1, then all i for constraint 2
        
        # Simpler: build in blocks
        idx = 0
        for k in range(6):
            nx, ny = normals[k]
            proj_col = projs[:, k]  # (n_pts,)
            # Constraint 1: -nx*cx - ny*cy - D <= -proj_col
            A_ub[idx:idx+n_pts, 0] = -nx
            A_ub[idx:idx+n_pts, 1] = -ny
            A_ub[idx:idx+n_pts, 2] = -1.0
            b_ub[idx:idx+n_pts] = -proj_col
            idx += n_pts
            # Constraint 2: nx*cx + ny*cy - D <= proj_col
            A_ub[idx:idx+n_pts, 0] = nx
            A_ub[idx:idx+n_pts, 1] = ny
            A_ub[idx:idx+n_pts, 2] = -1.0
            b_ub[idx:idx+n_pts] = proj_col
            idx += n_pts

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

    def smooth_calc_S(pts, beta, cx, cy, temperature=0.1):
        """Smooth surrogate using log-sum-exp for differentiable optimization."""
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        
        pts_centered = pts - np.array([cx, cy])
        projs = pts_centered @ normals.T  # (n_pts, 6)
        
        all_projs = np.concatenate([projs.ravel(), (-projs).ravel()])
        
        max_val = np.max(all_projs)
        if max_val > 100:
            return float('inf'), (cx, cy), float('inf')
        
        shifted = (all_projs - max_val) / temperature
        shifted = np.clip(shifted, -50, 50)
        log_sum = np.log(np.sum(np.exp(shifted))) + max_val / temperature
        D = temperature * log_sum
        S = (D + APOTHEM) / APOTHEM
        return S, (cx, cy), D

    def optimize_shape(pts):
        """Multi-stage optimization: grid search, scalar refinement, smooth surrogate, exact polish."""
        best_S = float('inf')
        best_beta = 0.0
        best_center = None
        
        if time.time() > _TIME_DEADLINE:
            return float('inf'), 0.0, None
        
        # Stage 1: Coarse grid search over beta in [0, pi/6]
        n_grid = 49
        for i in range(n_grid):
            if time.time() > _TIME_DEADLINE:
                break
            beta = i * (math.pi / 6) / (n_grid - 1)
            S, center, D = calc_S_for_beta(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = center
        
        if best_center is None:
            return float('inf'), 0.0, None
        
        # Stage 2: Refine beta with bounded scalar optimization
        try:
            if time.time() < _TIME_DEADLINE:
                res = minimize_scalar(lambda b: calc_S_for_beta(pts, b)[0], 
                                     bounds=(0, math.pi / 6), method='bounded',
                                     options={'maxiter': 300, 'xatol': 1e-12})
                if res.fun < best_S:
                    best_S = res.fun
                    best_beta = res.x
                    _, best_center, _ = calc_S_for_beta(pts, best_beta)
        except Exception:
            pass
        
        # Stage 3: Smooth surrogate optimization with adaptive temperature
        try:
            if time.time() < _TIME_DEADLINE:
                temperatures = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
                # Generate diverse start points around best
                start_betas = [best_beta]
                for offset in [0.03, -0.03, 0.06, -0.06, 0.1, -0.1, 0.15, -0.15]:
                    sb = best_beta + offset
                    sb = ((sb % (math.pi/6)) + math.pi/6) % (math.pi/6)
                    start_betas.append(sb)
                start_betas.extend([0.0, math.pi/12, math.pi/6 * 0.25, math.pi/6 * 0.75])
                
                for temp in temperatures:
                    if time.time() > _TIME_DEADLINE:
                        break
                    for sb in start_betas:
                        if time.time() > _TIME_DEADLINE:
                            break
                        
                        def smooth_obj(params, t=temp):
                            cx, cy, beta = params
                            beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
                            S, _, _ = smooth_calc_S(pts, beta, cx, cy, temperature=t)
                            return S
                        
                        init = [best_center[0], best_center[1], sb]
                        res = minimize(smooth_obj, init, method='Nelder-Mead',
                                     options={'maxiter': 500, 'xatol': 1e-11, 'fatol': 1e-11})
                        
                        if res.fun < 1e10:
                            cx_r, cy_r, beta_r = res.x
                            beta_r = ((beta_r % (math.pi/6)) + math.pi/6) % (math.pi/6)
                            S_exact, c_exact, _ = calc_S_for_beta(pts, beta_r)
                            if S_exact < best_S:
                                best_S = S_exact
                                best_beta = beta_r
                                best_center = c_exact
        except Exception:
            pass
        
        # Stage 4: Final high-precision polish using exact objective
        try:
            if time.time() < _TIME_DEADLINE:
                def objective(params):
                    cx, cy, beta = params
                    beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
                    angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
                    normals = np.column_stack([np.cos(angles), np.sin(angles)])
                    
                    pts_centered = pts - np.array([cx, cy])
                    projs = pts_centered @ normals.T
                    max_proj = np.max(np.abs(projs))
                    
                    S = (max_proj + APOTHEM) / APOTHEM
                    return S
                
                # Generate more diverse polish starting points
                polish_starts = [[best_center[0], best_center[1], best_beta]]
                for db in [0.01, -0.01, 0.02, -0.02, 0.03, -0.03, 0.05, -0.05]:
                    bb = ((best_beta + db) % (math.pi/6) + math.pi/6) % (math.pi/6)
                    polish_starts.append([best_center[0], best_center[1], bb])
                for dc in [0.01, -0.01, 0.02, -0.02]:
                    polish_starts.append([best_center[0] + dc, best_center[1], best_beta])
                    polish_starts.append([best_center[0], best_center[1] + dc, best_beta])
                
                for init in polish_starts:
                    if time.time() > _TIME_DEADLINE:
                        break
                    res = minimize(objective, init, method='Nelder-Mead',
                                 options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-14})
                    if res.fun < best_S:
                        best_S = res.fun
                        best_center = (res.x[0], res.x[1])
                        best_beta = ((res.x[2] % (math.pi/6)) + math.pi/6) % (math.pi/6)
        except Exception:
            pass
            
        # Stage 5: Ultra-precise final polish with tighter tolerances
        try:
            if time.time() < _TIME_DEADLINE:
                def precise_objective(params):
                    cx, cy, beta = params
                    beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
                    angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
                    normals = np.column_stack([np.cos(angles), np.sin(angles)])
                    
                    pts_centered = pts - np.array([cx, cy])
                    projs = pts_centered @ normals.T
                    max_proj = np.max(np.abs(projs))
                    
                    S = (max_proj + APOTHEM) / APOTHEM
                    return S
                
                # Very fine polish around current best
                fine_starts = [[best_center[0], best_center[1], best_beta]]
                for db in [0.005, -0.005, 0.002, -0.002, 0.001, -0.001]:
                    bb = ((best_beta + db) % (math.pi/6) + math.pi/6) % (math.pi/6)
                    fine_starts.append([best_center[0], best_center[1], bb])
                for dc in [0.005, -0.005, 0.002, -0.002]:
                    fine_starts.append([best_center[0] + dc, best_center[1], best_beta])
                    fine_starts.append([best_center[0], best_center[1] + dc, best_beta])
                
                for init in fine_starts:
                    if time.time() > _TIME_DEADLINE:
                        break
                    res = minimize(precise_objective, init, method='Nelder-Mead',
                                 options={'maxiter': 10000, 'xatol': 1e-15, 'fatol': 1e-15})
                    if res.fun < best_S:
                        best_S = res.fun
                        best_center = (res.x[0], res.x[1])
                        best_beta = ((res.x[2] % (math.pi/6)) + math.pi/6) % (math.pi/6)
        except Exception:
            pass
            
        return best_S, best_beta, best_center

    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

    # Sort candidates by a quick heuristic: prefer more compact shapes first
    def _compactness_score(axial_list):
        pts = axial_to_cartesian(axial_list)
        cx = np.mean(pts[:, 0])
        cy = np.mean(pts[:, 1])
        max_r = np.max(np.sqrt((pts[:, 0] - cx)**2 + (pts[:, 1] - cy)**2))
        return max_r

    sorted_candidates = sorted(candidates, key=_compactness_score)

    for axial in sorted_candidates:
        if time.time() > _TIME_DEADLINE:
            break
        pts = axial_to_cartesian(axial)
        S, beta, center = optimize_shape(pts)
        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts
            best_global_beta = beta
            best_global_center = center

    if best_global_center is None or best_global_S == float('inf'):
        # Fallback: use first candidate with simple centering
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

    # Stage 6: Global refinement - try to improve the best solution with more iterations
    try:
        if time.time() < _TIME_DEADLINE and best_global_center is not None:
            def final_objective(params):
                cx, cy, beta = params
                beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
                angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
                normals = np.column_stack([np.cos(angles), np.sin(angles)])
                
                pts_centered = best_global_pts - np.array([cx, cy])
                projs = pts_centered @ normals.T
                max_proj = np.max(np.abs(projs))
                
                S = (max_proj + APOTHEM) / APOTHEM
                return S
            
            # Multiple restarts with slight perturbations
            for trial in range(5):
                if time.time() > _TIME_DEADLINE:
                    break
                if trial == 0:
                    init = [best_global_center[0], best_global_center[1], best_global_beta]
                else:
                    db = (np.random.random() - 0.5) * 0.05
                    dcx = (np.random.random() - 0.5) * 0.05
                    dcy = (np.random.random() - 0.5) * 0.05
                    init = [best_global_center[0] + dcx, best_global_center[1] + dcy, 
                           ((best_global_beta + db) % (math.pi/6) + math.pi/6) % (math.pi/6)]
                
                res = minimize(final_objective, init, method='Nelder-Mead',
                             options={'maxiter': 8000, 'xatol': 1e-15, 'fatol': 1e-15})
                if res.fun < best_global_S:
                    best_global_S = res.fun
                    best_global_center = (res.x[0], res.x[1])
                    best_global_beta = ((res.x[2] % (math.pi/6)) + math.pi/6) % (math.pi/6)
    except Exception:
        pass

    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)
    
    # Very small safety scale to ensure no overlap due to floating point
    SCALE = 1.0 + 1e-12
    pts_scaled = []
    for x, y in best_global_pts:
        sx = cx + (x - cx) * SCALE
        sy = cy + (y - cy) * SCALE
        pts_scaled.append((sx, sy))
    
    pts_scaled = np.array(pts_scaled)
    
    # Compute final side length
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
