import math
import time
import numpy as np
from scipy.optimize import linprog, minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
R = 1.0
START_TIME = time.time()
TIME_LIMIT = 148.0


def _time_ok(margin=5.0):
    return time.time() - START_TIME < TIME_LIMIT - margin


def _axial_to_cartesian(axial_list, scale=1.0):
    pts = []
    for q, r in axial_list:
        x = math.sqrt(3) * (q + r / 2.0) * scale
        y = 1.5 * r * scale
        pts.append((x, y))
    return np.array(pts, dtype=float)


def _hexagon_vertices(center, angle, side=1.0):
    """Return 6 vertices of a regular hexagon centered at center, rotated by angle, side length side."""
    vs = []
    for k in range(6):
        a = angle + k * math.pi / 3.0
        vs.append((center[0] + side * math.cos(a), center[1] + side * math.sin(a)))
    return np.array(vs)


def _outer_side_given_config(pts, th, beta):
    """
    Given inner hexagon centers pts (n,2), angles th (n,), and outer orientation beta,
    solve LP to find outer center and min side S.
    """
    n = len(pts)
    normals = []
    for k in range(6):
        ang = beta + math.pi / 6.0 + k * math.pi / 3.0
        normals.append(np.array([math.cos(ang), math.sin(ang)]))
    
    A_ub = []
    b_ub = []
    
    for nk in normals:
        phi_n = math.atan2(nk[1], nk[0])
        for i in range(n):
            pi = pts[i]
            pi_n = float(pi @ nk)
            # h_i(nk) = R * max_j cos(th_i + j*pi/3 - phi_n)
            angles = th[i] + np.arange(6) * (math.pi / 3.0) - phi_n
            h_i = R * float(np.max(np.cos(angles)))
            # Constraint: (p_i - c)·n_k + h_i <= S * APOTHEM
            # => -c_x*nk_x - c_y*nk_y - S*APOTHEM <= -(pi_n + h_i)
            A_ub.append([-nk[0], -nk[1], -APOTHEM])
            b_ub.append(-(pi_n + h_i))
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    bounds = [(None, None), (None, None), (0.0, None)]
    c_obj = np.array([0.0, 0.0, 1.0])
    
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.status == 0:
            S = res.x[2]
            center = (res.x[0], res.x[1])
            return S, center
    except Exception:
        pass
    return float('inf'), None


def _best_orientation_given_config(pts, th):
    """Find best outer orientation beta for a given inner configuration."""
    best_S = float('inf')
    best_beta = 0.0
    best_c = None
    
    # Coarse sweep over beta in [0, pi/3) due to 6-fold symmetry of outer hex
    for i in range(61):
        beta = i * (math.pi / 3.0) / 60
        S, c = _outer_side_given_config(pts, th, beta)
        if S < best_S:
            best_S, best_beta, best_c = S, beta, c
    
    # Refine with local optimization
    try:
        def obj(b):
            S, _ = _outer_side_given_config(pts, th, b)
            return S if np.isfinite(S) else 1e10
        
        res = minimize_scalar(obj, bounds=(max(0, best_beta - 0.05), min(math.pi/3, best_beta + 0.05)),
                              method='bounded', options={'maxiter': 100, 'xatol': 1e-10})
        if res.fun < best_S:
            best_S = res.fun
            best_beta = res.x
            _, best_c = _outer_side_given_config(pts, th, best_beta)
    except Exception:
        pass
    
    return best_S, best_beta, best_c


def _gap_sat(p1, th1, p2, th2):
    """Compute minimum gap between two hexagons using SAT. Positive = separated, negative = overlap."""
    v1 = _hexagon_vertices(p1, th1)
    v2 = _hexagon_vertices(p2, th2)
    
    axes = []
    for v in [v1, v2]:
        for k in range(6):
            edge = v[(k+1) % 6] - v[k]
            nx = -edge[1]
            ny = edge[0]
            norm = math.sqrt(nx*nx + ny*ny)
            if norm > 1e-12:
                axes.append((nx/norm, ny/norm))
    
    min_gap = float('inf')
    for nx, ny in axes:
        proj1 = np.array([v[0]*nx + v[1]*ny for v in v1])
        proj2 = np.array([v[0]*nx + v[1]*ny for v in v2])
        if np.max(proj1) <= np.min(proj2):
            gap = np.min(proj2) - np.max(proj1)
        elif np.max(proj2) <= np.min(proj1):
            gap = np.min(proj1) - np.max(proj2)
        else:
            gap = min(np.max(proj1), np.max(proj2)) - max(np.min(proj1), np.min(proj2))
        min_gap = min(min_gap, gap)
    
    return float(min_gap)


def _check_nonoverlap(pts, th, tol=-1e-8):
    """Check if all pairs of inner hexagons are non-overlapping. Return True if valid."""
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            g = _gap_sat(pts[i], th[i], pts[j], th[j])
            if g < tol:
                return False
    return True


def _smooth_obj(x, pts0, th0, beta, temp=0.1):
    """
    Smooth objective: minimize outer side length with soft non-overlap penalty.
    x = [dx1, dy1, dth1, dx2, dy2, dth2, ...]
    """
    n = len(pts0)
    pts = pts0.copy()
    th = th0.copy()
    for i in range(n):
        pts[i] += np.array([x[3*i], x[3*i+1]])
        th[i] += x[3*i+2]
    
    # Compute outer side
    S, c = _outer_side_given_config(pts, th, beta)
    if not np.isfinite(S):
        return 1e10
    
    # Non-overlap penalty (soft)
    penalty = 0.0
    for i in range(n):
        for j in range(i+1, n):
            g = _gap_sat(pts[i], th[i], pts[j], th[j])
            if g < 0:
                penalty += -g**2 / temp
            else:
                penalty += 0.0
    
    return S + penalty


def _optimize_with_smooth(pts0, th0, beta, max_iter=300):
    """
    Use smooth surrogate to optimize positions and angles.
    """
    n = len(pts0)
    x0 = np.zeros(3*n)
    
    best_S = float('inf')
    best_x = x0.copy()
    best_pts = pts0.copy()
    best_th = th0.copy()
    
    # Multi-start
    for start in range(5):
        if not _time_ok(3.0):
            break
        
        if start == 0:
            x_start = x0.copy()
        else:
            x_start = np.random.randn(3*n) * 0.05
        
        try:
            res = minimize(_smooth_obj, x_start, args=(pts0, th0, beta, 0.05),
                          method='L-BFGS-B', options={'maxiter': max_iter, 'ftol': 1e-10})
            
            if res.fun < best_S:
                # Verify validity
                pts = pts0.copy()
                th = th0.copy()
                for i in range(n):
                    pts[i] += np.array([res.x[3*i], res.x[3*i+1]])
                    th[i] += res.x[3*i+2]
                
                if _check_nonoverlap(pts, th, tol=-1e-7):
                    S, c = _outer_side_given_config(pts, th, beta)
                    if np.isfinite(S) and S < best_S:
                        best_S = S
                        best_x = res.x.copy()
                        best_pts = pts.copy()
                        best_th = th.copy()
        except Exception:
            pass
    
    return best_pts, best_th, best_S


def construct():
    if not _time_ok(10.0):
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Candidate lattice configurations
    candidates = [
        # Flower (1+6+4)
        [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
         (1, 1), (-1, 2), (-2, 1), (2, -1)],
        # Rows 4-3-4
        [(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (2, 1),
         (0, 2), (1, 2), (2, 2), (3, 2)],
        # 3-5-3
        [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (3, 1),
         (4, 1), (0, 2), (1, 2), (2, 2)],
        # 5-3-3
        [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (0, 1), (1, 1), (2, 1),
         (0, 2), (1, 2), (2, 2)],
        # 4-4-3
        [(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (2, 1), (3, 1),
         (1, 2), (2, 2), (3, 2)],
        # 3-4-4
        [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (3, 1),
         (0, 2), (1, 2), (2, 2), (3, 2), (4, 2)],
        # 2-3-3-3
        [(0, 0), (1, 0), (0, 1), (1, 1), (2, 1), (0, 2), (1, 2),
         (2, 2), (0, 3), (1, 3), (2, 3)],
        # 3-3-3-2
        [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (0, 2),
         (1, 2), (2, 2), (0, 3), (1, 3)],
        # 4-3-4 variant
        [(0, 0), (1, 0), (2, 0), (3, 0), (1, 1), (2, 1), (3, 1),
         (0, 2), (1, 2), (2, 2), (3, 2)],
        # 5-4-2
        [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (0, 1), (1, 1),
         (2, 1), (3, 1), (1, 2), (2, 2)],
        # 2-4-5
        [(0, 0), (1, 0), (0, 1), (1, 1), (2, 1), (3, 1), (0, 2),
         (1, 2), (2, 2), (3, 2), (4, 2)],
        # 3-4-4 alt
        [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (3, 1),
         (1, 2), (2, 2), (3, 2), (4, 2), (5, 2)],
    ]

    best_S = float('inf')
    best_result = None

    for idx, axial in enumerate(candidates):
        if not _time_ok(5.0):
            break
        
        pts0 = _axial_to_cartesian(axial, scale=1.0 + 1e-6)
        centroid = pts0.mean(axis=0)
        pts0 = pts0 - centroid
        th0 = np.zeros(len(pts0))
        
        # Try multiple initial angle configurations
        angle_inits = [
            np.zeros(len(pts0)),
            np.full(len(pts0), math.pi/6),
            np.full(len(pts0), math.pi/12),
            np.linspace(0, math.pi/6, len(pts0)),
        ]
        
        for ai, th_init in enumerate(angle_inits):
            if not _time_ok(3.0):
                break
            
            # First find best orientation for this config
            S_or, beta_or, c_or = _best_orientation_given_config(pts0, th_init)
            
            # Optimize with smooth surrogate
            try:
                pts_opt, th_opt, S_opt = _optimize_with_smooth(pts0, th_init, beta_or, max_iter=200)
            except Exception:
                pts_opt, th_opt, S_opt = pts0, th_init, S_or
            
            # Check validity and update best
            if _check_nonoverlap(pts_opt, th_opt, tol=-1e-7):
                S_final, c_final = _outer_side_given_config(pts_opt, th_opt, beta_or)
                if np.isfinite(S_final) and S_final < best_S:
                    best_S = S_final
                    best_result = (pts_opt, th_opt, c_final, beta_or)
            
            # Also try the lattice configuration directly
            if S_or < best_S and c_or is not None:
                best_S = S_or
                best_result = (pts0, th_init, c_or, beta_or)

    if best_result is None or not _time_ok(2.0):
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    p_use, th_use, c_out, best_beta = best_result
    inner = []
    for k in range(len(p_use)):
        inner.append([float(p_use[k, 0]), float(p_use[k, 1]),
                      float(math.degrees(th_use[k]))])
    center = [float(c_out[0]), float(c_out[1])]
    side = float(best_S)
    rotation = float(math.degrees(best_beta))
    return inner, center, side, rotation
