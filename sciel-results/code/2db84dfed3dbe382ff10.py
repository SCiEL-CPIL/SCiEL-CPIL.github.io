import math
import numpy as np
import time
from scipy.optimize import linprog, minimize, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0
EPS = 1e-9

def _hexagon_vertices(cx, cy, angle_deg, side=1.0):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with given rotation."""
    a = math.radians(angle_deg)
    verts = []
    for k in range(6):
        theta = a + k * math.pi / 3.0
        verts.append((cx + side * math.cos(theta), cy + side * math.sin(theta)))
    return verts

def _hexagon_overlap(v1, v2, tol=1e-9):
    """Check if two hexagons (lists of 6 vertices) overlap using separating axis theorem."""
    axes = []
    for v in [v1, v2]:
        for i in range(6):
            p1 = v[i]
            p2 = v[(i + 1) % 6]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            length = math.sqrt(dx*dx + dy*dy)
            if length > 1e-12:
                axes.append((-dy/length, dx/length))
    for ax in axes:
        min1, max1 = float('inf'), float('-inf')
        min2, max2 = float('inf'), float('-inf')
        for p in v1:
            proj = p[0]*ax[0] + p[1]*ax[1]
            min1 = min(min1, proj)
            max1 = max(max1, proj)
        for p in v2:
            proj = p[0]*ax[0] + p[1]*ax[1]
            min2 = min(min2, proj)
            max2 = max(max2, proj)
        if max1 < min2 - tol or max2 < min1 - tol:
            return False
    return True

def _force_directed_layout(pts, max_iter=2000, dt=0.01, deadline=None):
    """Force-directed relaxation to find non-overlapping configuration with tighter bounding hexagon.
    Returns relaxed points if they fit in a hexagon no larger than the input's bounding hexagon."""
    if deadline is not None and time.time() > deadline:
        return pts
    
    n = len(pts)
    pts_f = pts.astype(float).copy()
    best_pts = pts_f.copy()
    best_S = float('inf')
    
    # Compute initial S for reference
    def compute_S(pts_arr):
        cx = np.mean(pts_arr[:, 0])
        cy = np.mean(pts_arr[:, 1])
        centered = pts_arr - np.array([cx, cy])
        max_proj = 0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = centered[:, 0] * nx + centered[:, 1] * ny
            max_proj = max(max_proj, np.max(projs), -np.min(projs))
        return (max_proj + APOTHEM) / APOTHEM, cx, cy
    
    init_S, _, _ = compute_S(pts_f)
    best_S = init_S
    
    # Precompute hexagon vertices for overlap check
    def get_hex_verts(pt, angle=0.0):
        a = math.radians(angle)
        return [(pt[0] + math.cos(a + k*math.pi/3), pt[1] + math.sin(a + k*math.pi/3)) for k in range(6)]
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        if it % 50 == 0 and it > 0:
            S_now, _, _ = compute_S(pts_f)
            if S_now < best_S:
                best_S = S_now
                best_pts = pts_f.copy()
        
        # Check for overlaps and apply repulsive forces
        forces = np.zeros_like(pts_f)
        overlap_found = False
        for i in range(n):
            for j in range(i+1, n):
                dist = np.linalg.norm(pts_f[i] - pts_f[j])
                # Minimum distance for non-overlapping unit hexagons is sqrt(3)
                min_dist = math.sqrt(3)
                if dist < min_dist - 1e-9:
                    overlap_found = True
                    if dist < 1e-12:
                        # Random perturbation
                        angle = np.random.uniform(0, 2*math.pi)
                        forces[i] += 0.1 * np.array([math.cos(angle), math.sin(angle)])
                        forces[j] -= 0.1 * np.array([math.cos(angle), math.sin(angle)])
                    else:
                        direction = (pts_f[i] - pts_f[j]) / dist
                        repel = (min_dist - dist) * 2.0
                        forces[i] += direction * repel
                        forces[j] -= direction * repel
        
        # Apply attractive force toward center of mass to keep compact
        com = np.mean(pts_f, axis=0)
        for i in range(n):
            forces[i] += (com - pts_f[i]) * 0.01
        
        # Update positions
        pts_f += dt * forces
        dt *= 0.999  # Slightly decay step size
    
    S_final, _, _ = compute_S(pts_f)
    if S_final < best_S:
        best_pts = pts_f.copy()
    
    return best_pts

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # --- Candidate point sets (axial coordinates) ---
    candidates = []

    # Hexagonal cluster: center + 6 neighbors + 4 more
    axial_1 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ]
    candidates.append(axial_1)

    # Star: center + 6 neighbors + 4 opposite
    axial_2 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_2)

    # 3x3 block + 2 extras
    axial_3 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1)
    ]
    candidates.append(axial_3)

    # 4x3 block + 1 extra
    axial_4 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2)
    ]
    candidates.append(axial_4)

    # 3x4 block + 1 extra
    axial_5 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_5)

    # 3x3 + 2
    axial_6 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3)
    ]
    candidates.append(axial_6)

    # Cross-like
    axial_7 = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_7)

    # 5+6 row
    axial_8 = [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)
    ]
    candidates.append(axial_8)

    # Center + 6 + 4 asymmetric
    axial_9 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (-2, 2), (-1, 3), (1, -2)
    ]
    candidates.append(axial_9)

    # 3x3 + 2 top
    axial_10 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3)
    ]
    candidates.append(axial_10)

    # 2+4+4+1
    axial_11 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (1, 3)
    ]
    candidates.append(axial_11)

    # 4+2+2+2+1
    axial_12 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3),
        (0, 4)
    ]
    candidates.append(axial_12)

    # Center + 6 + 4 (variant)
    axial_13 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (1, 2)
    ]
    candidates.append(axial_13)

    # Center + 6 + 4 (variant)
    axial_14 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (1, 1), (-1, -1)
    ]
    candidates.append(axial_14)

    # 3x3 with extras
    axial_15 = [
        (0, 0), (1, 0), (2, 0),
        (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1), (0, -1)
    ]
    candidates.append(axial_15)

    # Staircase
    axial_16 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3), (3, 2)
    ]
    candidates.append(axial_16)

    # Center + 6 + 4 (variant)
    axial_17 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (1, 2), (-1, 2), (0, 3)
    ]
    candidates.append(axial_17)

    # 4+3+3
    axial_18 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2), (3, 2)
    ]
    candidates.append(axial_18)

    # 2+3+2+3+1
    axial_19 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3), (3, 3)
    ]
    candidates.append(axial_19)

    # Center + 6 + 4 (variant)
    axial_20 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (1, 1), (-1, 1)
    ]
    candidates.append(axial_20)

    # 4+4+3
    axial_21 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (1, 2), (2, 2), (3, 2)
    ]
    candidates.append(axial_21)

    # 3+4+3+1
    axial_22 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3)
    ]
    candidates.append(axial_22)

    # Center + 6 + 4 (variant)
    axial_23 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_23)

    # 2+2+3+2+2
    axial_24 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3),
        (0, 4)
    ]
    candidates.append(axial_24)

    # 3+2+3+3
    axial_25 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_25)

    # Compact: 2+3+3+2+1
    axial_36 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3),
        (1, 4)
    ]
    candidates.append(axial_36)

    # Compact: 3+3+3+1+1
    axial_37 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3),
        (0, 3)
    ]
    candidates.append(axial_37)

    # Compact: 4+2+2+2+1 (alternate)
    axial_38 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (1, 1), (2, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3),
        (0, 4)
    ]
    candidates.append(axial_38)

    # Compact: 2+4+3+1+1
    axial_39 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3),
        (2, 3)
    ]
    candidates.append(axial_39)

    # Compact: 3+4+2+1+1
    axial_40 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (1, 2), (2, 2),
        (0, 3),
        (3, 3)
    ]
    candidates.append(axial_40)

    # Compact: 2+2+4+2+1
    axial_41 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (0, 3), (1, 3),
        (2, 3)
    ]
    candidates.append(axial_41)

    # Compact: 3+3+2+2+1
    axial_42 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3),
        (2, 3)
    ]
    candidates.append(axial_42)

    # Compact: 4+3+2+1+1
    axial_43 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3),
        (3, 3)
    ]
    candidates.append(axial_43)

    # Compact: 2+3+3+2+1 (alternate)
    axial_44 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3),
        (0, 4)
    ]
    candidates.append(axial_44)

    # 4+3+4+2
    axial_26 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (1, 3), (2, 3)
    ]
    candidates.append(axial_26)

    # Center + 6 + 4 (variant)
    axial_27 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (1, -2)
    ]
    candidates.append(axial_27)

    # 2+3+2+3+3
    axial_28 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3),
        (1, 4), (2, 4)
    ]
    candidates.append(axial_28)

    # Additional diverse candidates
    axial_29 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_29)

    axial_30 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (1, 2), (-1, 2), (0, 3)
    ]
    candidates.append(axial_30)

    axial_31 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3)
    ]
    candidates.append(axial_31)

    axial_32 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3), (3, 2)
    ]
    candidates.append(axial_32)

    axial_33 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3), (3, 2)
    ]
    candidates.append(axial_33)

    axial_34 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3), (3, 3)
    ]
    candidates.append(axial_34)

    axial_35 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_35)

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def calc_S_for_beta(pts, beta):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), None, None
        
        # Vectorized: compute normals and projections
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])  # (6, 2)
        
        n_pts = len(pts)
        A_ub_list = []
        b_ub_list = []
        for k in range(6):
            nx, ny = normals[k]
            projs = pts[:, 0] * nx + pts[:, 1] * ny
            A_ub_list.append(np.column_stack([-np.full(n_pts, nx), -np.full(n_pts, ny), -np.ones(n_pts)]))
            b_ub_list.append(-projs)
            A_ub_list.append(np.column_stack([np.full(n_pts, nx), np.full(n_pts, ny), -np.ones(n_pts)]))
            b_ub_list.append(projs)
        
        A_ub = np.vstack(A_ub_list)
        b_ub = np.concatenate(b_ub_list)
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

    def fast_exact_S(pts, cx, cy, beta):
        """Vectorized exact objective: minimize max over all (point, normal) of |proj|."""
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])  # (6, 2)
        centered = pts - np.array([cx, cy])  # (n, 2)
        projs = centered @ normals.T  # (n, 6)
        max_proj = np.max(np.abs(projs))
        return (max_proj + APOTHEM) / APOTHEM

    def smooth_calc_S(pts, beta, cx, cy, temperature=0.1):
        normals = np.array([
            (math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0))
            for k in range(6)
        ])
        
        pts_centered = pts - np.array([cx, cy])
        projs = pts_centered @ normals.T
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

    def optimize_shape(pts, current_best_S=float('inf')):
        best_S = float('inf')
        best_beta = 0.0
        best_center = None
        
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, None
        
        # Coarse grid search over beta using LP
        for i in range(37):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                return best_S, best_beta, best_center
            beta = i * (math.pi / 6) / 36
            S, center, D = calc_S_for_beta(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = center
        
        # Early exit if this candidate clearly can't beat global best
        if best_S >= current_best_S * 1.001 and current_best_S < float('inf'):
            return best_S, best_beta, best_center
        
        # Refine beta with bounded scalar optimization
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                res = minimize_scalar(lambda b: calc_S_for_beta(pts, b)[0], bounds=(0, math.pi / 6), method='bounded',
                                     options={'maxiter': 300, 'xatol': 1e-12})
                if res.fun < best_S:
                    best_S = res.fun
                    best_beta = res.x
                    _, best_center, _ = calc_S_for_beta(pts, best_beta)
        except Exception:
            pass
        
        # Smooth surrogate optimization with adaptive temperature
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                # Adaptive temperature schedule: start higher for global exploration, decrease for local refinement
                temperatures = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]
                start_betas = [best_beta, best_beta + 0.05, best_beta - 0.05, 0.0, math.pi/12, math.pi/6, 
                              best_beta + 0.1, best_beta - 0.1, math.pi/4, math.pi/8]
                
                for temp in temperatures:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    for sb in start_betas:
                        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                            break
                        
                        def smooth_obj(params, t=temp):
                            cx, cy, beta = params
                            beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
                            S, _, _ = smooth_calc_S(pts, beta, cx, cy, temperature=t)
                            return S
                        
                        init = [best_center[0], best_center[1], sb]
                        res = minimize(smooth_obj, init, method='Nelder-Mead',
                                     options={'maxiter': 800, 'xatol': 1e-12, 'fatol': 1e-12})
                        
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
        
        # Final high-precision polish using vectorized exact objective
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                def objective(params):
                    cx, cy, beta = params
                    beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
                    return fast_exact_S(pts, cx, cy, beta)
                
                bcx, bcy = best_center
                bb = best_beta
                # More diverse starting points for polish
                polish_starts = [
                    [bcx, bcy, bb],
                    [bcx, bcy, bb + 0.01],
                    [bcx, bcy, bb - 0.01],
                    [bcx, bcy, bb + 0.02],
                    [bcx, bcy, bb - 0.02],
                    [bcx, bcy, bb + 0.05],
                    [bcx, bcy, bb - 0.05],
                    [bcx, bcy, bb + 0.1],
                    [bcx, bcy, bb - 0.1],
                    [bcx + 0.01, bcy, bb],
                    [bcx - 0.01, bcy, bb],
                    [bcx, bcy + 0.01, bb],
                    [bcx, bcy - 0.01, bb],
                    [bcx + 0.02, bcy, bb],
                    [bcx - 0.02, bcy, bb],
                    [bcx, bcy + 0.02, bb],
                    [bcx, bcy - 0.02, bb],
                    [bcx + 0.01, bcy + 0.01, bb],
                    [bcx - 0.01, bcy - 0.01, bb],
                    [bcx + 0.01, bcy - 0.01, bb],
                    [bcx - 0.01, bcy + 0.01, bb],
                    [bcx + 0.02, bcy + 0.02, bb],
                    [bcx - 0.02, bcy - 0.02, bb],
                    [bcx + 0.02, bcy - 0.02, bb],
                    [bcx - 0.02, bcy + 0.02, bb],
                ]
                
                for init in polish_starts:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    res = minimize(objective, init, method='Nelder-Mead',
                                 options={'maxiter': 15000, 'xatol': 1e-15, 'fatol': 1e-15})
                    if res.fun < best_S:
                        best_S = res.fun
                        best_center = (res.x[0], res.x[1])
                        best_beta = ((res.x[2] % (math.pi/6)) + math.pi/6) % (math.pi/6)
        except Exception:
            pass
        
        # Force-directed relaxation to find better point configurations
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                deadline = TIME_LIMIT - SAFETY_MARGIN
                relaxed_pts = _force_directed_layout(pts, max_iter=3000, dt=0.05, deadline=deadline)
                if relaxed_pts is not None and not np.array_equal(relaxed_pts, pts):
                    # Re-optimize the relaxed configuration
                    S_rel, beta_rel, center_rel = optimize_shape(relaxed_pts, current_best_S=best_S)
                    if S_rel < best_S:
                        best_S = S_rel
                        best_beta = beta_rel
                        best_center = center_rel
        except Exception:
            pass
            
        return best_S, best_beta, best_center

    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

    # Process candidates in order of their initial bounding hexagon size
    candidate_scores = []
    for idx, axial in enumerate(candidates):
        pts = axial_to_cartesian(axial)
        # Quick estimate: bounding box diagonal
        cx_est = float(np.mean(pts[:, 0]))
        cy_est = float(np.mean(pts[:, 1]))
        centered = pts - np.array([cx_est, cy_est])
        max_proj = 0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = centered[:, 0] * nx + centered[:, 1] * ny
            max_proj = max(max_proj, np.max(projs), -np.min(projs))
        est_S = (max_proj + APOTHEM) / APOTHEM
        candidate_scores.append((est_S, idx))
    
    # Sort by estimated size (smallest first) to find good solutions early
    candidate_scores.sort()
    
    for est_S, idx in candidate_scores:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        axial = candidates[idx]
        pts = axial_to_cartesian(axial)
        S, beta, center = optimize_shape(pts, current_best_S=best_global_S)
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

    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)
    
    # Use a very small safety margin to ensure no overlap
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
