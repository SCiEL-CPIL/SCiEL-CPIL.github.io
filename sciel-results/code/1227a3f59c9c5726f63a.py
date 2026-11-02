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

def _hexagon_vertices(cx, cy, angle, side=1.0):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with given rotation."""
    angles = angle + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def _hexagon_overlap(v1, v2, tol=1e-9):
    """Check if two hexagons (as vertex arrays) overlap using SAT."""
    # Get edges of hexagon 1
    edges1 = np.roll(v1, -1, axis=0) - v1
    # Get edges of hexagon 2
    edges2 = np.roll(v2, -1, axis=0) - v2
    
    all_edges = np.vstack([edges1, edges2])
    
    for edge in all_edges:
        # Normal to this edge
        nx, ny = -edge[1], edge[0]
        norm = math.hypot(nx, ny)
        if norm < 1e-12:
            continue
        nx, ny = nx / norm, ny / norm
        
        # Project both hexagons onto this normal
        proj1 = v1[:, 0] * nx + v1[:, 1] * ny
        proj2 = v2[:, 0] * nx + v2[:, 1] * ny
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        # Check for gap
        if max1 + tol < min2 or max2 + tol < min1:
            return False  # No overlap on this axis
    
    return True  # Overlap on all axes

def _force_directed_layout(n_hex, max_iter=500, dt=0.01, repulse_strength=5.0, attract_strength=0.5):
    """
    Force-directed layout to find a compact arrangement of n_hex unit hexagons.
    Returns array of (x, y) positions.
    """
    # Initialize in a rough hexagonal pattern
    positions = np.zeros((n_hex, 2))
    idx = 0
    for r in range(-3, 4):
        for q in range(-3, 4):
            if idx >= n_hex:
                break
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            positions[idx] = [x, y]
            idx += 1
        if idx >= n_hex:
            break
    
    # Randomize slightly to break symmetry
    positions += np.random.uniform(-0.1, 0.1, size=(n_hex, 2))
    
    for iteration in range(max_iter):
        forces = np.zeros((n_hex, 2))
        
        # Repulsive forces between all pairs
        for i in range(n_hex):
            for j in range(i + 1, n_hex):
                diff = positions[i] - positions[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-9:
                    diff = np.random.uniform(-0.01, 0.01, size=2)
                    dist = np.linalg.norm(diff)
                
                # Strong repulsion when close
                if dist < 2.5:
                    force_mag = repulse_strength / (dist ** 2)
                    forces[i] += force_mag * diff / dist
                    forces[j] -= force_mag * diff / dist
        
        # Attractive force toward center (to keep cluster compact)
        centroid = np.mean(positions, axis=0)
        for i in range(n_hex):
            forces[i] += -attract_strength * (positions[i] - centroid)
        
        # Update positions
        positions += dt * forces
        # Damping
        dt *= 0.995
    
    return positions

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

    # New candidates
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

    # Additional candidates
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

    # Additional promising configurations
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

    # More diverse configurations
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

    # Add force-directed layouts as additional candidates
    np.random.seed(123)
    fd_configs = []
    for trial in range(5):
        pos = _force_directed_layout(N, max_iter=300, dt=0.02, 
                                      repulse_strength=3.0 + trial * 0.5,
                                      attract_strength=0.3 + trial * 0.1)
        # Normalize: center at origin
        pos -= np.mean(pos, axis=0)
        fd_configs.append(pos)
    
    # Also try some hand-crafted optimized layouts
    # 3-3-3-2 arrangement
    handcrafted = [
        np.array([
            [0, 0], [math.sqrt(3), 0], [2*math.sqrt(3), 0],
            [math.sqrt(3)/2, 1.5], [3*math.sqrt(3)/2, 1.5], [5*math.sqrt(3)/2, 1.5],
            [math.sqrt(3), 3.0], [2*math.sqrt(3), 3.0], [3*math.sqrt(3), 3.0],
            [math.sqrt(3)/2, 4.5], [3*math.sqrt(3)/2, 4.5]
        ]) - np.array([math.sqrt(3), 2.25]),
        np.array([
            [0, 0], [math.sqrt(3), 0], [2*math.sqrt(3), 0], [3*math.sqrt(3), 0],
            [math.sqrt(3)/2, 1.5], [3*math.sqrt(3)/2, 1.5], [5*math.sqrt(3)/2, 1.5],
            [math.sqrt(3), 3.0], [2*math.sqrt(3), 3.0],
            [math.sqrt(3)/2, 4.5], [3*math.sqrt(3)/2, 4.5]
        ]) - np.array([1.5*math.sqrt(3), 2.25]),
    ]
    fd_configs.extend(handcrafted)
    
    # Store force-directed configs separately (they're already in cartesian)
    fd_candidates = fd_configs

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    # Combine axial candidates (converted to cartesian) with force-directed candidates
    all_cartesian_candidates = []
    for axial in candidates:
        all_cartesian_candidates.append(axial_to_cartesian(axial))
    for fd in fd_candidates:
        all_cartesian_candidates.append(fd)

    def calc_S_for_beta(pts, beta):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), None, None
        
        # Vectorized normal computation
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])  # (6, 2)
        
        # Build LP matrices efficiently
        n_pts = len(pts)
        A_ub_list = []
        b_ub_list = []
        
        for nx, ny in normals:
            # -nx*(px-cx) - ny*(py-cy) <= D  =>  nx*cx + ny*cy - D <= nx*px + ny*py
            # nx*(px-cx) + ny*(py-cy) <= D   =>  -nx*cx - ny*cy - D <= -(nx*px + ny*py)
            projs = pts[:, 0] * nx + pts[:, 1] * ny
            A_ub_list.append(np.column_stack([-nx * np.ones(n_pts), -ny * np.ones(n_pts), -np.ones(n_pts)]))
            b_ub_list.append(-(projs))
            A_ub_list.append(np.column_stack([nx * np.ones(n_pts), ny * np.ones(n_pts), -np.ones(n_pts)]))
            b_ub_list.append(projs)
        
        A_ub = np.vstack(A_ub_list)
        b_ub = np.concatenate(b_ub_list)
        bounds = [(None, None), (None, None), (0, None)]
        
        try:
            res = linprog(np.array([0.0, 0.0, 1.0]), A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
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
        normals = np.array([
            (math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0))
            for k in range(6)
        ])
        
        # Compute all projections efficiently using matrix operations
        # For each point and each normal, compute projection
        pts_centered = pts - np.array([cx, cy])
        # Shape: (n_points, 6)
        projs = pts_centered @ normals.T
        
        # We need max over all |proj|
        all_projs = np.concatenate([projs.ravel(), (-projs).ravel()])
        
        max_val = np.max(all_projs)
        if max_val > 100:
            return float('inf'), (cx, cy), float('inf')
        
        # Soft max via log-sum-exp for numerical stability
        shifted = (all_projs - max_val) / temperature
        # Clip to avoid overflow in exp
        shifted = np.clip(shifted, -50, 50)
        log_sum = np.log(np.sum(np.exp(shifted))) + max_val / temperature
        D = temperature * log_sum
        S = (D + APOTHEM) / APOTHEM
        return S, (cx, cy), D

    def optimize_shape(pts):
        best_S = float('inf')
        best_beta = 0.0
        best_center = None
        
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, None
        
        # Coarse grid search over beta - more points for better resolution
        for i in range(73):
            beta = i * (math.pi / 6) / 72
            S, center, D = calc_S_for_beta(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = center
        
        # Refine beta with bounded scalar optimization
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                res = minimize_scalar(lambda b: calc_S_for_beta(pts, b)[0], bounds=(0, math.pi / 6), method='bounded',
                                     options={'maxiter': 500, 'xatol': 1e-14})
                if res.fun < best_S:
                    best_S = res.fun
                    best_beta = res.x
                    _, best_center, _ = calc_S_for_beta(pts, best_beta)
        except Exception:
            pass
        
        # Smooth surrogate optimization with adaptive temperature - more starting points
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                temperatures = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
                start_betas = [best_beta, best_beta + 0.08, best_beta - 0.08, 0.0, math.pi/12, math.pi/6, best_beta + 0.04, best_beta - 0.04]
                
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
                                     options={'maxiter': 800, 'xatol': 1e-11, 'fatol': 1e-11})
                        
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
        
        # Final high-precision polish using exact objective - vectorized
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                def objective(params):
                    cx, cy, beta = params
                    beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
                    angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
                    normals = np.column_stack([np.cos(angles), np.sin(angles)])
                    
                    # Vectorized projection computation
                    centered = pts - np.array([cx, cy])
                    projs = np.abs(centered @ normals.T)  # (n_pts, 6)
                    max_proj = np.max(projs)
                    
                    S = (max_proj + APOTHEM) / APOTHEM
                    return S
                
                polish_starts = [
                    [best_center[0], best_center[1], best_beta],
                    [best_center[0], best_center[1], best_beta + 0.03],
                    [best_center[0], best_center[1], best_beta - 0.03],
                    [best_center[0] + 0.03, best_center[1], best_beta],
                    [best_center[0], best_center[1] + 0.03, best_beta],
                    [best_center[0] - 0.03, best_center[1], best_beta],
                    [best_center[0], best_center[1] - 0.03, best_beta],
                    [best_center[0] + 0.01, best_center[1] + 0.01, best_beta],
                    [best_center[0] - 0.01, best_center[1] - 0.01, best_beta],
                ]
                
                for init in polish_starts:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    res = minimize(objective, init, method='Nelder-Mead',
                                 options={'maxiter': 8000, 'xatol': 1e-14, 'fatol': 1e-14})
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

    # Process all cartesian candidates
    # Sort by bounding box area as heuristic
    def bbox_heuristic(pts):
        extent_x = np.max(pts[:, 0]) - np.min(pts[:, 0])
        extent_y = np.max(pts[:, 1]) - np.min(pts[:, 1])
        return extent_x * extent_y
    
    sorted_cartesian = sorted(all_cartesian_candidates, key=bbox_heuristic)
    
    for pts in sorted_cartesian:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        S, beta, center = optimize_shape(pts)
        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts
            best_global_beta = beta
            best_global_center = center

    if best_global_center is None or best_global_S == float('inf'):
        best_global_pts = all_cartesian_candidates[0]
        best_global_center = (float(np.mean(best_global_pts[:, 0])), float(np.mean(best_global_pts[:, 1])))
        best_global_beta = 0.0
        max_proj = 0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = best_global_pts[:, 0] * nx + best_global_pts[:, 1] * ny
            max_proj = max(max_proj, np.max(projs), -np.min(projs))
        best_global_S = (max_proj + APOTHEM) / APOTHEM
    
    # Second pass: try to improve the best configuration with more aggressive optimization
    if best_global_S < float('inf') and time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
        # Try small perturbations of the best configuration
        def perturbed_objective(params, base_pts, base_beta, base_center):
            cx, cy, beta = params
            beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
            angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
            normals = np.column_stack([np.cos(angles), np.sin(angles)])
            
            centered = base_pts - np.array([cx, cy])
            projs = np.abs(centered @ normals.T)
            max_proj = np.max(projs)
            
            S = (max_proj + APOTHEM) / APOTHEM
            return S
        
        # Try multiple random-ish perturbations
        np.random.seed(42)
        for trial in range(5):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            
            # Random small perturbation
            dx = np.random.uniform(-0.1, 0.1)
            dy = np.random.uniform(-0.1, 0.1)
            dbeta = np.random.uniform(-0.1, 0.1)
            
            init = [best_global_center[0] + dx, best_global_center[1] + dy, 
                   ((best_global_beta + dbeta) % (math.pi/6) + math.pi/6) % (math.pi/6)]
            
            res = minimize(perturbed_objective, init, args=(best_global_pts, best_global_beta, best_global_center),
                         method='Nelder-Mead', options={'maxiter': 3000, 'xatol': 1e-13, 'fatol': 1e-13})
            
            if res.fun < best_global_S:
                best_global_S = res.fun
                best_global_center = (res.x[0], res.x[1])
                best_global_beta = ((res.x[2] % (math.pi/6)) + math.pi/6) % (math.pi/6)

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
