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

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Generate all 11-ominoes (polyhexes of size 11)
    def generate_polyhexes(n_cells):
        # Use a set of frozensets of axial coordinates to track unique shapes
        # We generate by adding cells to a base shape
        # To ensure uniqueness up to rotation/reflection, we canonicalize each shape
        
        def canonicalize(cells):
            # Translate to min x, min y
            min_x = min(c[0] for c in cells)
            min_y = min(c[1] for c in cells)
            translated = [(c[0] - min_x, c[1] - min_y) for c in cells]
            
            # Generate all 6 rotations and 6 reflections (total 12 symmetries)
            # Axial coords: (q, r)
            # Rotation 60 deg: (q, r) -> (-r, q+r)
            # Reflection: (q, r) -> (r, q) ? No, reflection across y-axis in axial is tricky.
            # Easier: convert to cartesian, rotate/reflect, snap back to axial?
            # Or use standard axial transformations.
            # Symmetries of hex grid:
            # 1. Identity: (q, r)
            # 2. Rot 60: (-r, q+r)
            # 3. Rot 120: (-q-r, q)
            # 4. Rot 180: (-q, -r)
            # 5. Rot 240: (q+r, -q)
            # 6. Rot 300: (r, -q-r)
            # Reflections (across axes):
            # Refl 1 (x-axis?): (q, -q-r) -- wait, let's just use cartesian for canonicalization to be safe and simple, 
            # though it's slower. For n=11, it's fine.
            
            def to_cart(q, r):
                x = math.sqrt(3) * (q + r / 2.0)
                y = 1.5 * r
                return (x, y)
                
            def from_cart(x, y):
                r = y / 1.5
                q = x / math.sqrt(3) - r / 2.0
                return (round(q), round(r))

            def transform_cell(q, r, angle_deg, reflect=False):
                x, y = to_cart(q, r)
                if reflect:
                    y = -y
                ang = math.radians(angle_deg)
                ca, sa = math.cos(ang), math.sin(ang)
                nx = x * ca - y * sa
                ny = x * sa + y * ca
                return from_cart(nx, ny)

            variants = set()
            for k in range(6):
                angle = k * 60
                for refl in [False, True]:
                    t_cells = []
                    for q, r in translated:
                        t_cells.append(transform_cell(q, r, angle, refl))
                    t_cells = set(t_cells)
                    # Re-normalize translation for this variant
                    min_x = min(c[0] for c in t_cells)
                    min_y = min(c[1] for c in t_cells)
                    norm_cells = frozenset((c[0] - min_x, c[1] - min_y) for c in t_cells)
                    variants.add(norm_cells)
            
            return min(variants)

        # BFS
        all_shapes = set()
        # Start with 1 cell
        current_level = [frozenset([(0, 0)])]
        
        for step in range(n_cells - 1):
            next_level = set()
            for shape in current_level:
                # Try adding a cell to each empty neighbor
                neighbors = set()
                for q, r in shape:
                    for dq, dr in [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]:
                        nq, nr = q + dq, r + dr
                        if (nq, nr) not in shape:
                            neighbors.add((nq, nr))
                
                for nq, nr in neighbors:
                    new_shape = frozenset(list(shape) + [(nq, nr)])
                    # Check if this shape is new (canonical)
                    # Optimization: Only add if it's not already in all_shapes or next_level
                    # Canonicalization is expensive, so we do it here
                    canon = canonicalize(new_shape)
                    if canon not in all_shapes:
                        all_shapes.add(canon)
                        next_level.add(canon)
            
            current_level = next_level
            if not current_level:
                break
                
        return list(all_shapes)

    # We need 11-ominoes. Generating all might take a bit of time but is feasible.
    # However, running full optimization on ~2000 shapes is too slow.
    # Strategy: 
    # 1. Generate all 11-ominoes.
    # 2. For each, do a very fast "bounding box" estimate or single LP to get a rough score.
    # 3. Keep top K candidates.
    # 4. Run full optimization on top K.

    # To save time, we can also just use the manual candidates + a subset of BFS if BFS is slow.
    # Let's try to generate them.
    
    try:
        all_11_ominoes = generate_polyhexes(11)
    except Exception:
        all_11_ominoes = []

    if not all_11_ominoes:
        # Fallback to manual candidates if generation fails or is empty
        candidates = [
            [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1), (1, 1), (-1, 2), (-2, 1), (2, -1)],
            [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1), (2, 0), (-2, 0), (0, 2), (0, -2)],
            [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (0, 2), (1, 2), (2, 2), (-1, 1), (3, 1)]
        ]
    else:
        candidates = all_11_ominoes

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
        
        normals = []
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))
        
        c_obj = np.array([0.0, 0.0, 1.0])
        A_ub = []
        b_ub = []
        
        for nx, ny in normals:
            for px, py in pts:
                A_ub.append([-nx, -ny, -1.0])
                b_ub.append(-(px * nx + py * ny))
                A_ub.append([nx, ny, -1.0])
                b_ub.append(px * nx + py * ny)
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
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

    def optimize_shape(pts):
        best_S = float('inf')
        best_beta = 0.0
        best_center = None
        
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, None
        
        # Coarse grid search over beta
        for i in range(37):
            beta = i * (math.pi / 6) / 36
            S, center, D = calc_S_for_beta(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = center
        
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
                temperatures = [0.3, 0.1, 0.05, 0.02, 0.01]
                start_betas = [best_beta, best_beta + 0.05, best_beta - 0.05, 0.0, math.pi/12, math.pi/6]
                
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
                                     options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10})
                        
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
        
        # Final high-precision polish using exact objective
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                def objective(params):
                    cx, cy, beta = params
                    beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
                    normals = []
                    for k in range(6):
                        ang = beta + math.pi / 6.0 + k * math.pi / 3.0
                        normals.append((math.cos(ang), math.sin(ang)))
                    
                    max_proj = 0.0
                    for nx, ny in normals:
                        for px, py in pts:
                            proj = abs((px - cx) * nx + (py - cy) * ny)
                            if proj > max_proj:
                                max_proj = proj
                    
                    S = (max_proj + APOTHEM) / APOTHEM
                    return S
                
                polish_starts = [
                    [best_center[0], best_center[1], best_beta],
                    [best_center[0], best_center[1], best_beta + 0.02],
                    [best_center[0], best_center[1], best_beta - 0.02],
                    [best_center[0] + 0.02, best_center[1], best_beta],
                    [best_center[0], best_center[1] + 0.02, best_beta],
                    [best_center[0] - 0.02, best_center[1], best_beta],
                    [best_center[0], best_center[1] - 0.02, best_beta],
                ]
                
                for init in polish_starts:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    res = minimize(objective, init, method='Nelder-Mead',
                                 options={'maxiter': 5000, 'xatol': 1e-13, 'fatol': 1e-13})
                    if res.fun < best_S:
                        best_S = res.fun
                        best_center = (res.x[0], res.x[1])
                        best_beta = ((res.x[2] % (math.pi/6)) + math.pi/6) % (math.pi/6)
        except Exception:
            pass
            
        return best_S, best_beta, best_center

    # Stage 1: Fast screening
    # We have many candidates. We can't run full optimize_shape on all.
    # We will run a simplified version: just the coarse grid search + scalar refine.
    
    def fast_optimize_shape(pts):
        best_S = float('inf')
        best_beta = 0.0
        best_center = None
        
        # Coarse grid
        for i in range(13): # Coarser grid for speed
            beta = i * (math.pi / 6) / 12
            S, center, D = calc_S_for_beta(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = center
        
        # Quick scalar refine
        try:
            res = minimize_scalar(lambda b: calc_S_for_beta(pts, b)[0], bounds=(0, math.pi / 6), method='bounded',
                                 options={'maxiter': 50, 'xatol': 1e-8})
            if res.fun < best_S:
                best_S = res.fun
                best_beta = res.x
                _, best_center, _ = calc_S_for_beta(pts, best_beta)
        except Exception:
            pass
            
        return best_S, best_beta, best_center

    screened = []
    for axial in candidates:
        if time.time() - START_TIME > TIME_LIMIT - 10.0: # Leave 10s for final phase
            break
        pts = axial_to_cartesian(list(axial))
        S, beta, center = fast_optimize_shape(pts)
        screened.append((S, axial, beta, center, pts))

    # Sort by S and take top K
    screened.sort(key=lambda x: x[0])
    K = 20 # Number of candidates to fully optimize
    top_candidates = screened[:K]

    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

    for S_init, axial, beta_init, center_init, pts in top_candidates:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
            
        # Full optimization
        S, beta, center = optimize_shape(pts)
        
        # It's possible the full optimization found something worse due to local minima? 
        # No, it starts from good points. But let's keep the best of init and full.
        if S_init < best_global_S:
            best_global_S = S_init
            best_global_pts = pts
            best_global_beta = beta_init
            best_global_center = center_init
            
        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts
            best_global_beta = beta
            best_global_center = center

    # Fallback if no candidates were processed or all failed
    if best_global_center is None or best_global_S == float('inf'):
        # Use first manual candidate
        fallback_axial = [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1), (1, 1), (-1, 2), (-2, 1), (2, -1)]
        best_global_pts = axial_to_cartesian(fallback_axial)
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
