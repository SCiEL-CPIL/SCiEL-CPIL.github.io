import math
import numpy as np
import time
from scipy.optimize import linprog, minimize, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    def hex_vertices(cx, cy, angle_deg, s=1.0):
        a = math.radians(angle_deg)
        verts = []
        for k in range(6):
            theta = a + k * math.pi / 3.0
            verts.append((cx + s * math.cos(theta), cy + s * math.sin(theta)))
        return np.array(verts)

    def minkowski_sum(v1, v2):
        # v1, v2: (6,2) arrays
        sums = []
        for a in v1:
            for b in v2:
                sums.append(a + b)
        return np.array(sums)

    def point_in_convex_polygon(p, poly):
        n = len(poly)
        if n < 3:
            return False
        for i in range(n):
            a = poly[i]
            b = poly[(i + 1) % n]
            cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
            if cross < -1e-9:
                return False
        return True

    def check_non_overlap(positions, angles_deg, centers):
        # positions: (n,2) base offsets, angles_deg: (n,) angles, centers: (2,) outer center
        # Check all pairs
        n = len(positions)
        for i in range(n):
            for j in range(i + 1, n):
                vi = hex_vertices(0, 0, angles_deg[i])
                vj = hex_vertices(0, 0, angles_deg[j])
                mink = minkowski_sum(vi, -vj)
                # Check if positions[i] - positions[j] is in mink
                diff = positions[i] - positions[j]
                if point_in_convex_polygon(diff, mink):
                    return False
        return True

    def enclosing_hex_size(positions, angles_deg, beta):
        # Find minimum enclosing hexagon of side S centered at (cx,cy) rotated by beta
        # All hexagons have same orientation angles_deg[k], outer hex rotated by beta
        # The outer hexagon has 6 half-plane constraints
        # For each inner hex vertex, project onto each outer normal
        n = len(positions)
        # Collect all vertices
        all_verts = []
        for k in range(n):
            verts = hex_vertices(0, 0, angles_deg[k]) + positions[k]
            all_verts.append(verts)
        all_verts = np.vstack(all_verts)  # (6n, 2)
        
        # Outer hexagon normals
        angles = beta + np.pi / 6.0 + np.arange(6) * np.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        
        # LP: minimize D subject to |n_k · (p - c)| <= D for all k, p
        n_pts = len(all_verts)
        A_ub = []
        b_ub = []
        for nx, ny in normals:
            projs = all_verts[:, 0] * nx + all_verts[:, 1] * ny
            A_ub.append(np.column_stack([-nx * np.ones(n_pts), -ny * np.ones(n_pts), -np.ones(n_pts)]))
            b_ub.append(-projs)
            A_ub.append(np.column_stack([nx * np.ones(n_pts), ny * np.ones(n_pts), -np.ones(n_pts)]))
            b_ub.append(projs)
        A_ub = np.vstack(A_ub)
        b_ub = np.concatenate(b_ub)
        bounds = [(None, None), (None, None), (0, None)]
        try:
            res = linprog(np.array([0.0, 0.0, 1.0]), A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                D = res.x[2]
                cx, cy = res.x[0], res.x[1]
                S = (D + APOTHEM) / APOTHEM
                return S, (cx, cy)
            else:
                return float('inf'), None
        except Exception:
            return float('inf'), None

    # Generate candidate configurations
    # Each config: list of (x, y, angle_deg)
    candidates = []

    # All-aligned configurations (from seed)
    axial_sets = [
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(1,1),(-1,2),(-2,1),(2,-1)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)],
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,1),(3,1)],
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)],
        [(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)],
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3)],
    ]
    for axial in axial_sets:
        pts = []
        for q, r in axial:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y, 0.0))
        candidates.append(pts)

    # Mixed orientation configurations
    # Base all-aligned, then flip some hexagons to 30°
    for axial in axial_sets[:3]:
        base = []
        for q, r in axial:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            base.append((x, y))
        # Try flipping different subsets
        # Flip the outer hexagons
        n = len(base)
        # Try flipping 1, 2, 3 hexagons
        for i in range(n):
            config = [(base[k][0], base[k][1], 30.0 if k == i else 0.0) for k in range(n)]
            candidates.append(config)
        for i in range(n):
            for j in range(i+1, n):
                config = [(base[k][0], base[k][1], 30.0 if k in (i,j) else 0.0) for k in range(n)]
                candidates.append(config)

    # More diverse: try 30° on all
    for axial in axial_sets[:2]:
        pts = []
        for q, r in axial:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y, 30.0))
        candidates.append(pts)

    # Interlocking configurations: alternate 0° and 30°
    for axial in axial_sets[:2]:
        base = []
        for q, r in axial:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            base.append((x, y))
        n = len(base)
        config = [(base[k][0], base[k][1], 30.0 if k % 2 == 0 else 0.0) for k in range(n)]
        candidates.append(config)
        config2 = [(base[k][0], base[k][1], 0.0 if k % 2 == 0 else 30.0) for k in range(n)]
        candidates.append(config2)

    # Evaluate each candidate
    best_S = float('inf')
    best_config = None
    best_beta = 0.0
    best_center = None

    for config in candidates:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        positions = np.array([(p[0], p[1]) for p in config])
        angles = np.array([p[2] for p in config])
        
        # Check non-overlap
        if not check_non_overlap(positions, angles, np.zeros(2)):
            continue
        
        # Optimize outer hexagon rotation
        best_local_S = float('inf')
        best_local_beta = 0.0
        best_local_center = None
        
        # Coarse grid
        for i in range(37):
            beta = i * (math.pi / 6) / 36
            S, center = enclosing_hex_size(positions, angles, beta)
            if S < best_local_S:
                best_local_S = S
                best_local_beta = beta
                best_local_center = center
        
        # Refine
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                def obj(beta):
                    S, _ = enclosing_hex_size(positions, angles, beta)
                    return S
                res = minimize_scalar(obj, bounds=(0, math.pi/6), method='bounded', options={'maxiter': 200})
                if res.fun < best_local_S:
                    best_local_S = res.fun
                    best_local_beta = res.x
                    _, best_local_center = enclosing_hex_size(positions, angles, best_local_beta)
        except Exception:
            pass
        
        if best_local_S < best_S:
            best_S = best_local_S
            best_config = config
            best_beta = best_local_beta
            best_center = best_local_center

    # If no valid config found, fall back
    if best_config is None:
        # Simplest valid: all at 0° in a ring
        config = []
        config.append((0.0, 0.0, 0.0))
        for k in range(6):
            x = math.sqrt(3) * math.cos(k * math.pi/3)
            y = math.sqrt(3) * math.sin(k * math.pi/3)
            config.append((x, y, 0.0))
        # Add 4 more in a ring
        for k in range(4):
            x = 2 * math.sqrt(3) * math.cos(k * math.pi/2)
            y = 2 * math.sqrt(3) * math.sin(k * math.pi/2)
            config.append((x, y, 0.0))
        best_config = config
        positions = np.array([(p[0], p[1]) for p in best_config])
        angles = np.array([p[2] for p in best_config])
        best_S, best_center = enclosing_hex_size(positions, angles, 0.0)
        best_beta = 0.0
        if best_center is None:
            best_center = (0.0, 0.0)

    # Polish: optimize centers and beta jointly
    positions = np.array([(p[0], p[1]) for p in best_config])
    angles = np.array([p[2] for p in best_config])
    n = len(positions)

    def joint_objective(params):
        # params: [cx, cy, beta, pos_flat...]
        cx, cy = params[0], params[1]
        beta = ((params[2] % (math.pi/6)) + math.pi/6) % (math.pi/6)
        pos = params[3:].reshape(n, 2)
        
        # Check non-overlap
        for i in range(n):
            for j in range(i+1, n):
                vi = hex_vertices(0, 0, angles[i])
                vj = hex_vertices(0, 0, angles[j])
                mink = minkowski_sum(vi, -vj)
                diff = pos[i] - pos[j]
                if point_in_convex_polygon(diff, mink):
                    return 1e10
        
        all_verts = []
        for k in range(n):
            verts = hex_vertices(0, 0, angles[k]) + pos[k]
            all_verts.append(verts)
        all_verts = np.vstack(all_verts)
        
        outer_angles = beta + np.pi/6.0 + np.arange(6) * np.pi/3.0
        normals = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
        
        max_proj = 0
        for nx, ny in normals:
            projs = all_verts[:, 0] * nx + all_verts[:, 1] * ny
            max_proj = max(max_proj, np.max(projs) - np.min(projs))
        
        S = (max_proj/2.0 + APOTHEM) / APOTHEM
        return S

    # Try polishing from current best
    if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
        init = [best_center[0], best_center[1], best_beta] + positions.flatten().tolist()
        try:
            res = minimize(joint_objective, init, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12})
            if res.fun < best_S and res.fun < 1e9:
                best_S = res.fun
                best_center = (res.x[0], res.x[1])
                best_beta = ((res.x[2] % (math.pi/6)) + math.pi/6) % (math.pi/6)
                positions = res.x[3:].reshape(n, 2)
        except Exception:
            pass

    # Build output
    inner = []
    for k in range(n):
        inner.append([float(positions[k][0]), float(positions[k][1]), float(angles[k])])
    
    center = [float(best_center[0]), float(best_center[1])]
    side = float(best_S)
    rotation = float(math.degrees(best_beta))
    
    return inner, center, side, rotation
