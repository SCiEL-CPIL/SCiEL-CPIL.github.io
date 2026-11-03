import math
import numpy as np
import time
from scipy.optimize import minimize_scalar, minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Generate candidate polyhexes
    # We use a set of axial coordinates (q, r)
    # Start with the 7-hex flower
    base = {(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)}
    
    # Directions for adding hexes
    dirs = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    
    def get_neighbors(hex_set):
        neighbors = set()
        for q, r in hex_set:
            for dq, dr in dirs:
                nq, nr = q + dq, r + dr
                if (nq, nr) not in hex_set:
                    neighbors.add((nq, nr))
        return neighbors
    
    def canonical_form(hex_set):
        # Normalize by translating so min q, min r is 0? 
        # Better: use a string representation of sorted coords, try all 6 rotations + 2 reflections
        # For speed, just use a frozenset of coords relative to centroid?
        # Actually, for small sets, just sorting is fast enough if we prune early.
        pass

    candidates = set()
    
    # Add the base flower
    candidates.add(frozenset(base))
    
    # BFS to add 4 more hexes
    current_sets = [frozenset(base)]
    # We want to add 4 hexes.
    # To avoid explosion, we only add to the "boundary".
    
    for step in range(4):
        next_sets = []
        for s in current_sets:
            s_set = set(s)
            neighbors = get_neighbors(s_set)
            for n in neighbors:
                new_set = frozenset(s_set | {n})
                # Prune: check if it's already in candidates or next_sets
                if new_set not in candidates and new_set not in next_sets:
                    next_sets.append(new_set)
                    candidates.add(new_set)
        current_sets = next_sets
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        if len(candidates) > 5000: # Limit candidates to avoid memory/time issues
            break

    # Convert candidates to Cartesian points and optimize
    best_S = float('inf')
    best_pts = None
    best_beta = 0.0
    best_center = None
    
    def axial_to_cartesian(hex_set):
        pts = []
        for q, r in hex_set:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def calc_S_fast(pts, beta):
        # Fast calculation of S for a given beta
        normals = np.array([
            [math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0)]
            for k in range(6)
        ])
        # Center is the centroid for fast estimate? 
        # Actually, optimal center minimizes max projection.
        # For a fast estimate, use centroid.
        cx = np.mean(pts[:, 0])
        cy = np.mean(pts[:, 1])
        pts_c = pts - np.array([cx, cy])
        projs = pts_c @ normals.T
        D = np.max(np.abs(projs))
        S = (D + APOTHEM) / APOTHEM
        return S, (cx, cy)

    def optimize_shape(pts):
        best_S_local = float('inf')
        best_beta_local = 0.0
        best_center_local = None
        
        # Coarse grid
        for i in range(37):
            beta = i * (math.pi / 6) / 36
            S, center = calc_S_fast(pts, beta)
            if S < best_S_local:
                best_S_local = S
                best_beta_local = beta
                best_center_local = center
        
        # Refine
        try:
            res = minimize_scalar(lambda b: calc_S_fast(pts, b)[0], bounds=(0, math.pi / 6), method='bounded',
                                 options={'maxiter': 100, 'xatol': 1e-10})
            if res.fun < best_S_local:
                best_S_local = res.fun
                best_beta_local = res.x
                _, best_center_local = calc_S_fast(pts, best_beta_local)
        except Exception:
            pass
            
        # Polish center with Newton
        try:
            cx, cy = best_center_local
            normals = np.array([
                [math.cos(best_beta_local + math.pi / 6.0 + k * math.pi / 3.0),
                 math.sin(best_beta_local + math.pi / 6.0 + k * math.pi / 3.0)]
                for k in range(6)
            ])
            for _ in range(20):
                pts_c = pts - np.array([cx, cy])
                projs = pts_c @ normals.T
                abs_projs = np.abs(projs)
                D = np.max(abs_projs)
                active_mask = abs_projs >= (D - 1e-8)
                active_indices = np.argwhere(active_mask)
                if len(active_indices) == 0: break
                A_rows = []
                b_rows = []
                for idx in active_indices:
                    i, j = idx
                    px, py = pts[i]
                    nx, ny = normals[j]
                    sign = np.sign(projs[i, j])
                    if sign == 0: sign = 1
                    A_rows.append([nx, ny])
                    b_rows.append(sign * D - (px * nx + py * ny))
                A = np.array(A_rows)
                b = np.array(b_rows)
                sol, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                dx, dy = sol
                cx += dx
                cy += dy
                if abs(dx) < 1e-12 and abs(dy) < 1e-12: break
            best_center_local = (cx, cy)
            S_final, _ = calc_S_fast(pts, best_beta_local) # Recalc with new center? 
            # Actually calc_S_fast uses centroid. Let's calc exactly.
            normals = np.array([
                [math.cos(best_beta_local + math.pi / 6.0 + k * math.pi / 3.0),
                 math.sin(best_beta_local + math.pi / 6.0 + k * math.pi / 3.0)]
                for k in range(6)
            ])
            pts_c = pts - np.array([cx, cy])
            projs = pts_c @ normals.T
            D = np.max(np.abs(projs))
            S_final = (D + APOTHEM) / APOTHEM
            if S_final < best_S_local:
                best_S_local = S_final
        except Exception:
            pass
            
        return best_S_local, best_beta_local, best_center_local

    # Iterate through candidates
    for i, cand in enumerate(candidates):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        if i % 100 == 0 and time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
            
        pts = axial_to_cartesian(cand)
        S, beta, center = optimize_shape(pts)
        if S < best_S:
            best_S = S
            best_pts = pts
            best_beta = beta
            best_center = center

    if best_pts is None:
        # Fallback to the 7-hex flower + 4 random
        pts = axial_to_cartesian(base)
        best_center = (float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])))
        best_beta = 0.0
        normals = np.array([
            [math.cos(math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(math.pi / 6.0 + k * math.pi / 3.0)]
            for k in range(6)
        ])
        pts_c = pts - np.array(best_center)
        projs = pts_c @ normals.T
        D = np.max(np.abs(projs))
        best_S = (D + APOTHEM) / APOTHEM

    cx, cy = best_center
    angle_deg = math.degrees(best_beta)
    
    inner = []
    for x, y in best_pts:
        inner.append([float(x), float(y), float(angle_deg)])
        
    center = [float(cx), float(cy)]
    side = float(best_S)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation
