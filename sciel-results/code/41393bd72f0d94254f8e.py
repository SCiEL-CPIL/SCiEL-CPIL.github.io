import math
import numpy as np
import time
from scipy.optimize import linprog, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Strategy:
    # 1. Define a set of rigid configurations of 11 hexagons.
    #    We focus on "triangular" packings which are dense.
    # 2. For each configuration, we assume the inner hexagons are all aligned (angle 0).
    #    This is optimal for triangular packings.
    # 3. We optimize the outer hexagon's rotation (beta), center (cx, cy), and scale (S).
    #    Since the inner hexagons are rigid relative to each other, we can center them at (0,0).
    #    The outer hexagon is defined by 6 half-spaces.
    #    We want to minimize S such that all inner hexagon vertices are inside.
    #    
    #    Actually, the inner hexagons have size. The condition is that the Minkowski sum of
    #    the set of inner hexagons and a unit hexagon (dilated by apothem? No, the inner hexagons
    #    are the objects. The outer hexagon must contain them.
    #    
    #    Let's refine the geometry:
    #    Inner hexagon i is at position p_i. It is a regular hexagon of side 1.
    #    The outer hexagon is centered at C, rotated by beta, side length S.
    #    The outer hexagon is the intersection of 6 half-planes:
    #    n_k . (x - C) <= D_outer, where D_outer = S * APOTHEM.
    #    The normals n_k are at angles beta + pi/6 + k*pi/3.
    #    
    #    For the inner hexagon at p_i to be inside, ALL its vertices must be inside.
    #    Equivalently, the maximum projection of the inner hexagon onto n_k must be <= n_k . C + D_outer.
    #    The max projection of a unit hexagon centered at p_i onto direction n is:
    #    n . p_i + (max projection of unit hexagon at origin onto n).
    #    
    #    Let R(n) be the radius of the unit hexagon in direction n (distance from center to vertex/edge).
    #    For a regular hexagon of side 1 (apothem A = sqrt(3)/2):
    #    The support function h(n) is the max dot product with vertices.
    #    Vertices are at angles 0, 60, ... relative to the hexagon's orientation.
    #    If the inner hexagons are aligned with the global axes (angle 0), their vertices are fixed.
    #    h(n) depends on the angle of n relative to the hexagon.
    #    
    #    So the constraint for direction k is:
    #    n_k . C + D_outer >= max_i ( n_k . p_i + h(n_k) )
    #    
    #    We want to minimize S = D_outer / APOTHEM.
    #    Variables: Cx, Cy, D_outer.
    #    Minimize D_outer.
    #    Subject to:
    #    n_k . C + D_outer - max_i(n_k . p_i) - h(n_k) >= 0  (for k=0..5)
    #    -n_k . C + D_outer - max_i(-n_k . p_i) - h(-n_k) >= 0 (for k=0..5)
    #    
    #    This is a Linear Program!
    #    Variables: x = [Cx, Cy, D]
    #    Objective: c = [0, 0, 1]
    #    Constraints: A_ub * x <= b_ub
    
    # Precompute support function h(angle) for a unit hexagon (side 1, apothem A) aligned with x-axis.
    # Vertices of unit hexagon (side 1):
    # (1, 0), (0.5, A), (-0.5, A), (-1, 0), (-0.5, -A), (0.5, -A)
    # Note: Side length 1 means distance from center to vertex is 1.
    # Apothem is sqrt(3)/2.
    # Let's verify vertices for side length 1.
    # If side length is 1, the circumradius is 1.
    # Yes, vertices are at distance 1 from center.
    
    hex_verts = np.array([
        [1.0, 0.0],
        [0.5, APOTHEM],
        [-0.5, APOTHEM],
        [-1.0, 0.0],
        [-0.5, -APOTHEM],
        [0.5, -APOTHEM]
    ])
    
    def support_function(angle):
        # Max dot product of hex_verts with unit vector at 'angle'
        n = np.array([math.cos(angle), math.sin(angle)])
        return np.max(hex_verts @ n)

    # Define candidate configurations in Cartesian coordinates
    # We use axial coordinates to generate dense packings, then convert.
    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    # Generate a diverse set of "rigid" configurations
    # We want configurations that are "compact" (small bounding box)
    
    candidates = []
    
    # Config 1: Central + 6 ring + 4 extra (from seed, but we will re-evaluate)
    axial_1 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ]
    candidates.append(axial_1)

    # Config 2: Triangular block 3x3 minus corners?
    # Let's try to generate configurations programmatically to ensure density
    # We'll use a simple heuristic: start with a hexagon, add neighbors that minimize bounding box growth
    
    def add_neighbor(pts, q, r):
        # Check if (q, r) is adjacent to any existing point
        existing = set(pts)
        neighbors = [(q+1, r), (q-1, r), (q, r+1), (q, r-1), (q+1, r-1), (q-1, r+1)]
        for n in neighbors:
            if n in existing:
                return True
        return False

    def generate_config(start, n_target):
        # Greedy growth
        pts = [start]
        visited = set(pts)
        while len(pts) < n_target:
            best_point = None
            best_score = float('inf')
            # Try adding neighbors of existing points
            candidates_to_add = []
            for p in pts:
                q, r = p
                neighbors = [(q+1, r), (q-1, r), (q, r+1), (q, r-1), (q+1, r-1), (q-1, r+1)]
                for n in neighbors:
                    if n not in visited:
                        candidates_to_add.append(n)
            
            for n in candidates_to_add:
                # Score: distance from center (0,0) to maintain compactness
                # Or score: increase in bounding box
                # Let's use distance from centroid of current set
                cx = sum(p[0] for p in pts) / len(pts)
                cy = sum(p[1] for p in pts) / len(pts)
                # Convert new point to cartesian for distance
                nx = math.sqrt(3) * (n[0] + n[1] / 2.0)
                ny = 1.5 * n[1]
                dist = (nx - cx)**2 + (ny - cy)**2
                if dist < best_score:
                    best_score = dist
                    best_point = n
            
            if best_point:
                pts.append(best_point)
                visited.add(best_point)
            else:
                break
        return pts

    # Generate several random starting seeds for the greedy growth
    import random
    random.seed(42)
    
    for i in range(20):
        start = (random.randint(-2, 2), random.randint(-2, 2))
        config = generate_config(start, N)
        if len(config) == N:
            candidates.append(config)

    # Also include some hand-crafted dense ones
    axial_2 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (1, 2), (2, 2), (3, 2)
    ]
    candidates.append(axial_2)
    
    axial_3 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3)
    ]
    candidates.append(axial_3)

    axial_4 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_4)
    
    # Deduplicate configurations (by set of axial coords)
    unique_candidates = []
    seen = set()
    for c in candidates:
        key = frozenset(c)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)
    candidates = unique_candidates

    def solve_enclosing_hexagon(pts, beta):
        # pts: (N, 2) array of inner hexagon centers
        # beta: rotation of outer hexagon
        # Returns S (side length) or inf if infeasible
        
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), None, None
            
        normals = []
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append(np.array([math.cos(ang), math.sin(ang)]))
        normals = np.array(normals)
        
        # Precompute max projection of inner centers for each normal
        # max_proj_pos[k] = max_i (n_k . p_i)
        # max_proj_neg[k] = max_i (-n_k . p_i)
        
        projections = pts @ normals.T # (N, 6)
        max_proj_pos = np.max(projections, axis=0) # (6,)
        max_proj_neg = np.max(-projections, axis=0) # (6,)
        
        # Support function h(n) for the unit hexagon
        # h(n_k) and h(-n_k)
        h_pos = np.array([support_function(beta + math.pi / 6.0 + k * math.pi / 3.0) for k in range(6)])
        h_neg = np.array([support_function(-(beta + math.pi / 6.0 + k * math.pi / 3.0)) for k in range(6)])
        
        # Constraints:
        # n_k . C + D - max_proj_pos[k] - h_pos[k] >= 0  => -n_k . C - D <= -max_proj_pos[k] - h_pos[k]
        # -n_k . C + D - max_proj_neg[k] - h_neg[k] >= 0  => n_k . C - D <= max_proj_neg[k] + h_neg[k]
        
        A_ub = []
        b_ub = []
        
        for k in range(6):
            nx, ny = normals[k]
            # Constraint 1: n_k . C + D >= max_proj_pos[k] + h_pos[k]
            A_ub.append([-nx, -ny, -1.0])
            b_ub.append(-(max_proj_pos[k] + h_pos[k]))
            
            # Constraint 2: -n_k . C + D >= max_proj_neg[k] + h_neg[k]
            A_ub.append([nx, ny, -1.0])
            b_ub.append(max_proj_neg[k] + h_neg[k])
            
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        c_obj = np.array([0.0, 0.0, 1.0])
        bounds = [(None, None), (None, None), (0, None)]
        
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                D = res.x[2]
                S = D / APOTHEM
                return S, (res.x[0], res.x[1]), D
            else:
                return float('inf'), None, None
        except Exception:
            return float('inf'), None, None

    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None
    
    # Sort candidates by a quick heuristic: bounding box area
    def quick_score(axial_list):
        pts = axial_to_cartesian(axial_list)
        # Normalize to center at 0
        cx = np.mean(pts[:, 0])
        cy = np.mean(pts[:, 1])
        pts_centered = pts - [cx, cy]
        # Max extent in 6 directions
        max_ext = 0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = pts_centered[:, 0] * nx + pts_centered[:, 1] * ny
            max_ext = max(max_ext, np.max(projs), -np.min(projs))
        return (max_ext + APOTHEM) / APOTHEM

    scored_candidates = [(quick_score(c), idx, c) for idx, c in enumerate(candidates)]
    scored_candidates.sort(key=lambda x: x[0])
    
    # Process top candidates
    # We will do a 1D minimization over beta for each candidate
    for _, _, axial in scored_candidates:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
            
        pts = axial_to_cartesian(axial)
        
        # Center the points to help the LP
        cx_mean = np.mean(pts[:, 0])
        cy_mean = np.mean(pts[:, 1])
        pts_centered = pts - [cx_mean, cy_mean]
        
        # Objective function for minimize_scalar
        def obj_beta(beta):
            S, center, D = solve_enclosing_hexagon(pts_centered, beta)
            if S == float('inf'):
                return 1e10
            return S
        
        # Scan beta in [0, pi/6)
        # We can use a grid first, then refine
        best_S_for_config = float('inf')
        best_beta_for_config = 0.0
        best_center_for_config = None
        
        # Coarse grid
        grid_size = 50
        for i in range(grid_size):
            beta = i * (math.pi / 6) / grid_size
            S, center, D = solve_enclosing_hexagon(pts_centered, beta)
            if S < best_S_for_config:
                best_S_for_config = S
                best_beta_for_config = beta
                best_center_for_config = center
        
        # Refine with minimize_scalar
        if best_S_for_config < float('inf'):
            try:
                # Find the local minimum around the best grid point
                # Since the function is periodic and piecewise linear (convex in D for fixed beta, but beta makes it non-convex),
                # we should check neighbors or use a bounded minimizer.
                # The function S(beta) is likely unimodal or has few minima.
                
                # Let's try minimize_scalar on the interval [best_beta - step, best_beta + step]
                step = (math.pi / 6) / grid_size
                lower = max(0, best_beta_for_config - step)
                upper = min(math.pi / 6, best_beta_for_config + step)
                
                res = minimize_scalar(obj_beta, bounds=(lower, upper), method='bounded', options={'xatol': 1e-10})
                if res.fun < best_S_for_config:
                    best_S_for_config = res.fun
                    best_beta_for_config = res.x
                    # Get center
                    _, best_center_for_config, _ = solve_enclosing_hexagon(pts_centered, best_beta_for_config)
            except Exception:
                pass
        
        if best_S_for_config < best_global_S:
            best_global_S = best_S_for_config
            best_global_pts = pts_centered
            best_global_beta = best_beta_for_config
            # Center from LP is relative to pts_centered. 
            # We need to output absolute coordinates.
            # The LP found center (cx, cy) such that outer hexagon centered at (cx, cy) contains pts_centered.
            # So absolute center is cx_mean + cx, cy_mean + cy.
            if best_center_for_config:
                best_global_center = (cx_mean + best_center_for_config[0], cy_mean + best_center_for_config[1])
            else:
                best_global_center = (cx_mean, cy_mean)

    # If no solution found, fallback
    if best_global_center is None or best_global_S == float('inf'):
        # Use first candidate
        axial = candidates[0]
        pts = axial_to_cartesian(axial)
        cx_mean = np.mean(pts[:, 0])
        cy_mean = np.mean(pts[:, 1])
        pts_centered = pts - [cx_mean, cy_mean]
        S, center, D = solve_enclosing_hexagon(pts_centered, 0.0)
        if S == float('inf'):
            S = 10.0
            center = (0,0)
        best_global_S = S
        best_global_pts = pts_centered
        best_global_beta = 0.0
        if center:
            best_global_center = (cx_mean + center[0], cy_mean + center[1])
        else:
            best_global_center = (cx_mean, cy_mean)

    # Final Output Preparation
    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)
    
    # The inner hexagons are at best_global_pts (centered at 0,0 relative to their own centroid? No, relative to the LP center)
    # Wait, in the LP, we used pts_centered (shifted by cx_mean, cy_mean).
    # The LP found that an outer hexagon centered at (cx, cy) [in the pts_centered frame] contains pts_centered.
    # So in the global frame, the inner hexagons are at pts_centered + [cx_mean, cy_mean].
    # And the outer hexagon is centered at (cx, cy) in the pts_centered frame, which is (cx + cx_mean, cy + cy_mean) in the global frame.
    # This matches best_global_center.
    
    # So the inner hexagon positions in the global frame are:
    global_pts = best_global_pts + np.array([cx_mean, cy_mean]) if 'cx_mean' in locals() else best_global_pts
    
    # Wait, 'cx_mean' is local to the loop. I need to store it.
    # Let's re-derive from best_global_center and best_global_pts.
    # best_global_pts are the points used in the LP (centered at 0,0? No, they were pts_centered).
    # Let's just store the global points directly in the loop to avoid confusion.
    
    # Re-doing the loop logic slightly to store global points
    # Actually, I can just transform best_global_pts back.
    # In the loop: pts_centered = pts - [cx_mean, cy_mean].
    # So pts = pts_centered + [cx_mean, cy_mean].
    # I didn't store cx_mean globally.
    # Let's just recalculate the global points from the best candidate found.
    
    # To be safe, I will re-run the best candidate's calculation to get the exact global coordinates.
    # But I don't have the axial list for the best candidate stored separately.
    # I'll just assume the stored best_global_pts are the coordinates relative to the outer center?
    # No, best_global_pts are the coordinates in the frame where the outer center is (cx, cy).
    # So the global coordinates of the inner hexagons are best_global_pts + best_global_center?
    # No.
    # LP variables: Cx, Cy are the center of the outer hexagon in the frame of pts_centered.
    # pts_centered are the inner hexagon centers.
    # So, relative to the outer hexagon center, the inner hexagons are at pts_centered - C.
    # But the output requires (x, y) of inner hexagons in the global frame.
    # And (cx, cy) of outer hexagon in the global frame.
    
    # Let's trace:
    # 1. axial -> pts (global-ish, centered at mean)
    # 2. pts_centered = pts - mean
    # 3. LP finds C (outer center in pts_centered frame) and D.
    # 4. best_global_center = mean + C (outer center in global frame)
    # 5. best_global_pts = pts_centered (inner centers in pts_centered frame)
    # 
    # To get inner centers in global frame:
    # inner_global = best_global_pts + mean
    # 
    # I didn't store 'mean'.
    # However, I can compute 'mean' if I store the axial list.
    # Let's modify the loop to store the best axial list.

    # --- Corrected Loop Implementation ---
    best_global_S = float('inf')
    best_global_pts_global = None
    best_global_beta = 0.0
    best_global_center_global = None
    best_axial = None

    for _, _, axial in scored_candidates:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
            
        pts = axial_to_cartesian(axial)
        cx_mean = np.mean(pts[:, 0])
        cy_mean = np.mean(pts[:, 1])
        pts_centered = pts - [cx_mean, cy_mean]
        
        def obj_beta(beta):
            S, center, D = solve_enclosing_hexagon(pts_centered, beta)
            if S == float('inf'):
                return 1e10
            return S
        
        best_S_for_config = float('inf')
        best_beta_for_config = 0.0
        best_center_for_config = None
        
        grid_size = 50
        for i in range(grid_size):
            beta = i * (math.pi / 6) / grid_size
            S, center, D = solve_enclosing_hexagon(pts_centered, beta)
            if S < best_S_for_config:
                best_S_for_config = S
                best_beta_for_config = beta
                best_center_for_config = center
        
        if best_S_for_config < float('inf'):
            try:
                step = (math.pi / 6) / grid_size
                lower = max(0, best_beta_for_config - step)
                upper = min(math.pi / 6, best_beta_for_config + step)
                res = minimize_scalar(obj_beta, bounds=(lower, upper), method='bounded', options={'xatol': 1e-10})
                if res.fun < best_S_for_config:
                    best_S_for_config = res.fun
                    best_beta_for_config = res.x
                    _, best_center_for_config, _ = solve_enclosing_hexagon(pts_centered, best_beta_for_config)
            except Exception:
                pass
        
        if best_S_for_config < best_global_S:
            best_global_S = best_S_for_config
            best_global_beta = best_beta_for_config
            best_axial = axial
            # Store global center
            if best_center_for_config:
                best_global_center_global = (cx_mean + best_center_for_config[0], cy_mean + best_center_for_config[1])
            else:
                best_global_center_global = (cx_mean, cy_mean)
            # Store global inner points
            # Inner points in global frame = pts_centered + [cx_mean, cy_mean] = pts
            # Wait, pts_centered is just pts - mean.
            # The LP was run on pts_centered.
            # The result C is the center in the pts_centered frame.
            # The inner points are at pts_centered.
            # So in the global frame, inner points are at pts_centered + [cx_mean, cy_mean] = pts.
            # So best_global_pts_global = pts.
            best_global_pts_global = pts

    if best_global_center_global is None:
        axial = candidates[0]
        pts = axial_to_cartesian(axial)
        cx_mean = np.mean(pts[:, 0])
        cy_mean = np.mean(pts[:, 1])
        pts_centered = pts - [cx_mean, cy_mean]
        S, center, D = solve_enclosing_hexagon(pts_centered, 0.0)
        if S == float('inf'): S = 10.0
        if center is None: center = (0,0)
        best_global_S = S
        best_global_beta = 0.0
        best_global_center_global = (cx_mean + center[0], cy_mean + center[1])
        best_global_pts_global = pts

    cx, cy = best_global_center_global
    angle_deg = math.degrees(best_global_beta)
    
    inner = []
    for x, y in best_global_pts_global:
        inner.append([float(x), float(y), float(angle_deg)])
        
    center = [float(cx), float(cy)]
    side = float(best_global_S)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation
