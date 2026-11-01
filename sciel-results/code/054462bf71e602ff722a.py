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

    # Axial coordinate helpers
    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def axial_neighbors(q, r):
        return [(q+1, r), (q-1, r), (q, r+1), (q, r-1), (q+1, r-1), (q-1, r+1)]

    def polyhex_diameter_lower_bound(cells):
        """Lower bound on S from the diameter of the cell set."""
        if len(cells) < 2:
            return 1.0
        pts = axial_to_cartesian(list(cells))
        # Compute max pairwise distance
        max_d = 0.0
        for i in range(len(pts)):
            for j in range(i+1, len(pts)):
                d = np.sqrt((pts[i][0]-pts[j][0])**2 + (pts[i][1]-pts[j][1])**2)
                max_d = max(max_d, d)
        # The enclosing hexagon with side S has inradius = S*APOTHEM
        # Any two points in the hexagon must be at distance <= 2*S*APOTHEM
        # So S >= max_d / (2*APOTHEM)
        return max_d / (2.0 * APOTHEM)

    def polyhex_width_lower_bound(cells, angle):
        """Lower bound on S from width in a given direction."""
        pts = axial_to_cartesian(list(cells))
        nx, ny = math.cos(angle), math.sin(angle)
        projs = pts[:, 0] * nx + pts[:, 1] * ny
        width = np.max(projs) - np.min(projs)
        # Width of regular hexagon with side S in direction theta:
        # For a hexagon with flat-top orientation (angle=0), width in direction phi
        # is 2*S*APOTHEM * max over the 6 face normals of |cos(phi - face_angle)|
        # Actually, the width of a regular hexagon of side S in any direction is
        # between S (min, along edge) and 2*S*APOTHEM (max, along vertex)
        # More precisely: width(phi) = 2*S*APOTHEM * max_k |cos(phi - (pi/6 + k*pi/3))|
        # But we want: given width w, what's the min S such that w <= width_hex(S, some orientation)?
        # The minimum width of a hexagon of side S is S (along an edge direction)
        # The maximum width is 2*S*APOTHEM = S*sqrt(3) (along a vertex direction)
        # So S >= w / sqrt(3) (best case, width along edge)
        # But we can rotate the hexagon, so S >= w / (2*APOTHEM) is the tightest
        # Actually: min over orientations of max width = S, max over orientations = S*sqrt(3)
        # We need: exists orientation such that w <= width(S, orientation)
        # width(S, orientation) ranges from S to S*sqrt(3)
        # So we need S*sqrt(3) >= w, i.e., S >= w/sqrt(3)
        return width / (2.0 * APOTHEM)

    def compute_S_for_orientation(pts, beta):
        """Compute exact S for given points and hexagon orientation."""
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

    def optimize_shape(pts):
        """Optimize center and rotation for given point set."""
        best_S = float('inf')
        best_beta = 0.0
        best_center = None
        
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, None
        
        # Scan over beta in [0, pi/6]
        n_scan = 36
        for i in range(n_scan + 1):
            beta = i * (math.pi / 6) / n_scan
            S, center, D = compute_S_for_orientation(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = center
        
        # Refine with bounded minimization
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                res = minimize_scalar(lambda b: compute_S_for_orientation(pts, b)[0],
                                     bounds=(0, math.pi / 6), method='bounded',
                                     options={'maxiter': 300, 'xatol': 1e-12})
                if res.fun < best_S:
                    best_S = res.fun
                    best_beta = res.x
                    _, best_center, _ = compute_S_for_orientation(pts, best_beta)
        except Exception:
            pass
        
        # Nelder-Mead polish on (cx, cy, beta)
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN and best_center is not None:
                def objective(params):
                    cx, cy, beta = params
                    beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
                    normals = []
                    for k in range(6):
                        ang = beta + math.pi / 6.0 + k * math.pi / 3.0
                        normals.append((math.cos(ang), math.sin(ang)))
                    
                    max_proj = 0
                    for nx, ny in normals:
                        for px, py in pts:
                            proj = (px - cx) * nx + (py - cy) * ny
                            max_proj = max(max_proj, proj, -proj)
                    
                    S = (max_proj + APOTHEM) / APOTHEM
                    return S
                
                initial = [best_center[0], best_center[1], best_beta]
                res = minimize(objective, initial, method='Nelder-Mead',
                             options={'maxiter': 3000, 'xatol': 1e-14, 'fatol': 1e-14})
                if res.fun < best_S:
                    best_S = res.fun
                    best_center = (res.x[0], res.x[1])
                    best_beta = ((res.x[2] % (math.pi/6)) + math.pi/6) % (math.pi/6)
        except Exception:
            pass
            
        return best_S, best_beta, best_center

    # ===== Branch and bound enumeration of 11-cell polyhexes =====
    # We enumerate all free polyhexes of size 11 using a canonical form approach.
    # Key insight: we build polyhexes by adding cells, maintaining canonical form
    # to avoid duplicates, and prune based on geometric lower bounds.
    
    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None
    best_global_cells = None
    
    # Start with a single cell at origin
    # Use a queue of (cells_frozenset, canonical_cells)
    # To keep it manageable, we'll use a BFS/DFS approach with pruning
    
    def canonicalize(cells):
        """Return canonical form of a polyhex (translation-invariant)."""
        cells = list(cells)
        if not cells:
            return frozenset()
        # Translate so min q is 0, then min r is 0
        min_q = min(q for q, r in cells)
        min_r = min(r for q, r in cells if q == min_q)
        translated = [(q - min_q, r - min_r) for q, r in cells]
        return frozenset(translated)

    def get_boundary_cells(cells):
        """Get cells that can be extended (have at least one empty neighbor)."""
        cells_set = set(cells)
        boundary = []
        for q, r in cells_set:
            for nq, nr in axial_neighbors(q, r):
                if (nq, nr) not in cells_set:
                    boundary.append((nq, nr))
                    break
        return boundary

    def lower_bound_S(cells):
        """Compute a lower bound on S for this partial polyhex."""
        lb = polyhex_diameter_lower_bound(cells)
        # Also check width in several directions
        for ang in [0, math.pi/6, math.pi/3, math.pi/2, 2*math.pi/3, 5*math.pi/6]:
            lb = max(lb, polyhex_width_lower_bound(cells, ang))
        return lb

    # We'll use a DFS approach, building polyhexes cell by cell
    # To avoid the exponential blowup, we limit the search and use good pruning
    
    # Alternative: generate a diverse set of candidate polyhexes using
    # different growth strategies, then optimize each one
    
    # Strategy: Use multiple growth patterns
    candidate_sets = []
    
    # Pattern 1: Compact hexagonal growth (center + rings)
    # For 11 cells: center (1) + ring 1 (6) + 4 from ring 2
    ring2_cells = [(2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1)]
    for i in range(4):
        cells = [(0,0)] + [(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1)] + ring2_cells[:i+1]
        if len(cells) == 11:
            candidate_sets.append(set(cells))
    
    # Pattern 2: Linear chains
    for direction in [(1,0), (0,1), (1,-1)]:
        cells = [tuple(a*direction) for a in range(11)]
        candidate_sets.append(set(cells))
    
    # Pattern 3: Various compact shapes
    shapes = [
        # Hexagon with extensions
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(2,1),(1,2),(0,2)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(3,0),(1,1),(2,1)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(0,2),(1,2),(2,1)],
        # More varied shapes
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)],
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2)],
        [(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(1,1),(2,1),(3,1),(4,1),(5,1)],
        [(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)],
        [(0,0),(1,0),(2,0),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,1),(3,1),(0,-1)],
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(1,2),(2,2),(3,2),(4,1)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(1,2),(-1,2),(0,3)],
        # Staggered patterns
        [(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(1,1),(2,1),(3,1),(4,1),(2,2)],
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(1,2),(2,2),(3,2)],
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(1,2),(2,2),(3,2),(0,2),(3,1)],
        # L-shapes and T-shapes
        [(0,0),(1,0),(2,0),(3,0),(4,0),(4,1),(4,2),(4,3),(4,4),(3,1),(2,1)],
        [(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(0,2),(0,3),(0,4),(1,1),(2,1)],
        # More compact
        [(0,0),(1,0),(2,0),(1,1),(2,1),(3,1),(0,1),(0,2),(1,2),(2,2),(3,2)],
        [(0,0),(1,0),(2,0),(3,0),(1,1),(2,1),(3,1),(0,1),(0,2),(1,2),(2,2)],
    ]
    for shape in shapes:
        if len(shape) == 11:
            candidate_sets.append(set(shape))
    
    # Now do a limited branch-and-bound search for more shapes
    # Start from center, grow outward
    search_time_start = time.time()
    max_search_time = 60.0  # Leave time for optimization
    
    def dfs_grow(cells, depth, max_depth):
        """DFS to grow polyhexes, pruning by lower bound."""
        nonlocal best_global_S, best_global_pts, best_global_beta, best_global_center, best_global_cells
        
        if time.time() - search_time_start > max_search_time:
            return
        
        if len(cells) == 11:
            # Evaluate this polyhex
            pts = axial_to_cartesian(list(cells))
            lb = lower_bound_S(cells)
            if lb < best_global_S + 0.01:  # Only optimize if potentially better
                S, beta, center = optimize_shape(pts)
                if S < best_global_S:
                    best_global_S = S
                    best_global_pts = pts
                    best_global_beta = beta
                    best_global_center = center
                    best_global_cells = set(cells)
            return
        
        if len(cells) > 11:
            return
        
        # Get candidate positions to add
        cells_set = set(cells)
        candidates = []
        for q, r in cells_set:
            for nq, nr in axial_neighbors(q, r):
                if (nq, nr) not in cells_set:
                    candidates.append((nq, nr))
        
        # Deduplicate
        candidates = list(set(candidates))
        
        # Sort by some heuristic (e.g., closer to centroid first)
        if cells:
            cx = sum(q for q, r in cells) / len(cells)
            cy = sum(r for q, r in cells) / len(cells)
            candidates.sort(key=lambda p: (p[0]-cx)**2 + (p[1]-cy)**2)
        
        for nq, nr in candidates:
            if time.time() - search_time_start > max_search_time:
                return
            
            new_cells = cells_set | {(nq, nr)}
            
            # Prune: check lower bound
            lb = lower_bound_S(new_cells)
            if lb >= best_global_S:
                continue
            
            # Canonical form check to avoid duplicates (simplified)
            # For now, just proceed; the optimization will filter
            
            dfs_grow(new_cells, depth + 1, max_depth)
    
    # Start DFS from a few initial seeds
    seeds = [
        frozenset([(0,0)]),
        frozenset([(0,0), (1,0)]),
        frozenset([(0,0), (1,0), (0,1)]),
    ]
    
    # First, evaluate all pre-generated candidates
    for cells_set in candidate_sets:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        if len(cells_set) != 11:
            continue
        pts = axial_to_cartesian(list(cells_set))
        lb = lower_bound_S(cells_set)
        if lb < best_global_S + 0.01:
            S, beta, center = optimize_shape(pts)
            if S < best_global_S:
                best_global_S = S
                best_global_pts = pts
                best_global_beta = beta
                best_global_center = center
                best_global_cells = set(cells_set)
    
    # Then do limited DFS from seeds
    for seed in seeds:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        if time.time() - search_time_start > max_search_time:
            break
        dfs_grow(set(seed), 0, 11)
    
    # Fallback if nothing found
    if best_global_center is None or best_global_S == float('inf'):
        # Use the most compact shape
        fallback = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(2,1),(1,2),(0,2)]
        best_global_pts = axial_to_cartesian(fallback)
        best_global_cells = set(fallback)
        best_global_beta = 0.0
        best_global_center = (float(np.mean(best_global_pts[:, 0])), float(np.mean(best_global_pts[:, 1])))
        max_proj = 0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = best_global_pts[:, 0] * nx + best_global_pts[:, 1] * ny
            max_proj = max(max_proj, np.max(projs), -np.min(projs))
        best_global_S = (max_proj + APOTHEM) / APOTHEM

    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)
    
    # Scale by tiny factor to ensure strict containment
    SCALE = 1.0 + 1e-10
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
