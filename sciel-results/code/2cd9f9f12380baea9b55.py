import math
import numpy as np
import time
from scipy.optimize import minimize
import itertools

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0
EPS = 1e-9

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

    # Generate candidate configurations
    # We will generate a larger set of candidates by combining base shapes
    # and random perturbations, ensuring they are valid polyhexes (connected, no overlaps)
    
    # Helper to check if a set of axial coordinates is a valid polyhex (connected, no dups)
    def is_valid_polyhex(coords):
        if len(coords) != len(set(coords)):
            return False
        coord_set = set(coords)
        if not coord_set:
            return True
        start = next(iter(coord_set))
        visited = {start}
        stack = [start]
        neighbors = [(1,0),(-1,0),(0,1),(0,-1),(1,-1),(-1,1)]
        while stack:
            q, r = stack.pop()
            for dq, dr in neighbors:
                nq, nr = q+dq, r+dr
                if (nq, nr) in coord_set and (nq, nr) not in visited:
                    visited.add((nq, nr))
                    stack.append((nq, nr))
        return len(visited) == len(coord_set)

    # Generate candidates
    candidates = []
    
    # 1. The "Star" or "Flower" shape (center + 6 neighbors) + 4 more
    base_star = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1)]
    # Add 4 more to the star
    # Common extensions:
    candidates.append(base_star + [(2,0),(-2,0),(0,2),(0,-2)]) # 3x3 block essentially? No, 3x3 is 9. Star is 7.
    # 3x3 block:
    candidates.append([(0,0),(1,0),(2,0),(-1,0),(0,1),(1,1),(2,1),(-1,1),(0,2),(1,2),(2,2)])
    
    # 2. Random search for compact polyhexes
    # We can do a simple random walk from a center to build connected shapes
    import random
    random.seed(42)
    
    def generate_random_polyhex(n):
        coords = set([(0,0)])
        current = (0,0)
        neighbors = [(1,0),(-1,0),(0,1),(0,-1),(1,-1),(-1,1)]
        while len(coords) < n:
            # Pick a random existing point to expand from
            parent = random.choice(list(coords))
            # Pick a random neighbor
            dq, dr = random.choice(neighbors)
            new_point = (parent[0]+dq, parent[1]+dr)
            if new_point not in coords:
                coords.add(new_point)
                current = new_point
        return list(coords)

    # Generate 100 random candidates
    for _ in range(100):
        cand = generate_random_polyhex(N)
        if is_valid_polyhex(cand):
            # Center it roughly
            q_min = min(q for q,r in cand)
            r_min = min(r for q,r in cand)
            # Shift to center
            # For axial, centering is tricky, but we can just use the mean
            q_avg = sum(q for q,r in cand) / N
            r_avg = sum(r for q,r in cand) / N
            # Shift so mean is 0
            cand = [(int(round(q - q_avg)), int(round(r - r_avg))) for q,r in cand]
            # Re-check validity after rounding (might break if not integer)
            # Actually, axial coords are integers. The mean might not be integer.
            # Better: just use the generated coords as is, the optimization will find the center.
            # But for deduplication, let's normalize by min q, min r
            q_min = min(q for q,r in cand)
            r_min = min(r for q,r in cand)
            norm_cand = tuple(sorted((q-q_min, r-r_min) for q,r in cand))
            if norm_cand not in [tuple(sorted(c)) for c in candidates]:
                candidates.append(list(norm_cand))

    # Add specific known good shapes
    # 4x3 block minus 1
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)])
    # 5x2 block
    candidates.append([(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(1,1),(2,1),(3,1),(4,1),(5,1)])
    # 3x4 block
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3)])
    # 2x6 block
    candidates.append([(0,0),(1,0),(2,0),(3,0),(4,0),(5,0),(0,1),(1,1),(2,1),(3,1),(4,1)])

    # Remove duplicates
    unique_candidates = []
    seen = set()
    for c in candidates:
        key = tuple(sorted(c))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)
    candidates = unique_candidates

    def calc_S_fast(pts, betas):
        """
        Vectorized calculation of S for multiple betas.
        Returns array of S values.
        """
        n_pts = len(pts)
        n_betas = len(betas)
        
        # For each beta, we have 6 normals.
        # We need to find the min enclosing hexagon.
        # This is equivalent to finding the max projection of points onto the 6 normals,
        # but we also need to optimize the center.
        # For a fixed beta, the optimal center is the midpoint of the bounding box in the projection space?
        # No, for a regular hexagon, the optimal center is the centroid of the "tightest" fit.
        # Actually, for a set of points, the minimum enclosing regular hexagon with fixed orientation
        # has its center at the intersection of the midlines of the 3 pairs of parallel supporting lines.
        # This can be solved by LP, but we can approximate it or use a fast heuristic.
        # Given the time constraint, let's stick to the LP for the final polish,
        # but for the grid search, we can use a simpler bound:
        # The side length S is determined by D = max |proj|.
        # The center can be chosen to minimize D.
        # For a fixed orientation, the problem is convex.
        # Let's just use the centroid as a quick estimate for the center in the fast pass,
        # then refine with LP for the best few.
        
        # Actually, a better fast approximation:
        # For each beta, compute the projections of all points onto the 6 normals.
        # The minimal D for a centered hexagon is max(proj_max - proj_min) / 2?
        # No, D is the distance from center to side.
        # If we center at the origin, D = max(abs(proj)).
        # We can shift the center to minimize this.
        # The optimal center is the solution to the LP.
        # To save time, we will compute the "bounding box" in the projection space.
        # Let P be the matrix of projections (n_pts x 6).
        # For each direction k, the width is max(P[:,k]) - min(P[:,k]).
        # The minimal D is at least max(widths)/2.
        # This is a lower bound. The actual D might be larger due to coupling.
        # However, for compact shapes, this is a good estimator.
        
        # Let's just do the LP for the best 5 candidates from a coarse grid.
        
        # For the fast pass, let's just evaluate the "width" metric.
        # Width metric = max_k (max_i proj_ik - min_i proj_ik) / 2
        # This is fast to compute vectorized.
        
        # Angles for each beta
        # beta is a scalar array (n_betas,)
        # angles = beta[:, None] + pi/6 + np.arange(6)[None, :] * pi/3
        # This gives (n_betas, 6)
        
        # We need projections for all points and all angles.
        # pts is (n_pts, 2)
        # We can compute dot products.
        
        # This might be too heavy if n_betas is large.
        # Let's stick to the LP for a smaller grid, or use the width metric.
        
        # Let's use the width metric for the coarse search.
        # It's an admissible lower bound on D?
        # D >= width_k / 2 for all k.
        # So D >= max_k (width_k / 2).
        # S = (D + APOTHEM) / APOTHEM.
        # So S_lower = (max_k(width_k/2) + APOTHEM) / APOTHEM.
        
        # This is very fast.
        
        # Compute normals for all betas
        # betas: (n_betas,)
        # angles: (n_betas, 6)
        angles = betas[:, None] + math.pi / 6.0 + np.arange(6)[None, :] * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)]) # (n_betas, 6, 2)
        
        # Project points
        # pts: (n_pts, 2)
        # We want (n_betas, n_pts, 6)
        # projs = np.einsum('ij,bkj->bki', pts, normals)
        
        projs = np.einsum('ij,bkj->bki', pts, normals)
        
        # Widths
        max_projs = np.max(projs, axis=1) # (n_betas, 6)
        min_projs = np.min(projs, axis=1) # (n_betas, 6)
        widths = max_projs - min_projs # (n_betas, 6)
        
        max_widths = np.max(widths, axis=1) # (n_betas,)
        
        D_lower = max_widths / 2.0
        S_lower = (D_lower + APOTHEM) / APOTHEM
        
        return S_lower

    def optimize_shape(pts):
        """
        1. Coarse grid search using fast width metric.
        2. Fine grid search around best using LP.
        3. Polish with minimize.
        """
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, None

        # Stage 1: Coarse grid with fast metric
        n_grid = 100
        betas = np.linspace(0, math.pi / 6, n_grid)
        S_lower = calc_S_fast(pts, betas)
        
        best_idx = np.argmin(S_lower)
        best_S_lower = S_lower[best_idx]
        best_beta = betas[best_idx]
        
        # Stage 2: Fine grid with LP around best_beta
        # We need to find the exact S for the best beta.
        # Let's do a fine grid of 10 points around best_beta.
        window = (math.pi / 6) / (n_grid - 1)
        lo = max(0.0, best_beta - 2 * window)
        hi = min(math.pi / 6, best_beta + 2 * window)
        
        fine_betas = np.linspace(lo, hi, 20)
        
        best_S = float('inf')
        best_center = None
        
        for b in fine_betas:
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            S, center, _ = calc_S_for_beta(pts, b)
            if S < best_S:
                best_S = S
                best_center = center
                best_beta = b

        if best_center is None:
            return float('inf'), 0.0, None

        # Stage 3: Polish
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                def exact_obj(params):
                    cx, cy, beta = params
                    beta = ((beta % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                    return exact_S(pts, cx, cy, beta)

                res = minimize(exact_obj, [best_center[0], best_center[1], best_beta], method='Nelder-Mead',
                             options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-12})
                if res.fun < best_S:
                    best_S = res.fun
                    best_center = (res.x[0], res.x[1])
                    best_beta = ((res.x[2] % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
        except Exception:
            pass

        return best_S, best_beta, best_center

    def calc_S_for_beta(pts, beta):
        """LP for minimum enclosing hexagon with given rotation."""
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
            from scipy.optimize import linprog
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
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        pts_centered = pts - np.array([cx, cy])
        projs = pts_centered @ normals.T
        D = np.max(np.abs(projs))
        return (D + APOTHEM) / APOTHEM

    # Evaluate all candidates
    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

    for idx, axial in enumerate(candidates):
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

    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)

    # Final: compute side length with tiny safety scale
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
