import math
import numpy as np
import time
from scipy.optimize import linprog, minimize, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0
EPS = 1e-9

def force_directed_relax(pts, n_iter=200, repulse_strength=1.0, attract_strength=0.01):
    """Simple force-directed relaxation to find more compact configurations."""
    pts = pts.astype(float).copy()
    n = len(pts)
    min_dist = 2 * APOTHEM  # sqrt(3)
    
    for it in range(n_iter):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        forces = np.zeros_like(pts)
        for i in range(n):
            for j in range(i+1, n):
                diff = pts[i] - pts[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    diff = np.array([0.1, 0.1])
                    dist = np.linalg.norm(diff)
                # Repulsion if too close
                if dist < min_dist:
                    force = repulse_strength * (min_dist - dist) / dist
                    forces[i] += force * diff
                    forces[j] -= force * diff
            # Attraction toward centroid
            centroid = np.mean(pts, axis=0)
            forces[i] += attract_strength * (centroid - pts[i])
        pts += 0.1 * forces
    return pts

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
    candidates = []

    # Standard hexagonal cluster
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(1,1),(-1,2),(-2,1),(2,-1)])
    # Cross
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)])
    # 3x3 block + 2
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,1),(3,1)])
    # 4x3 minus corners
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(1,2),(2,2),(3,2)])
    # Staircase
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)])
    # Compact 3x3+2
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3)])
    # Star
    candidates.append([(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(2,0),(-2,0),(0,2),(0,-2)])
    # Elongated
    candidates.append([(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(1,1),(2,1),(3,1),(4,1),(5,1)])
    # Asymmetric star
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(-2,2),(-1,3),(1,-2)])
    # Compact with top
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3)])
    # Wide base
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),(1,3)])
    # Tall narrow
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(0,2),(1,2),(0,3),(1,3),(0,4)])
    # Star with asymmetric outer
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(1,2)])
    # Star with diagonal
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,-1)])
    # Dense cluster
    candidates.append([(0,0),(1,0),(2,0),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,1),(3,1),(0,-1)])
    # Zigzag
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(3,2)])
    # Star with top cluster
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(1,2),(-1,2),(0,3)])
    # Trapezoid
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(1,2),(2,2),(3,2)])
    # Wide top
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(3,3)])
    # Star with inner diagonal
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,1)])
    # 4x3 variant
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)])
    # Wide with top
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3)])
    # Symmetric star
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)])
    # Very tall
    candidates.append([(0,0),(1,0),(0,1),(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(0,4)])
    # 3x3+row
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)])
    # Hexagonal ring (7 inner + 4 outer)
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)])
    # Offset grid
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2)])
    # Dense diamond
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(3,2),(1,3),(2,3),(0,2)])
    # Dense 4-wide
    candidates.append([(0,0),(1,0),(2,0),(3,0),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),(1,3),(2,3)])
    # Star with asymmetric bottom
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(1,-2)])
    # Tall with top row
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(0,3),(1,3),(2,3),(1,4),(2,4)])
    # Additional
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3),(2,3)])
    candidates.append([(0,0),(1,0),(2,0),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3),(0,-1),(3,1)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,-1)])
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(1,4),(2,4)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3)])

    def calc_S_for_beta(pts, beta):
        """Vectorized LP for minimum enclosing hexagon with given rotation."""
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), None, None

        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])

        n_pts = len(pts)
        # For each normal and point: two constraints
        # Total: 12 * n_pts
        n_constraints = 12 * n_pts
        A_ub = np.zeros((n_constraints, 3))
        b_ub = np.zeros(n_constraints)

        # Vectorize: for each normal k, project all points
        for k in range(6):
            nx, ny = normals[k]
            projs = pts[:, 0] * nx + pts[:, 1] * ny  # (n_pts,)
            start = 2 * k * n_pts
            # Constraint 1: -nx*cx - ny*cy - D <= -proj_i
            A_ub[start:start+n_pts, 0] = -nx
            A_ub[start:start+n_pts, 1] = -ny
            A_ub[start:start+n_pts, 2] = -1.0
            b_ub[start:start+n_pts] = -projs
            # Constraint 2: nx*cx + ny*cy - D <= proj_i
            A_ub[start+n_pts:start+2*n_pts, 0] = nx
            A_ub[start+n_pts:start+2*n_pts, 1] = ny
            A_ub[start+n_pts:start+2*n_pts, 2] = -1.0
            b_ub[start+n_pts:start+2*n_pts] = projs

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

    def newton_polish_center(pts, beta, cx, cy, n_iter=50):
        """2D Newton polish for optimal center at fixed rotation."""
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        
        for it in range(n_iter):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            projs = (pts - np.array([cx, cy])) @ normals.T  # (n_pts, 6)
            D = np.max(np.abs(projs))
            
            # Find active constraints
            active_mask = np.abs(np.abs(projs) - D) < 1e-10
            if not np.any(active_mask):
                break
            
            # Solve for center that minimizes max projection
            # This is equivalent to finding the Chebyshev center in 2D
            # Use simple gradient step
            grad_x = 0.0
            grad_y = 0.0
            count = 0
            for i in range(len(pts)):
                for k in range(6):
                    if active_mask[i, k]:
                        sign = 1.0 if projs[i, k] > 0 else -1.0
                        grad_x += sign * normals[k, 0]
                        grad_y += sign * normals[k, 1]
                        count += 1
                    if active_mask[i, k]:  # Also check negative side
                        sign = -1.0 if projs[i, k] > 0 else 1.0
                        grad_x += sign * normals[k, 0]
                        grad_y += sign * normals[k, 1]
                        count += 1
            
            if count == 0:
                break
            
            grad_x /= count
            grad_y /= count
            
            # Damping factor
            lr = 0.5 / (1.0 + it * 0.1)
            cx -= lr * grad_x
            cy -= lr * grad_y
        
        return cx, cy

    def exact_S(pts, cx, cy, beta):
        """Exact computation of S for given center and rotation."""
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        pts_centered = pts - np.array([cx, cy])
        projs = pts_centered @ normals.T  # (n_pts, 6)
        D = np.max(np.abs(projs))
        return (D + APOTHEM) / APOTHEM

    def smooth_calc_S(pts, cx, cy, beta, temperature=0.05):
        """Smooth surrogate using log-sum-exp."""
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        pts_centered = pts - np.array([cx, cy])
        projs = pts_centered @ normals.T  # (n_pts, 6)
        all_projs = np.concatenate([projs.ravel(), (-projs).ravel()])
        max_val = np.max(all_projs)
        if max_val > 200:
            return float('inf')
        shifted = (all_projs - max_val) / temperature
        shifted = np.clip(shifted, -50, 50)
        log_sum = np.log(np.sum(np.exp(shifted))) + max_val / temperature
        D = temperature * log_sum
        return (D + APOTHEM) / APOTHEM

    def optimize_shape(pts):
        """Multi-stage optimization for a fixed point configuration."""
        best_S = float('inf')
        best_beta = 0.0
        best_center = None

        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, None

        # Stage 1: Grid search over beta in [0, pi/6]
        n_grid = 49
        for i in range(n_grid):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            beta = i * (math.pi / 6) / (n_grid - 1)
            S, center, D = calc_S_for_beta(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = center

        if best_center is None:
            return float('inf'), 0.0, None

        # Stage 2: Scalar refinement of beta
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                res = minimize_scalar(lambda b: calc_S_for_beta(pts, b)[0],
                                     bounds=(0, math.pi / 6), method='bounded',
                                     options={'maxiter': 300, 'xatol': 1e-11})
                if res.fun < best_S:
                    best_S = res.fun
                    best_beta = res.x
                    _, best_center, _ = calc_S_for_beta(pts, best_beta)
        except Exception:
            pass

        # Stage 3: Joint (cx, cy, beta) optimization with smooth surrogate
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                temperatures = [0.1, 0.05, 0.02, 0.01]
                for temp in temperatures:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break

                    def smooth_obj(params, t=temp):
                        cx, cy, beta = params
                        beta = ((beta % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                        return smooth_calc_S(pts, cx, cy, beta, temperature=t)

                    start_betas = [best_beta, best_beta + 0.04, best_beta - 0.04, 0.0, math.pi / 12]
                    for sb in start_betas:
                        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                            break
                        init = [best_center[0], best_center[1], sb]
                        try:
                            res = minimize(smooth_obj, init, method='Nelder-Mead',
                                         options={'maxiter': 500, 'xatol': 1e-11, 'fatol': 1e-11})
                            if res.fun < 1e10:
                                cx_r, cy_r, beta_r = res.x
                                beta_r = ((beta_r % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                                S_exact = exact_S(pts, cx_r, cy_r, beta_r)
                                if S_exact < best_S:
                                    best_S = S_exact
                                    best_beta = beta_r
                                    best_center = (cx_r, cy_r)
                        except Exception:
                            pass
        except Exception:
            pass

        # Stage 4: Newton polish for center at optimal beta
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                cx_new, cy_new = newton_polish_center(pts, best_beta, best_center[0], best_center[1])
                S_new = exact_S(pts, cx_new, cy_new, best_beta)
                if S_new < best_S:
                    best_S = S_new
                    best_center = (cx_new, cy_new)
        except Exception:
            pass

        # Stage 5: High-precision polish with exact objective
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                def exact_obj(params):
                    cx, cy, beta = params
                    beta = ((beta % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                    return exact_S(pts, cx, cy, beta)

                polish_starts = [
                    [best_center[0], best_center[1], best_beta],
                    [best_center[0], best_center[1], best_beta + 0.01],
                    [best_center[0], best_center[1], best_beta - 0.01],
                    [best_center[0] + 0.01, best_center[1], best_beta],
                    [best_center[0] - 0.01, best_center[1], best_beta],
                    [best_center[0], best_center[1] + 0.01, best_beta],
                    [best_center[0], best_center[1] - 0.01, best_beta],
                    [best_center[0] + 0.005, best_center[1] + 0.005, best_beta],
                    [best_center[0] - 0.005, best_center[1] - 0.005, best_beta],
                ]

                for init in polish_starts:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    try:
                        res = minimize(exact_obj, init, method='Nelder-Mead',
                                     options={'maxiter': 5000, 'xatol': 1e-13, 'fatol': 1e-13})
                        if res.fun < best_S:
                            best_S = res.fun
                            best_center = (res.x[0], res.x[1])
                            best_beta = ((res.x[2] % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                    except Exception:
                        pass
        except Exception:
            pass

        return best_S, best_beta, best_center

    def optimize_positions(pts, beta, center):
        """Optimize inner hexagon positions for fixed outer rotation."""
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return pts
        
        n = len(pts)
        # Build constraint matrix for non-overlap: |p_i - p_j| >= 2*APOTHEM (face-to-face)
        # Actually for regular hexagons of side 1, minimum center distance is 2*APOTHEM = sqrt(3)
        min_dist = math.sqrt(3)
        
        def pos_obj(params):
            """Objective: maximize minimum distance to outer boundary (i.e., minimize S)."""
            positions = params.reshape(n, 2)
            cx, cy = center
            angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
            normals = np.column_stack([np.cos(angles), np.sin(angles)])
            pts_centered = positions - np.array([cx, cy])
            projs = pts_centered @ normals.T  # (n, 6)
            D = np.max(np.abs(projs))
            return (D + APOTHEM) / APOTHEM
        
        def pos_constraints(params):
            """Non-overlap constraints: ||p_i - p_j||^2 >= min_dist^2"""
            positions = params.reshape(n, 2)
            conns = []
            for i in range(n):
                for j in range(i+1, n):
                    diff = positions[i] - positions[j]
                    dist_sq = diff[0]**2 + diff[1]**2
                    # Constraint: dist_sq >= min_dist^2, i.e., min_dist^2 - dist_sq <= 0
                    conns.append(min_dist**2 - dist_sq)
            return conns
        
        # Use SLSQP with non-overlap constraints
        x0 = pts.flatten()
        try:
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                return pts
            res = minimize(pos_obj, x0, method='SLSQP', 
                         options={'maxiter': 200, 'ftol': 1e-12})
            if res.success and res.fun < float('inf'):
                new_pts = res.x.reshape(n, 2)
                # Verify no overlaps
                valid = True
                for i in range(n):
                    for j in range(i+1, n):
                        d = np.linalg.norm(new_pts[i] - new_pts[j])
                        if d < min_dist - 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    return new_pts
        except Exception:
            pass
        return pts

    # Evaluate all candidates
    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

    # Also try force-directed relaxation on some candidates
    fd_candidates = []
    for idx in [0, 2, 5, 7, 14, 22]:  # Pick diverse candidates
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        pts_base = axial_to_cartesian(candidates[idx])
        pts_fd = force_directed_relax(pts_base, n_iter=100, repulse_strength=0.5, attract_strength=0.005)
        # Verify no overlaps
        valid = True
        min_dist = 2 * APOTHEM
        for i in range(len(pts_fd)):
            for j in range(i+1, len(pts_fd)):
                if np.linalg.norm(pts_fd[i] - pts_fd[j]) < min_dist - 1e-6:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            fd_candidates.append(pts_fd)

    # Combine original and force-directed candidates
    all_candidates = [(axial_to_cartesian(axial), False) for axial in candidates] + \
                     [(pts, True) for pts in fd_candidates]

    for idx, (pts, is_fd) in enumerate(all_candidates):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        S, beta, center = optimize_shape(pts)
        
        # Try continuous position optimization from this starting point
        if S < best_global_S and time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN and not is_fd:
            new_pts = optimize_positions(pts, beta, center)
            if not np.array_equal(new_pts, pts):
                S2, beta2, center2 = optimize_shape(new_pts)
                if S2 < best_global_S:
                    S = S2
                    beta = beta2
                    center = center2
                    pts = new_pts
        
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
    SCALE = 1.0 + 1e-12
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
