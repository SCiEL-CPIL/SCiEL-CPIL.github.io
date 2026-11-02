import numpy as np
from scipy.optimize import differential_evolution, minimize
import time
import itertools

N = 11
H = np.sqrt(3) / 2
SQRT3 = np.sqrt(3)

def construct():
    start_time = time.time()
    timeout = 140.0  # Safety margin for 150s limit
    
    # Precompute indices for combinations
    indices = list(itertools.combinations(range(N), 3))
    
    def check_validity(points):
        x = points[:, 0]
        y = points[:, 1]
        if np.any(y < -1e-9):
            return False
        if np.any(y > SQRT3 * x + 1e-9):
            return False
        if np.any(y > -SQRT3 * x + SQRT3 + 1e-9):
            return False
        return True

    def min_area(points):
        x = points[:, 0]
        y = points[:, 1]
        min_area = float('inf')
        for i, j, k in indices:
            area = 0.5 * abs(x[i] * (y[j] - y[k]) + x[j] * (y[k] - y[i]) + x[k] * (y[i] - y[j]))
            if area < min_area:
                min_area = area
        return min_area

    def _random_valid_points(n):
        points = []
        for _ in range(n):
            while True:
                u, v = np.random.rand(2)
                if u + v < 1:
                    x = v * 1.0 + (1 - u - v) * 0.5
                    y = (1 - u - v) * H
                    points.append([x, y])
                    break
        return np.array(points)

    # ===== Symmetry-based parameterization =====
    # The equilateral triangle has 3-fold rotational symmetry around its centroid.
    # Centroid: (0.5, H/3)
    CX, CY = 0.5, H / 3.0

    def symmetry_points(params):
        """
        params: array of shape (3, 3) where each row is [r, theta, w]
        r: radial distance from centroid
        theta: base angle
        w: weight/breaking factor (0 = fully symmetric, 1 = max breaking)
        
        For each of the 3 symmetry copies, we place points.
        Point 0 is at centroid.
        Points 1-3: first ring (3 points, 120 deg apart)
        Points 4-6: second ring
        Points 7-9: third ring
        Point 10: at centroid (or offset)
        
        Actually let's do:
        - 1 point at centroid
        - 3 points in ring 1
        - 3 points in ring 2
        - 3 points in ring 3
        - 1 point at centroid offset (for 11 total)
        
        So 11 = 1 (center) + 3 + 3 + 3 + 1 (extra)
        """
        pts = np.zeros((N, 2))
        pts[0] = [CX, CY]  # centroid
        
        # 3 rings of 3 points each
        # params structure: for each ring, 3 parameters: (r, theta_offset, w)
        # w breaks symmetry: if w > 0, the 3 points in the ring are not exactly 120 deg apart
        # Instead of full symmetry, we use: base_angle + w * perturbation
        
        for ring_idx in range(3):
            r = params[ring_idx, 0]
            theta_base = params[ring_idx, 1]
            w = params[ring_idx, 2]
            
            for k in range(3):
                angle = theta_base + 2 * np.pi * k / 3.0 + w * 0.3 * np.sin(k * 2 * np.pi / 3.0)
                idx = 1 + ring_idx * 3 + k
                pts[idx, 0] = CX + r * np.cos(angle)
                pts[idx, 1] = CY + r * np.sin(angle)
        
        # Point 10: an extra point, parameterized by params[3, :] if we extend
        # Let's add 1 more parameter set for the 11th point
        r10 = params[3, 0]
        theta10 = params[3, 1]
        pts[10, 0] = CX + r10 * np.cos(theta10)
        pts[10, 1] = CY + r10 * np.sin(theta10)
        
        return pts

    def project_to_triangle(points):
        """Project points to be inside the triangle (hard constraint enforcement)."""
        pts = points.copy()
        x = pts[:, 0]
        y = pts[:, 1]
        
        # For each point, check and clamp
        for i in range(N):
            px, py = pts[i]
            # Check against three edges
            # Edge 1: y >= 0
            if py < 0:
                py = 0.0
            # Edge 2: y <= SQRT3 * x
            if py > SQRT3 * px:
                # Project onto line y = SQRT3 * x
                # Closest point on line to (px, py)
                # Line: SQRT3*x - y = 0
                # Normal: (SQRT3, -1), norm = 2
                # Distance = (SQRT3*px - py) / 2
                # Projection: (px, py) - (SQRT3*px - py)/4 * (SQRT3, -1)
                d = (SQRT3 * px - py) / 4.0
                px = px - d * SQRT3
                py = py + d
            # Edge 3: y <= -SQRT3 * x + SQRT3
            if py > -SQRT3 * px + SQRT3:
                # Line: SQRT3*x + y - SQRT3 = 0
                # Normal: (SQRT3, 1), norm = 2
                d = (SQRT3 * px + py - SQRT3) / 4.0
                px = px - d * SQRT3
                py = py - d
            
            # Clamp to reasonable bounds
            px = np.clip(px, -0.1, 1.1)
            py = np.clip(py, -0.1, H + 0.1)
            pts[i] = [px, py]
        
        return pts

    def objective_sym(params):
        pts = symmetry_points(params)
        pts = project_to_triangle(pts)
        
        if not check_validity(pts):
            return 1e10
        
        ma = min_area(pts)
        if ma < 1e-12:
            return 1e10
        return -ma

    # Bounds for symmetry parameters
    # r: 0 to H (max distance from centroid)
    # theta: 0 to 2*pi
    # w: -1 to 1
    bounds_sym = [
        (0.0, H), (0.0, 2 * np.pi), (-1.0, 1.0),  # ring 0
        (0.0, H), (0.0, 2 * np.pi), (-1.0, 1.0),  # ring 1
        (0.0, H), (0.0, 2 * np.pi), (-1.0, 1.0),  # ring 2
        (0.0, H), (0.0, 2 * np.pi), (0.0, 0.0),   # point 10 (r, theta, unused)
    ]

    # Initial symmetric guess
    init_sym = np.array([
        [H / 6, 0.0, 0.0],
        [H / 3, np.pi / 6, 0.0],
        [H / 2, 0.0, 0.0],
        [H / 4, np.pi / 3, 0.0],
    ])

    # ===== Phase 1: Optimize symmetry parameters =====
    elapsed = time.time() - start_time
    if elapsed < timeout - 60:
        try:
            result_sym = differential_evolution(
                objective_sym,
                bounds_sym,
                maxiter=300,
                popsize=30,
                tol=1e-8,
                seed=42,
                mutation=(0.5, 1.5),
                recombination=0.9,
                polish=False,
                init='latinhypercube',
            )
            best_params_sym = result_sym.x
            best_pts_sym = symmetry_points(best_params_sym)
            best_pts_sym = project_to_triangle(best_pts_sym)
            if not check_validity(best_pts_sym):
                best_pts_sym = _random_valid_points(N)
            best_score = min_area(best_pts_sym)
        except Exception:
            best_pts_sym = _random_valid_points(N)
            best_score = min_area(best_pts_sym)
    else:
        best_pts_sym = _random_valid_points(N)
        best_score = min_area(best_pts_sym)

    best_points = best_pts_sym.copy()

    # ===== Phase 2: Full coordinate refinement =====
    elapsed = time.time() - start_time
    if elapsed > timeout - 30:
        if not check_validity(best_points):
            return _random_valid_points(N)
        return best_points

    def objective_softmin(params, T=0.01):
        points = params.reshape(N, 2)
        x = points[:, 0]
        y = points[:, 1]
        
        pen = 0.0
        if np.any(y < 0):
            pen += np.sum(np.maximum(0, -y) ** 2)
        if np.any(y > SQRT3 * x):
            pen += np.sum(np.maximum(0, y - SQRT3 * x) ** 2)
        if np.any(y > -SQRT3 * x + SQRT3):
            pen += np.sum(np.maximum(0, y - (-SQRT3 * x + SQRT3)) ** 2)
        
        if pen > 1e-6:
            return 1e10 + pen
        
        areas = np.zeros(len(indices))
        for idx, (i, j, k) in enumerate(indices):
            areas[idx] = 0.5 * abs(x[i] * (y[j] - y[k]) + x[j] * (y[k] - y[i]) + x[k] * (y[i] - y[j]))
        
        areas_clipped = np.maximum(areas, 1e-12)
        log_sum_exp = np.log(np.sum(np.exp(-areas_clipped / T)))
        soft_min = -T * log_sum_exp
        
        return -soft_min

    bounds_full = [(0.0, 1.0), (0.0, H)] * N

    # Polish the symmetric solution
    try:
        res_lbfgs = minimize(
            objective_softmin,
            best_points.flatten(),
            method='L-BFGS-B',
            bounds=bounds_full,
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10},
        )
        if res_lbfgs.fun < 1e9:
            new_points = res_lbfgs.x.reshape(N, 2)
            if check_validity(new_points):
                new_score = min_area(new_points)
                if new_score > best_score:
                    best_points = new_points
                    best_score = new_score
    except Exception:
        pass

    elapsed = time.time() - start_time
    if elapsed > timeout - 15:
        if not check_validity(best_points):
            return _random_valid_points(N)
        return best_points

    # ===== Phase 3: Multi-start with soft-min polish =====
    num_restarts = 3
    for i in range(num_restarts):
        if time.time() - start_time > timeout - 5:
            break
        
        pts = _random_valid_points(N)
        
        try:
            res_r = minimize(
                objective_softmin,
                pts.flatten(),
                method='L-BFGS-B',
                bounds=bounds_full,
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8},
            )
            if res_r.fun < 1e9:
                if check_validity(res_r.x.reshape(N, 2)):
                    score_r = min_area(res_r.x.reshape(N, 2))
                    if score_r > best_score:
                        best_points = res_r.x.reshape(N, 2)
                        best_score = score_r
        except Exception:
            pass

    if not check_validity(best_points):
        return _random_valid_points(N)

    return best_points
