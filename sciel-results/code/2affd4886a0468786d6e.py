import numpy as np
import time
from scipy.optimize import differential_evolution, minimize, minimize_scalar

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    n = 8
    # Precompute pair indices for 28 unordered pairs
    pair_indices = np.array([(i, j) for i in range(n) for j in range(i + 1, n)])
    idx_i = pair_indices[:, 0]
    idx_j = pair_indices[:, 1]

    def calculate_score(points):
        """
        points: 8x2 array
        Returns: log of the scaled product of distances.
        """
        diff = points[idx_i, :] - points[idx_j, :]
        dists = np.sqrt(np.sum(diff**2, axis=1))
        if np.any(dists < 1e-10):
            return -np.inf
        D = np.max(dists)
        if D < 1e-10:
            return -np.inf
        sum_log_d = np.sum(np.log(dists))
        log_scaled_sum = sum_log_d + 28.0 * np.log(2.0) - 28.0 * np.log(D)
        log_score = 2.0 * log_scaled_sum - 8.0 * np.log(8.0)
        return log_score

    def build_concentric_squares(R1, R2, theta):
        """
        Build 8 points from two concentric squares.
        Square 1: centered at origin, radius R1, axis-aligned.
        Square 2: centered at origin, radius R2, rotated by theta.
        """
        angles1 = np.array([0.0, np.pi / 2, np.pi, 3.0 * np.pi / 2])
        p1 = np.column_stack([R1 * np.cos(angles1), R1 * np.sin(angles1)])

        angles2 = angles1 + theta
        p2 = np.column_stack([R2 * np.cos(angles2), R2 * np.sin(angles2)])

        return np.vstack([p1, p2])

    best_points = None
    best_score = -np.inf

    # --- Candidate 0: Regular octagon ---
    angles_oct = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    pts_oct = np.column_stack([np.cos(angles_oct), np.sin(angles_oct)])
    s_oct = calculate_score(pts_oct)
    if s_oct > best_score:
        best_score = s_oct
        best_points = pts_oct.copy()

    # --- Phase 1: 1D Optimization on R1=R2 slice (D4 symmetric orbit) ---
    # In this slice, points are (x,y), (x,-y)... with R1=R2=R.
    # theta determines the relative rotation.
    # For the D4 orbit of (x,y), the two squares are rotated by pi/2 - 2*atan(y/x).
    # But we can just optimize theta and R.
    # Actually, if R1=R2, the configuration is determined by theta.
    # Let's optimize theta for a fixed R (scale doesn't matter).
    
    def score_1d_theta(theta):
        if time.time() > deadline:
            return 1e10
        R = 1.0 # Scale doesn't matter
        pts = build_concentric_squares(R, R, theta)
        s = calculate_score(pts)
        if np.isinf(s) or np.isnan(s):
            return 1e10
        return -s

    # Search theta in [0, pi/4] due to symmetry
    # Symmetry: theta and pi/2 - theta are equivalent?
    # Square 1 at 0, Square 2 at theta.
    # If we rotate whole config, it's same.
    # If we swap squares, theta -> -theta.
    # Optimal theta is likely around pi/8 (22.5 deg) for octagon (R1=R2, theta=pi/4? No, octagon is theta=pi/4? No.)
    # Regular octagon: vertices at 0, pi/4, pi/2...
    # Sq 1: 0, pi/2, pi, 3pi/2.
    # Sq 2: pi/4, 3pi/4, 5pi/4, 7pi/4.
    # So theta = pi/4.
    
    if time.time() < deadline - 30.0:
        try:
            res_1d = minimize_scalar(score_1d_theta, bounds=(0.0, np.pi/4), method='bounded', options={'xatol': 1e-12})
            if res_1d.fun < 1e9:
                theta_opt = res_1d.x
                pts_cand = build_concentric_squares(1.0, 1.0, theta_opt)
                s_cand = calculate_score(pts_cand)
                if s_cand > best_score:
                    best_score = s_cand
                    best_points = pts_cand.copy()
        except Exception:
            pass

    # --- Phase 2: 3D Differential Evolution on (R1, R2, theta) ---
    # This allows R1 != R2, breaking the single-orbit D4 symmetry.
    
    def objective_3d(x):
        if time.time() > deadline:
            return 1e10
        R1, R2, theta = x
        if R1 < 0.05 or R2 < 0.05:
            return 1e10
        pts = build_concentric_squares(R1, R2, theta)
        s = calculate_score(pts)
        if np.isinf(s) or np.isnan(s):
            return 1e10
        return -s

    bounds_3d = [
        (0.1, 2.0),      # R1
        (0.1, 2.0),      # R2
        (0.0, np.pi / 4) # theta
    ]

    candidates = []

    if time.time() < deadline - 40.0:
        try:
            res_de = differential_evolution(
                objective_3d,
                bounds_3d,
                maxiter=500,
                popsize=40,
                tol=1e-12,
                seed=42,
                init='sobol',
                workers=1,
                polish=True
            )
            if res_de.fun < 1e9:
                x_opt = res_de.x
                pts_cand = build_concentric_squares(*x_opt)
                s_cand = calculate_score(pts_cand)
                candidates.append((s_cand, pts_cand))
                if s_cand > best_score:
                    best_score = s_cand
                    best_points = pts_cand.copy()
        except Exception:
            pass

    # --- Phase 3: Local polish on 16D from best candidate ---
    def obj_16(y):
        if time.time() > deadline:
            return 1e10
        pts = y.reshape(8, 2)
        s = calculate_score(pts)
        if np.isinf(s) or np.isnan(s):
            return 1e10
        return -s

    if best_points is not None and time.time() < deadline - 10.0:
        try:
            # Start from the best symmetric candidate
            res_nm = minimize(
                obj_16,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            s_nm = -res_nm.fun
            if s_nm > best_score and not np.isinf(s_nm) and not np.isnan(s_nm):
                best_score = s_nm
                best_points = res_nm.x.reshape(8, 2).copy()
        except Exception:
            pass

    # Final check
    if best_points is None:
        angles_oct = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        best_points = np.column_stack([np.cos(angles_oct), np.sin(angles_oct)])

    # Verify distinctness
    diffs = best_points[:, np.newaxis, :] - best_points[:, :, np.newaxis]
    d = np.sqrt(np.sum(diffs**2, axis=2))
    np.fill_diagonal(d, 1.0)
    if np.min(d) < 1e-6:
        angles_oct = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        best_points = np.column_stack([np.cos(angles_oct), np.sin(angles_oct)])

    # Scale to diameter 2
    D = np.max(d)
    if D > 0:
        best_points = best_points * (2.0 / D)

    return [[float(p[0]), float(p[1])] for p in best_points]
