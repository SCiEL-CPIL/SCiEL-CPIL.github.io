# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def moving_sofa_2d() -> np.ndarray:
    """
    Returns a path (n, 3) array for the moving sofa problem.
    """
    start_time = time.time()
    deadline = start_time + 590  # safety margin before 630s kill

    # Sofa grid points
    sofa_x = np.arange(-3, 1.5, 0.01)
    sofa_y = np.arange(-1.5, 1.5, 0.01)
    sofa_X, sofa_Y = np.meshgrid(sofa_x, sofa_y, indexing='ij')
    sofa_x_flat = sofa_X.ravel()
    sofa_y_flat = sofa_Y.ravel()
    n_sofa = len(sofa_x_flat)

    def _densify(path):
        """Add the evaluator's start/end and interpolate between consecutive rows."""
        full = np.vstack([[[0.0, 0.0, 0.0]], path, [[-2.0, 0.0, 90.0]]])
        result = [full[0].copy()]
        for i in range(len(full) - 1):
            p1, p2 = full[i], full[i + 1]
            da = abs(p2[2] - p1[2])
            m = max(1, int(np.ceil(da)))
            for k in range(1, m):
                t = k / m
                result.append(p1 * (1 - t) + p2 * t)
            result.append(p2.copy())
        return np.array(result)

    def _count_inside(path):
        """Count sofa points inside all placements in the densified path."""
        full = _densify(path)
        inside = np.ones(n_sofa, dtype=bool)
        for px, py, pa in full:
            a_rad = np.deg2rad(pa)
            c, s = np.cos(a_rad), np.sin(a_rad)
            dx = sofa_x_flat - px
            dy = sofa_y_flat - py
            u = c * dx + s * dy
            v = -s * dx + c * dy
            in1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
            in2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
            inside &= (in1 | in2)
            if not np.any(inside):
                return 0
        return int(np.sum(inside))

    def _score(path):
        return _count_inside(path) * 0.0001

    def _bezier_path(ctrl_pts, n_samples):
        """
        Generate a path from a cubic Bezier curve parameterized by control points.
        ctrl_pts: array of shape (4, 3) - control points for the Bezier curve
        n_samples: number of points to sample along the curve
        """
        t_vals = np.linspace(0, 1, n_samples + 1)
        path = np.zeros((n_samples + 1, 3))
        for i, t in enumerate(t_vals):
            # Cubic Bezier: B(t) = (1-t)^3*P0 + 3*(1-t)^2*t*P1 + 3*(1-t)*t^2*P2 + t^3*P3
            w0 = (1 - t) ** 3
            w1 = 3 * (1 - t) ** 2 * t
            w2 = 3 * (1 - t) * t ** 2
            w3 = t ** 3
            path[i] = w0 * ctrl_pts[0] + w1 * ctrl_pts[1] + w2 * ctrl_pts[2] + w3 * ctrl_pts[3]
        # Ensure start and end are exactly (0,0,0) and (-2,0,90)
        path[0] = [0.0, 0.0, 0.0]
        path[-1] = [-2.0, 0.0, 90.0]
        return path

    def _neg_score_from_params(params):
        """
        Objective function for optimization: negative score from Bezier control points.
        params: array of shape (6,) - [x1, y1, x2, y2] for middle two control points
        """
        if time.time() > deadline - 10:
            return 0.0
        # Control points: P0=(0,0,0), P1=(x1,y1,a1), P2=(x2,y2,a2), P3=(-2,0,90)
        a1 = params[4] if len(params) > 4 else 30.0
        a2 = params[5] if len(params) > 5 else 60.0
        ctrl_pts = np.array([
            [0.0, 0.0, 0.0],
            [params[0], params[1], a1],
            [params[2], params[3], a2],
            [-2.0, 0.0, 90.0]
        ])
        path = _bezier_path(ctrl_pts, 10)
        return -_score(path)

    def _neg_score_4d(params):
        """
        4-parameter version: [x1, y1, x2, y2] with fixed angles a1=30, a2=60
        """
        if time.time() > deadline - 10:
            return 0.0
        ctrl_pts = np.array([
            [0.0, 0.0, 0.0],
            [params[0], params[1], 30.0],
            [params[2], params[3], 60.0],
            [-2.0, 0.0, 90.0]
        ])
        path = _bezier_path(ctrl_pts, 10)
        return -_score(path)

    def _neg_score_6d(params):
        """
        6-parameter version: [x1, y1, x2, y2, a1, a2]
        """
        if time.time() > deadline - 10:
            return 0.0
        ctrl_pts = np.array([
            [0.0, 0.0, 0.0],
            [params[0], params[1], params[4]],
            [params[2], params[3], params[5]],
            [-2.0, 0.0, 90.0]
        ])
        path = _bezier_path(ctrl_pts, 10)
        return -_score(path)

    def _two_arc_path(c1x, c1y, c2x, c2y, n1, n2):
        """
        Two-segment path: arc1 from (0,0,0) to junction, then arc2 from junction to (-2,0,90).
        """
        # Arc1: from (0,0) with center (c1x, c1y)
        R1 = np.sqrt(c1x * c1x + c1y * c1y)
        if R1 < 1e-10:
            return None
        # Arc2: from (-2,0) with center (c2x, c2y)
        R2 = np.sqrt((c2x + 2.0) ** 2 + c2y * c2y)
        if R2 < 1e-10:
            return None

        # Find intersection of the two circles
        # Circle 1: (x - c1x)^2 + (y - c1y)^2 = R1^2
        # Circle 2: (x - c2x)^2 + (y - c2y)^2 = R2^2
        dx = c2x - c1x
        dy = c2y - c1y
        d = np.sqrt(dx * dx + dy * dy)
        if d < 1e-10 or d > R1 + R2 or d < abs(R1 - R2):
            return None

        # Distance from center1 to the line connecting the intersection points
        a = (R1 ** 2 - R2 ** 2 + d ** 2) / (2 * d)
        h_sq = R1 ** 2 - a ** 2
        if h_sq < 0:
            return None
        h = np.sqrt(h_sq)

        # Midpoint
        mx = c1x + a * dx / d
        my = c1y + a * dy / d

        # Two intersection points
        x_int1 = mx + h * dy / d
        y_int1 = my - h * dx / d
        x_int2 = mx - h * dy / d
        y_int2 = my + h * dx / d

        # Try both intersection points
        best_result = None
        best_score_val = -1

        for x_j, y_j in [(x_int1, y_int1), (x_int2, y_int2)]:
            # Check if the junction makes sense (should be between start and end in some sense)
            # Compute angles
            a0_1 = np.arctan2(0.0 - c1y, 0.0 - c1x)
            a1_1 = np.arctan2(y_j - c1y, x_j - c1x)
            a0_2 = np.arctan2(y_j - c2y, x_j - c2x)
            a1_2 = np.arctan2(0.0 - c2y, -2.0 - c2x)

            # Sweep angles
            sweep1 = (a1_1 - a0_1) % (2 * np.pi)
            if sweep1 > np.pi:
                sweep1 -= 2 * np.pi
            sweep2 = (a1_2 - a0_2) % (2 * np.pi)
            if sweep2 > np.pi:
                sweep2 -= 2 * np.pi

            # Total sweep should be close to 90 degrees
            total_sweep = abs(sweep1) + abs(sweep2)
            if abs(total_sweep - np.pi / 2) > 0.5:
                continue

            # Generate path
            path = []
            for i in range(n1 + 1):
                t = i / n1
                angle = a0_1 + t * sweep1
                x = c1x + R1 * np.cos(angle)
                y = c1y + R1 * np.sin(angle)
                a_deg = 90.0 * (i / (n1 + n2))
                path.append([x, y, a_deg])
            for i in range(1, n2 + 1):
                t = i / n2
                angle = a0_2 + t * sweep2
                x = c2x + R2 * np.cos(angle)
                y = c2y + R2 * np.sin(angle)
                a_deg = 90.0 * ((n1 + i) / (n1 + n2))
                path.append([x, y, a_deg])

            path = np.array(path, dtype=np.float64)
            s = _score(path)
            if s > best_score_val:
                best_score_val = s
                best_result = path

        return best_result

    # --- Phase 1: Bezier optimization with L-BFGS-B ---
    best_score = -1
    best_path = None

    # Initial guesses for control points
    initial_guesses = [
        [-0.5, 0.5, -1.5, 0.5],
        [-1.0, 1.0, -1.0, 1.0],
        [-0.5, 1.0, -1.5, 1.0],
        [-1.0, 0.5, -1.0, 0.5],
        [-0.7, 0.7, -1.3, 0.7],
        [-1.2, 0.8, -0.8, 0.8],
    ]

    for init in initial_guesses:
        if time.time() > deadline - 120:
            break
        try:
            result = minimize(
                _neg_score_4d,
                init,
                method='L-BFGS-B',
                options={'maxiter': 50, 'maxfun': 200}
            )
            if result.success or result.fun < 0:
                ctrl_pts = np.array([
                    [0.0, 0.0, 0.0],
                    [result.x[0], result.x[1], 30.0],
                    [result.x[2], result.x[3], 60.0],
                    [-2.0, 0.0, 90.0]
                ])
                path = _bezier_path(ctrl_pts, 10)
                s = _score(path)
                if s > best_score:
                    best_score = s
                    best_path = path
        except Exception:
            pass

    # --- Phase 2: 6-parameter Bezier optimization ---
    if time.time() < deadline - 90:
        initial_guesses_6d = [
            [-0.5, 0.5, -1.5, 0.5, 30, 60],
            [-1.0, 1.0, -1.0, 1.0, 30, 60],
            [-0.5, 1.0, -1.5, 1.0, 40, 50],
            [-1.0, 0.5, -1.0, 0.5, 25, 65],
        ]

        for init in initial_guesses_6d:
            if time.time() > deadline - 60:
                break
            try:
                result = minimize(
                    _neg_score_6d,
                    init,
                    method='L-BFGS-B',
                    options={'maxiter': 30, 'maxfun': 150}
                )
                if result.success or result.fun < 0:
                    ctrl_pts = np.array([
                        [0.0, 0.0, 0.0],
                        [result.x[0], result.x[1], result.x[4]],
                        [result.x[2], result.x[3], result.x[5]],
                        [-2.0, 0.0, 90.0]
                    ])
                    path = _bezier_path(ctrl_pts, 10)
                    s = _score(path)
                    if s > best_score:
                        best_score = s
                        best_path = path
            except Exception:
                pass

    # --- Phase 3: Two-arc path search ---
    if time.time() < deadline - 60:
        cx1_vals = np.linspace(-2, 1, 15)
        cy1_vals = np.linspace(-2, 2, 15)
        cx2_vals = np.linspace(-4, -1, 15)
        cy2_vals = np.linspace(-2, 2, 15)

        for cy1 in cy1_vals:
            if time.time() > deadline - 45:
                break
            for cx1 in cx1_vals:
                if time.time() > deadline - 45:
                    break
                for cy2 in cy2_vals:
                    if time.time() > deadline - 45:
                        break
                    for cx2 in cx2_vals:
                        if time.time() > deadline - 45:
                            break
                        path = _two_arc_path(cx1, cy1, cx2, cy2, 5, 5)
                        if path is not None:
                            s = _score(path)
                            if s > best_score:
                                best_score = s
                                best_path = path

    # --- Phase 4: Local refinement of best path ---
    if best_path is not None and time.time() < deadline - 20:
        current = best_path.copy()
        n = len(current)
        steps = [0.1, 0.05, 0.02, 0.01]
        for step in steps:
            if time.time() > deadline - 15:
                break
            improved = True
            while improved and time.time() < deadline - 15:
                improved = False
                for i in range(1, n - 1):
                    if time.time() > deadline - 15:
                        break
                    for dx in [-step, 0.0, step]:
                        if time.time() > deadline - 15:
                            break
                        for dy in [-step, 0.0, step]:
                            if time.time() > deadline - 15:
                                break
                            if dx == 0.0 and dy == 0.0:
                                continue
                            trial = current.copy()
                            trial[i, 0] += dx
                            trial[i, 1] += dy
                            s = _score(trial)
                            if s > best_score:
                                best_score = s
                                best_path = trial
                                current = trial
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break

    # --- Phase 5: Try different numbers of Bezier samples ---
    if best_path is not None and time.time() < deadline - 10:
        # Extract approximate control points from best path
        # Just try varying n_samples for a simple Bezier
        for n_samples in [8, 12, 14, 16]:
            if time.time() > deadline - 10:
                break
            ctrl_pts = np.array([
                [0.0, 0.0, 0.0],
                [-0.8, 0.8, 30.0],
                [-1.2, 0.8, 60.0],
                [-2.0, 0.0, 90.0]
            ])
            path = _bezier_path(ctrl_pts, n_samples)
            s = _score(path)
            if s > best_score:
                best_score = s
                best_path = path

    if best_path is None:
        # Fallback: simple linear path
        best_path = np.array([
            [0.0, 0.0, 0.0],
            [-0.5, 0.5, 22.5],
            [-1.0, 1.0, 45.0],
            [-1.5, 0.5, 67.5],
            [-2.0, 0.0, 90.0]
        ], dtype=np.float64)

    if not np.all(np.isfinite(best_path)):
        best_path = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)

    return best_path


# EVOLVE-BLOCK-END
