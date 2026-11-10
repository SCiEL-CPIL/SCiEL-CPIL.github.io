import numpy as np
import time
from scipy.optimize import minimize
from scipy.interpolate import BSpline

def moving_sofa_2d() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590  # Safety margin

    # Grid definition
    xs = np.arange(-3.0, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    n_points = X_flat.size

    P_start = np.array([0.0, 0.0, 0.0])
    P_end = np.array([-2.0, 0.0, 90.0])

    def densify_path(path):
        if len(path) == 0:
            return np.array([P_start, P_end])
        
        full_path = np.vstack([P_start, path, P_end])
        
        dense_points = [full_path[0]]
        for i in range(len(full_path) - 1):
            p1 = full_path[i]
            p2 = full_path[i+1]
            delta_a = abs(p2[2] - p1[2])
            m = max(1, int(np.ceil(delta_a)))
            for k in range(1, m + 1):
                t = k / m
                p_int = p1 + t * (p2 - p1)
                dense_points.append(p_int)
        
        return np.array(dense_points)

    def score_path_fast(path):
        dense = densify_path(path)
        M = dense.shape[0]
        if M == 0:
            return 0.0
        
        alive = np.ones(n_points, dtype=bool)
        
        # Precompute cos/sin for all dense points
        a_rad = np.deg2rad(dense[:, 2])
        cos_all = np.cos(a_rad)
        sin_all = np.sin(a_rad)
        
        for i in range(M):
            if not np.any(alive):
                break
            
            x0, y0, a = dense[i]
            c, s = cos_all[i], sin_all[i]
            
            X_alive = X_flat[alive]
            Y_alive = Y_flat[alive]
            
            dx = X_alive - x0
            dy = Y_alive - y0
            
            u = dx * c + dy * s
            v = -dx * s + dy * c
            
            # Base corridor:
            # R1: 0 <= u <= 1, -3 <= v <= 1
            # R2: -3 <= u <= 1, 0 <= v <= 1
            
            in_r1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
            in_r2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
            
            inside = in_r1 | in_r2
            
            # Update alive mask
            alive_indices = np.where(alive)[0]
            alive[alive_indices[~inside]] = False
        
        count = np.count_nonzero(alive)
        return count * 0.0001

    def bspline_to_waypoints(ctrl_pts, n_samples=20):
        """Convert B-spline control points to actual waypoints."""
        # Control points shape: (n_ctrl, 3)
        # We use a cubic B-spline
        n_ctrl = ctrl_pts.shape[0]
        
        # Knot vector for clamped cubic B-spline
        # For n_ctrl control points, we need n_ctrl + 4 knots for cubic
        k = 3  # degree
        # Clamped knot vector: [0,0,0,0, 1, 2, ..., n_ctrl-4, n_ctrl-3, n_ctrl-3, n_ctrl-3, n_ctrl-3]
        # Actually for n control points and degree k, we need n+k+1 knots
        # Clamped: first k+1 knots = 0, last k+1 knots = n_ctrl-1
        n_knots = n_ctrl + k + 1
        knots = np.zeros(n_knots)
        # Interior knots evenly spaced
        interior = np.arange(1, n_knots - k - 1)
        knots[1:n_knots-k-1] = interior / (n_knots - 2*k - 2) if n_knots > 2*k+2 else interior
        knots[1:k+1] = 0.0
        knots[n_knots-k-1:] = 1.0
        
        # Actually let's use a simpler approach with scipy
        # Use make_interp_spline or direct BSpline
        # For simplicity, let's use a piecewise cubic with clamped ends
        
        # Simpler: use cubic interpolation on control points
        # Parameterize t from 0 to 1
        t_ctrl = np.linspace(0, 1, n_ctrl)
        t_samples = np.linspace(0, 1, n_samples)
        
        # Cubic interpolation for each coordinate
        from scipy.interpolate import CubicSpline
        waypoints = np.zeros((n_samples, 3))
        for d in range(3):
            cs = CubicSpline(t_ctrl, ctrl_pts[:, d])
            waypoints[:, d] = cs(t_samples)
        
        return waypoints

    def objective(ctrl_pts):
        """Objective function to minimize (negative of score)."""
        if time.time() > deadline - 30:
            return 1e10
        
        waypoints = bspline_to_waypoints(ctrl_pts, n_samples=15)
        
        # Ensure angles are monotonic
        angles = waypoints[:, 2]
        for i in range(1, len(angles)):
            if angles[i] < angles[i-1]:
                angles[i] = angles[i-1]
        
        waypoints[:, 2] = angles
        
        score = score_path_fast(waypoints)
        return -score

    def optimize_bspline(n_ctrl, n_samples=15, max_iter=50):
        """Optimize B-spline control points using SLSQP."""
        # Initial control points: linear interpolation from start to end with some curvature
        t = np.linspace(0, 1, n_ctrl)
        
        # x: 0 to -2
        x0 = -2.0 * t
        # y: slight arc
        y0 = 0.5 * np.sin(np.pi * t)
        # angle: 0 to 90
        a0 = 90.0 * t
        
        ctrl_init = np.column_stack([x0, y0, a0])
        
        # Add some randomness
        ctrl_init[:, 1] += np.random.normal(0, 0.1, n_ctrl)
        ctrl_init[:, 1] = np.clip(ctrl_init[:, 1], -0.5, 1.5)
        
        # Bounds for control points
        bounds = [
            (-3.0, 1.0),  # x range
            (-1.0, 2.0),  # y range
            (0.0, 90.0)   # angle range
        ] * n_ctrl
        
        # Constraints: angle monotonicity
        def angle_mono_constraint(x):
            ctrl = x.reshape(-1, 3)
            angles = ctrl[:, 2]
            return np.diff(angles)
        
        constraints = [
            {'type': 'ineq', 'fun': angle_mono_constraint}
        ]
        
        try:
            result = minimize(
                objective,
                ctrl_init.flatten(),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': max_iter, 'ftol': 1e-4}
            )
            
            if result.success or result.fun < 1e9:
                best_ctrl = result.x.reshape(-1, 3)
                # Ensure monotonic angles
                for i in range(1, best_ctrl.shape[0]):
                    if best_ctrl[i, 2] < best_ctrl[i-1, 2]:
                        best_ctrl[i, 2] = best_ctrl[i-1, 2]
                return best_ctrl
        except Exception:
            pass
        
        return ctrl_init

    best_path = None
    best_score = 0.0
    
    # Try different numbers of control points
    for n_ctrl in [4, 5, 6, 7]:
        if time.time() > deadline - 60:
            break
        
        # Multiple random restarts
        for restart in range(3):
            if time.time() > deadline - 40:
                break
            
            np.random.seed(42 + restart * 100 + n_ctrl)
            
            ctrl = optimize_bspline(n_ctrl, max_iter=30)
            
            # Convert to waypoints and evaluate
            waypoints = bspline_to_waypoints(ctrl, n_samples=12)
            
            # Ensure monotonic angles
            angles = waypoints[:, 2]
            for i in range(1, len(angles)):
                if angles[i] < angles[i-1]:
                    angles[i] = angles[i-1]
            waypoints[:, 2] = angles
            
            s = score_path_fast(waypoints)
            if s > best_score:
                best_score = s
                best_path = waypoints.copy()
    
    # If B-spline didn't find anything good, fall back to hill climbing
    if best_path is None or best_score < 0.01:
        # Use the original hill climbing approach as fallback
        def make_initial_path(n_wp, seed_offset=0):
            angles = np.linspace(0, 90, n_wp + 2)[1:-1]
            angles += np.random.normal(0, 2, n_wp)
            angles = np.maximum.accumulate(angles)
            angles = np.clip(angles, 1, 89)
            
            t = np.linspace(0, 1, n_wp)
            xs_init = -2.0 * t + np.random.normal(0, 0.2, n_wp)
            ys_init = 0.5 * np.sin(np.pi * t) + np.random.normal(0, 0.2, n_wp)
            
            return np.column_stack([xs_init, ys_init, angles])
        
        np.random.seed(42)
        
        for n_wp in [5, 7, 9]:
            if time.time() > deadline - 30:
                break
            
            path = make_initial_path(n_wp)
            s = score_path_fast(path)
            if s > best_score:
                best_score = s
                best_path = path.copy()
    
    # Final hill climbing refinement
    if best_path is not None and time.time() < deadline - 15:
        current_path = best_path.copy()
        current_score = score_path_fast(current_path)
        
        sigma = 0.15
        no_improve = 0
        
        while time.time() < deadline - 10 and no_improve < 80:
            improved = False
            n_coords = current_path.shape[0] * 3
            
            for _ in range(6):
                if time.time() > deadline - 10:
                    break
                
                coord_idx = np.random.randint(0, n_coords)
                row = coord_idx // 3
                col = coord_idx % 3
                
                for delta in [sigma, -sigma, sigma*0.5, -sigma*0.5]:
                    new_path = current_path.copy()
                    new_path[row, col] += delta
                    
                    if col == 2:
                        if row > 0:
                            new_path[row, 2] = max(new_path[row, 2], new_path[row-1, 2])
                        if row < current_path.shape[0] - 1:
                            new_path[row, 2] = min(new_path[row, 2], new_path[row+1, 2])
                    
                    new_score = score_path_fast(new_path)
                    
                    if new_score > current_score:
                        current_path = new_path
                        current_score = new_score
                        improved = True
                        break
            
            if improved:
                no_improve = 0
                sigma *= 0.9
            else:
                no_improve += 1
                sigma *= 0.95
        
        if current_score > best_score:
            best_score = current_score
            best_path = current_path.copy()
    
    if best_path is None:
        best_path = np.array([[-0.5, 0.3, 22.5], [-1.0, 0.5, 45.0], [-1.5, 0.3, 67.5]])
    
    return best_path
