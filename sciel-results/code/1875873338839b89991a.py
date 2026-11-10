import numpy as np
import time
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline

def moving_sofa_2d() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    
    # Sofa points (grid)
    x_sofa = np.arange(-3, 1.5, 0.01)
    y_sofa = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(x_sofa, y_sofa)
    sofa_pts = np.vstack([X.ravel(), Y.ravel()]).T
    n_pts = len(sofa_pts)
    
    def densify_path(path):
        """Apply the evaluator's densification rule."""
        full = np.vstack([np.array([[0.0, 0.0, 0.0]]), path, np.array([[-2.0, 0.0, 90.0]])])
        result = [full[0]]
        for i in range(len(full) - 1):
            p1, p2 = full[i], full[i + 1]
            da = abs(p2[2] - p1[2])
            m = max(1, int(np.ceil(da)))
            if m == 1:
                result.append(p2)
            else:
                for k in range(1, m + 1):
                    t = k / m
                    result.append(p1 + t * (p2 - p1))
        return np.array(result)
    
    def score_path_fast(path, subsample_idx=None):
        """Fast score using subsampled sofa points."""
        if time.time() > deadline - 5:
            return 0.0
        full = densify_path(path)
        if subsample_idx is None:
            idx = np.random.choice(n_pts, size=800, replace=False)
            pts = sofa_pts[idx]
            scale = n_pts / 800.0
        else:
            pts = sofa_pts
            scale = 1.0
        
        inside = np.ones(len(pts), dtype=bool)
        for row in full:
            xc, yc, a_deg = row
            a_rad = np.radians(a_deg)
            dx = pts[:, 0] - xc
            dy = pts[:, 1] - yc
            ca = np.cos(-a_rad)
            sa = np.sin(-a_rad)
            u = ca * dx - sa * dy
            v = sa * dx + ca * dy
            m1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
            m2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
            inside &= (m1 | m2)
            if not inside.any():
                return 0.0
        return np.sum(inside) / len(pts) * scale * 0.0001
    
    def score_path_full(path):
        """Full score with all sofa points."""
        full = densify_path(path)
        inside = np.ones(n_pts, dtype=bool)
        for row in full:
            xc, yc, a_deg = row
            a_rad = np.radians(a_deg)
            dx = sofa_pts[:, 0] - xc
            dy = sofa_pts[:, 1] - yc
            ca = np.cos(-a_rad)
            sa = np.sin(-a_rad)
            u = ca * dx - sa * dy
            v = sa * dx + ca * dy
            m1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
            m2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
            inside &= (m1 | m2)
            if not inside.any():
                return 0.0
        return np.sum(inside) * 0.0001
    
    def spline_to_path(cp, n_samples=60):
        """Convert control points to a sampled path."""
        # cp is array of shape (n_cp, 3)
        n_cp = len(cp)
        if n_cp < 2:
            return cp
        t = np.linspace(0, 1, n_cp)
        t_eval = np.linspace(0, 1, n_samples)
        # Build cubic splines for x, y, a
        # Use natural boundary conditions
        cs_x = CubicSpline(t, cp[:, 0], bc_type='not-a-knot')
        cs_y = CubicSpline(t, cp[:, 1], bc_type='not-a-knot')
        cs_a = CubicSpline(t, cp[:, 2], bc_type='not-a-knot')
        x_s = cs_x(t_eval)
        y_s = cs_y(t_eval)
        a_s = cs_a(t_eval)
        path = np.column_stack([x_s, y_s, a_s])
        # Ensure monotonic angle (no backtracking)
        for i in range(1, len(path)):
            if path[i, 2] < path[i-1, 2]:
                path[i, 2] = path[i-1, 2]
        return path
    
    def objective(cp):
        if time.time() > deadline - 10:
            return 0.0
        path = spline_to_path(cp, n_samples=50)
        # Ensure endpoints are reasonable
        # The spline should start near (0,0,0) and end near (-2,0,90)
        # We fix the first and last control points
        return -score_path_fast(path)
    
    # Initial guess: linear path from (0,0,0) to (-2,0,90)
    n_cp = 8
    t_init = np.linspace(0, 1, n_cp)
    cp_init = np.column_stack([
        -2.0 * t_init,
        0.5 * np.sin(np.pi * t_init),  # Arc up
        90.0 * t_init
    ])
    # Fix endpoints
    cp_init[0] = [0.0, 0.0, 0.0]
    cp_init[-1] = [-2.0, 0.0, 90.0]
    
    best_path = spline_to_path(cp_init, n_samples=60)
    best_score = score_path_fast(best_path)
    
    # SLSQP optimization on interior control points
    def obj_interior(x):
        cp = cp_init.copy()
        cp[1:-1] = x.reshape(-1, 3)
        return objective(cp)
    
    x0 = cp_init[1:-1].ravel()
    bounds = [(-2.5, 2.5)] * (len(x0))
    
    # Run SLSQP with time check
    elapsed = time.time() - start_time
    if elapsed < deadline - 60:
        try:
            res = minimize(obj_interior, x0, method='SLSQP', bounds=bounds,
                          options={'maxiter': 50, 'ftol': 1e-6})
            if res.success and time.time() < deadline - 30:
                cp_opt = cp_init.copy()
                cp_opt[1:-1] = res.x.reshape(-1, 3)
                path_opt = spline_to_path(cp_opt, n_samples=60)
                s_opt = score_path_fast(path_opt)
                if s_opt > best_score:
                    best_path = path_opt
                    best_score = s_opt
        except Exception:
            pass
    
    # Local polish: perturb and re-evaluate
    elapsed = time.time() - start_time
    if elapsed < deadline - 30:
        for _ in range(20):
            if time.time() > deadline - 15:
                break
            cp_cur = spline_to_path(best_path, n_samples=n_cp)
            # Random perturbation
            cp_pert = cp_cur.copy()
            cp_pert[1:-1] += np.random.randn(n_cp - 2, 3) * 0.1
            cp_pert[0] = [0.0, 0.0, 0.0]
            cp_pert[-1] = [-2.0, 0.0, 90.0]
            path_pert = spline_to_path(cp_pert, n_samples=60)
            s_pert = score_path_fast(path_pert)
            if s_pert > best_score:
                best_path = path_pert
                best_score = s_pert
    
    # Final dense evaluation
    final_path = spline_to_path(best_path, n_samples=100)
    # Make sure it's a valid array
    if not np.all(np.isfinite(final_path)):
        final_path = np.array([[-1.0, 0.5, 45.0]])
    
    return final_path
