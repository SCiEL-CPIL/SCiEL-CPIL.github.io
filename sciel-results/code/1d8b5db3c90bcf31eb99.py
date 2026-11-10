# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize, differential_evolution
from scipy.interpolate import make_interp_spline

def moving_sofa_2d() -> np.ndarray:
    """
    Find a path of (x, y, angle) placements for the L-shaped corridor that
    maximizes the area of the intersection of all corridor placements.
    
    Strategy:
    1. Parameterize the path using cubic splines for x(t), y(t), and theta(t)
       with a small number of control points (e.g., 5-8).
    2. The parameter t ranges from 0 to 1, corresponding to angles 0 to 90.
       Actually, the angles are determined by the spline theta(t).
       The evaluator interpolates between consecutive returned points.
       We return a set of points sampled from the spline.
    3. Optimize the control points using differential_evolution or multi-start Nelder-Mead.
    """
    start_time = time.time()
    deadline = start_time + 580  # safety margin before 630s limit

    # Precompute the sofa grid
    xs = np.arange(-3.0, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    gx = X.ravel()
    gy = Y.ravel()
    n_pts = len(gx)

    def _densify_path(path):
        """
        Densify path exactly as evaluator: prepend (0,0,0), append (-2,0,90),
        then interpolate between consecutive points in m = max(1, ceil(|da|)) steps.
        Returns array of shape (N, 3).
        """
        full = np.vstack([[0.0, 0.0, 0.0], path, [-2.0, 0.0, 90.0]])
        result = [full[0]]
        for i in range(len(full) - 1):
            p1 = full[i]
            p2 = full[i + 1]
            da = abs(p2[2] - p1[2])
            m = max(1, int(math.ceil(da)))
            for k in range(1, m + 1):
                t = k / m
                interp = p1 + t * (p2 - p1)
                result.append(interp)
        return np.array(result)

    def _evaluate_path_vectorized(path):
        """
        Vectorized evaluation: compute for all placements at once using broadcasting.
        Returns the raw value (area) as a float.
        """
        dense = _densify_path(path)
        n_placements = len(dense)
        
        # Extract placement parameters
        cx = dense[:, 0]
        cy = dense[:, 1]
        a_deg = dense[:, 2]
        a_rad = np.radians(a_deg)
        cos_a = np.cos(a_rad)
        sin_a = np.sin(a_rad)
        
        # Compute rotated coordinates for all grid points and all placements
        # Shape: (n_placements, n_pts)
        dx = gx[None, :] - cx[:, None]
        dy = gy[None, :] - cy[:, None]
        u = cos_a[:, None] * dx + sin_a[:, None] * dy
        v = -sin_a[:, None] * dx + cos_a[:, None] * dy
        
        # Check base corridor membership for all points
        in_arm1 = (u >= 0.0) & (u <= 1.0) & (v >= -3.0) & (v <= 1.0)
        in_arm2 = (u >= -3.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)
        in_base = in_arm1 | in_arm2  # Shape: (n_placements, n_pts)
        
        # A point is in the sofa if it's in ALL placements
        mask = in_base.all(axis=0)  # Shape: (n_pts,)
        
        return float(np.sum(mask)) * 0.0001

    def _evaluate_path(path):
        """
        Evaluate the area of the intersection of all corridor placements.
        Returns the raw value (area) as a float.
        """
        return _evaluate_path_vectorized(path)

    # Parameterization using splines
    # We define control points for x, y, and theta at specific t values.
    # Let's use a fixed set of t values for control points, e.g., [0, 0.25, 0.5, 0.75, 1]
    # The angles at these points will be optimized.
    # Actually, to ensure the path starts at angle 0 and ends at 90, we fix theta(0)=0 and theta(1)=90.
    # The intermediate theta values are optimized.
    
    n_ctrl = 5 # Number of control points
    t_ctrl = np.linspace(0, 1, n_ctrl)
    
    # We optimize x_ctrl, y_ctrl, and theta_ctrl (intermediate only)
    # x: n_ctrl, y: n_ctrl, theta: n_ctrl - 2 (middle ones)
    n_params = n_ctrl * 2 + (n_ctrl - 2)
    
    def _params_to_path(params, n_samples=20):
        """
        Convert optimization parameters to a valid path.
        params: array of shape (n_params,)
        """
        x_ctrl = params[0:n_ctrl]
        y_ctrl = params[n_ctrl:2*n_ctrl]
        theta_ctrl = params[2*n_ctrl:]
        
        # Construct full theta control points
        theta_full = np.zeros(n_ctrl)
        theta_full[0] = 0.0
        theta_full[-1] = 90.0
        theta_full[1:-1] = theta_ctrl
        
        # Create splines
        try:
            spline_x = make_interp_spline(t_ctrl, x_ctrl, k=3)
            spline_y = make_interp_spline(t_ctrl, y_ctrl, k=3)
            spline_theta = make_interp_spline(t_ctrl, theta_full, k=3)
        except Exception:
            # Fallback to linear if spline fails (shouldn't happen with distinct t)
            spline_x = make_interp_spline(t_ctrl, x_ctrl, k=1)
            spline_y = make_interp_spline(t_ctrl, y_ctrl, k=1)
            spline_theta = make_interp_spline(t_ctrl, theta_full, k=1)
        
        # Sample the splines at n_samples points
        t_samples = np.linspace(0, 1, n_samples)
        x_samples = spline_x(t_samples)
        y_samples = spline_y(t_samples)
        theta_samples = spline_theta(t_samples)
        
        # Ensure angles are monotonic? The evaluator handles interpolation, 
        # but non-monotonic angles might cause issues with 'ceil(|da|)'.
        # Let's assume the optimizer keeps them monotonic or close to it.
        
        path = np.column_stack((x_samples, y_samples, theta_samples))
        return path

    def _objective(params):
        """Negative area for minimization."""
        if time.time() > deadline:
            return 0.0
        path = _params_to_path(params)
        return -_evaluate_path(path)

    # Bounds for optimization
    # x, y should be in a reasonable range, e.g., [-3, 3]
    # theta should be in [0, 90]
    bounds = [(-3.0, 3.0)] * (2 * n_ctrl) + [(0.0, 90.0)] * (n_ctrl - 2)
    
    # Initial guess
    # A simple linear path in x, y and theta
    x0_x = np.linspace(0.0, -2.0, n_ctrl)
    x0_y = np.zeros(n_ctrl)
    x0_theta = np.linspace(0.0, 90.0, n_ctrl)
    x0 = np.concatenate([x0_x, x0_y, x0_theta[1:-1]])
    
    best_params = x0.copy()
    best_score = -_objective(x0)
    
    # Use Differential Evolution for global search
    # This is slower but more robust for non-convex landscapes
    # We need to be careful with the budget.
    
    # Try a few local optimizations from different starts first
    np.random.seed(42)
    n_restarts = 5
    for r in range(n_restarts):
        if time.time() > deadline - 120:
            break
        # Random perturbation
        noise = np.random.normal(0, 0.5, size=n_params)
        x_try = best_params + noise
        x_try = np.clip(x_try, [b[0] for b in bounds], [b[1] for b in bounds])
        
        res = minimize(_objective, x_try, method='Nelder-Mead',
                       options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-7, 'adaptive': True})
        if res.fun < -best_score:
            best_score = -res.fun
            best_params = res.x

    # Now try Differential Evolution if time permits
    if time.time() < deadline - 180:
        # DE is expensive, so use a small population and maxiter
        try:
            res_de = differential_evolution(_objective, bounds, 
                                             maxiter=10, popsize=10, 
                                             tol=1e-4, seed=42, 
                                             polish=True)
            if res_de.fun < -best_score:
                best_score = -res_de.fun
                best_params = res_de.x
        except Exception:
            pass

    # Final polish with tighter tolerances
    if time.time() < deadline - 30:
        res = minimize(_objective, best_params, method='Nelder-Mead',
                       options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
        if res.fun < -best_score:
            best_score = -res.fun
            best_params = res.x

    final_path = _params_to_path(best_params, n_samples=15)
    return final_path

# EVOLVE-BLOCK-END
