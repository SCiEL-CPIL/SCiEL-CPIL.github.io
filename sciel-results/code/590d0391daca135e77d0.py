# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize
from scipy.interpolate import UnivariateSpline

def moving_sofa_2d() -> np.ndarray:
    """
    Find a path of (x, y, angle) placements for the L-shaped corridor that
    maximizes the area of the intersection of all corridor placements.
    
    Strategy: 
    1. Parameterize the path using B-splines for x(t), y(t), and theta(t) 
       to ensure smoothness.
    2. Use CMA-ES (implemented via scipy's differential_evolution or manual 
       implementation if needed, but scipy.optimize.differential_evolution is 
       robust) or a custom CMA-ES to optimize the control points of the splines.
       Actually, let's use a simple, robust global optimizer: 
       scipy.optimize.differential_evolution is good but can be slow.
       Let's try a hybrid: 
       1. Coarse search with Random Restarts + Nelder-Mead on a B-spline control point representation.
       2. The B-spline representation reduces the dimensionality and enforces smoothness.
    """
    start_time = time.time()
    deadline = start_time + 590  # Safety margin

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

    # B-Spline Parameterization
    # We define the path by control points for x, y, and theta.
    # t goes from 0 to 1.
    # Fixed endpoints:
    # t=0: x=0, y=0, theta=0
    # t=1: x=-2, y=0, theta=90
    # We use cubic B-splines.
    # Number of interior control points: n_ctrl
    # Total control points: n_ctrl + 2 (including endpoints)
    
    def _get_spline_values(control_points, t):
        """
        Evaluate a cubic B-spline at t.
        control_points: array of shape (n_ctrl_total,)
        t: array of shape (n_eval,)
        """
        # Simple cubic interpolation with fixed endpoints
        # We'll use numpy interp with a dense set of points for speed and simplicity,
        # or better, use scipy.interpolate.CubicSpline
        # Since we need to call this frequently, let's precompute the spline basis or use a fast method.
        # For now, let's use a simple piecewise cubic Hermite or just linear for speed in the inner loop?
        # No, smoothness is key. Let's use a fixed set of knots.
        pass

    # Let's use a simpler, faster parameterization: 
    # Polynomial of degree 3 for x(t), y(t), theta(t).
    # x(t) = c0 + c1*t + c2*t^2 + c3*t^3
    # Constraints: x(0)=0, x(1)=-2
    # y(t) = d0 + d1*t + d2*t^2 + d3*t^3
    # Constraints: y(0)=0, y(1)=0
    # theta(t) = e0 + e1*t + e2*t^2 + e3*t^3
    # Constraints: theta(0)=0, theta(1)=90
    
    # This reduces the problem to optimizing 4 coefficients for each, with 2 fixed by boundaries.
    # x(t) = c2*t^2 + c3*t^3  (since x(0)=0, x(1)=c2+c3=-2 => c3 = -2-c2)
    # So x(t) = c2*t^2 + (-2-c2)*t^3 = c2*(t^2-t^3) - 2*t^3
    # Wait, x(1) = c2 - 2 - c2 = -2. Correct.
    # y(t) = d2*t^2 + d3*t^3, y(1)=d2+d3=0 => d3=-d2.
    # y(t) = d2*(t^2-t^3)
    # theta(t) = e2*t^2 + e3*t^3, theta(1)=e2+e3=90 => e3=90-e2.
    # theta(t) = e2*(t^2-t^3) + 90*t^3
    
    # This is a 3-parameter optimization! (c2, d2, e2)
    # This is extremely fast to evaluate.
    
    def _params_to_path_poly(params):
        """
        params: [c2, d2, e2]
        Returns path points at t=0.05, 0.1, ..., 0.95 (interior points)
        """
        c2, d2, e2 = params
        t_vals = np.linspace(0.05, 0.95, 10) # 10 interior points
        
        x_vals = c2 * (t_vals**2 - t_vals**3) - 2 * t_vals**3
        y_vals = d2 * (t_vals**2 - t_vals**3)
        theta_vals = e2 * (t_vals**2 - t_vals**3) + 90 * t_vals**3
        
        path = np.column_stack((x_vals, y_vals, theta_vals))
        return path

    def _objective_poly(params):
        if time.time() > deadline:
            return 0.0
        path = _params_to_path_poly(params)
        return -_evaluate_path_vectorized(path)

    # Initial guess
    # We want smooth movement.
    # c2, d2, e2 start at 0 (linear-ish? No, t^3 is slow start, fast end)
    # Let's try small values.
    x0 = np.array([0.0, 0.0, 0.0])
    
    best_params = x0.copy()
    best_score = -_objective_poly(x0)
    
    # Optimize using Nelder-Mead on the 3 parameters
    res = minimize(_objective_poly, x0, method='Nelder-Mead',
                   options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8})
    if res.fun < -best_score:
        best_score = -res.fun
        best_params = res.x

    # Try random restarts
    np.random.seed(42)
    for r in range(50):
        if time.time() > deadline - 30:
            break
        noise = np.random.normal(0, 5.0, size=3)
        x_try = best_params + noise
        res = minimize(_objective_poly, x_try, method='Nelder-Mead',
                       options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
        if res.fun < -best_score:
            best_score = -res.fun
            best_params = res.x

    # Now, let's see if we can improve with more flexibility.
    # Use 5th degree polynomials?
    # x(t) = c2*t^2 + c3*t^3 + c4*t^4 + c5*t^5
    # x(0)=0, x(1)=-2
    # y(t) = d2*t^2 + d3*t^3 + d4*t^4 + d5*t^5
    # y(0)=0, y(1)=0
    # theta(t) = e2*t^2 + e3*t^3 + e4*t^4 + e5*t^5
    # theta(0)=0, theta(1)=90
    
    # This increases dimension to 9.
    
    def _params_to_path_poly5(params):
        c2, c3, c4, c5, d2, d3, d4, d5, e2, e3, e4, e5 = params
        t_vals = np.linspace(0.05, 0.95, 10)
        
        t2 = t_vals**2
        t3 = t_vals**3
        t4 = t_vals**4
        t5 = t_vals**5
        
        x_vals = c2*t2 + c3*t3 + c4*t4 + c5*t5
        # Constraint x(1) = c2+c3+c4+c5 = -2
        # We enforce this by adjusting c5? Or just let the optimizer find it?
        # Better to parameterize with free params and enforce constraints via projection?
        # Or just add a penalty?
        # Let's stick to the 3-parameter version if it's good enough, or use a B-spline.
        
        # Actually, the 3-parameter version might be too restrictive.
        # Let's use a B-spline with 4 interior control points.
        pass

    # Let's refine the 3-parameter solution first.
    # Then try a 6-parameter version (quartic).
    
    def _params_to_path_poly4(params):
        # x(t) = c2*t^2 + c3*t^3 + c4*t^4
        # x(1) = c2+c3+c4 = -2 => c4 = -2 - c2 - c3
        # y(t) = d2*t^2 + d3*t^3 + d4*t^4
        # y(1) = d2+d3+d4 = 0 => d4 = -d2 - d3
        # theta(t) = e2*t^2 + e3*t^3 + e4*t^4
        # theta(1) = e2+e3+e4 = 90 => e4 = 90 - e2 - e3
        
        c2, c3, d2, d3, e2, e3 = params
        c4 = -2 - c2 - c3
        d4 = -d2 - d3
        e4 = 90 - e2 - e3
        
        t_vals = np.linspace(0.05, 0.95, 10)
        t2 = t_vals**2
        t3 = t_vals**3
        t4 = t_vals**4
        
        x_vals = c2*t2 + c3*t3 + c4*t4
        y_vals = d2*t2 + d3*t3 + d4*t4
        theta_vals = e2*t2 + e3*t3 + e4*t4
        
        path = np.column_stack((x_vals, y_vals, theta_vals))
        return path

    def _objective_poly4(params):
        if time.time() > deadline:
            return 0.0
        path = _params_to_path_poly4(params)
        return -_evaluate_path_vectorized(path)

    # Initialize poly4 from poly3
    # poly3: c2, d2, e2
    # poly4: c2, c3=0, d2, d3=0, e2, e3=0
    x0_4 = np.array([best_params[0], 0.0, best_params[1], 0.0, best_params[2], 0.0])
    
    best_params_4 = x0_4.copy()
    best_score_4 = -_objective_poly4(x0_4)
    
    if best_score_4 < best_score:
        best_score = best_score_4
        best_params = best_params_4
        # Use poly4 for further optimization
        opt_params = best_params_4
        obj_func = _objective_poly4
        n_params = 6
    else:
        opt_params = best_params
        obj_func = _objective_poly
        n_params = 3

    # Optimize the chosen representation
    res = minimize(obj_func, opt_params, method='Nelder-Mead',
                   options={'maxiter': 2000, 'xatol': 1e-7, 'fatol': 1e-9})
    if res.fun < -best_score:
        best_score = -res.fun
        best_params = res.x

    # Random restarts on the best representation
    for r in range(30):
        if time.time() > deadline - 10:
            break
        noise = np.random.normal(0, 2.0, size=n_params)
        x_try = best_params + noise
        res = minimize(obj_func, x_try, method='Nelder-Mead',
                       options={'maxiter': 1000, 'xatol': 1e-7, 'fatol': 1e-9})
        if res.fun < -best_score:
            best_score = -res.fun
            best_params = res.x

    # Generate final path
    if n_params == 6:
        final_path = _params_to_path_poly4(best_params)
    else:
        final_path = _params_to_path_poly(best_params)
        
    return final_path

# EVOLVE-BLOCK-END
