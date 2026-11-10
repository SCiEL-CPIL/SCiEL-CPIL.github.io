# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt
from scipy.interpolate import CubicSpline

def moving_sofa_2d() -> np.ndarray:
    """
    Moves a sofa through an L-shaped corridor.
    Uses a cubic B-spline path parameterized by angle, optimized with Nelder-Mead.
    Endpoints are fixed: (0,0) at angle 0 and (-2,0) at angle 90.
    """
    # 1. Setup Grid
    x_grid = np.arange(-3, 1.5, 0.01)
    y_grid = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(x_grid, y_grid)
    P = np.column_stack((X.ravel(), Y.ravel()))
    N_points = P.shape[0]
    
    # Start and End states
    start_state = np.array([0.0, 0.0, 0.0])
    end_state = np.array([-2.0, 0.0, 90.0])
    
    N_KNOTS = 6  # number of control points per coordinate (including fixed endpoints)
    
    def densify_path(path_array):
        """
        Densifies a path array of shape (n, 3) such that angle steps are <= 1 degree.
        Returns a list of [x, y, a] arrays.
        """
        if path_array.size == 0:
            return [start_state, end_state]
            
        full_path = np.vstack([start_state, path_array, end_state])
        
        densified = [full_path[0]]
        for i in range(len(full_path) - 1):
            p1 = full_path[i]
            p2 = full_path[i + 1]
            
            a1 = p1[2]
            a2 = p2[2]
            
            da = a2 - a1
            steps = max(1, int(np.ceil(abs(da))))
            
            if steps == 1:
                densified.append(p2)
            else:
                dx = (p2[0] - p1[0]) / steps
                dy = (p2[1] - p1[1]) / steps
                da_step = da / steps
                
                for j in range(1, steps):
                    nx = p1[0] + j * dx
                    ny = p1[1] + j * dy
                    na = a1 + j * da_step
                    densified.append(np.array([nx, ny, na]))
                densified.append(p2)
                
        return densified

    def evaluate_path_fast(path_array):
        """
        Evaluates the area of the sofa for a given path array of shape (n, 3).
        Returns: float area
        """
        dense_path = densify_path(path_array)
        n_dense = len(dense_path)
        
        if n_dense == 0:
            return 0.0
        
        wx = P[:, 0]
        wy = P[:, 1]
        
        # Precompute all rotation parameters
        xs = np.array([p[0] for p in dense_path])
        ys = np.array([p[1] for p in dense_path])
        a_rads = np.deg2rad(np.array([p[2] for p in dense_path]))
        cos_as = np.cos(a_rads)
        sin_as = np.sin(a_rads)
        
        # Process in chunks to manage memory
        chunk_size = max(1, min(n_dense, 50))
        mask = np.ones(N_points, dtype=bool)
        
        for start_idx in range(0, n_dense, chunk_size):
            end_idx = min(start_idx + chunk_size, n_dense)
            
            # For each placement in the chunk, compute whether each point is inside
            for k in range(start_idx, end_idx):
                x = xs[k]
                y = ys[k]
                ca = cos_as[k]
                sa = sin_as[k]
                
                dx = wx - x
                dy = wy - y
                
                u = dx * ca + dy * sa
                v = -dx * sa + dy * ca
                
                in_r1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
                in_r2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
                
                in_corridor = in_r1 | in_r2
                
                mask &= in_corridor
                
                if not np.any(mask):
                    return 0.0
                    
        return float(np.sum(mask)) * 0.0001

    def get_path_from_coeffs(coeffs, n_waypoints=30):
        """
        coeffs: array of shape (2*(N_KNOTS-2),) -> interior control points for x and y
        Endpoints are fixed: x[0]=0, x[-1]=-2, y[0]=0, y[-1]=0
        Returns: np.ndarray of shape (n_waypoints, 3)
        """
        n_interior = N_KNOTS - 2
        xs_interior = coeffs[:n_interior]
        ys_interior = coeffs[n_interior:]
        
        # Build full control point arrays with fixed endpoints
        xs_ctrl = np.concatenate([[0.0], xs_interior, [-2.0]])
        ys_ctrl = np.concatenate([[0.0], ys_interior, [0.0]])
        
        # Knot locations in [0, 1]
        t_knots = np.linspace(0, 1, N_KNOTS)
        
        # Create cubic splines
        try:
            cs_x = CubicSpline(t_knots, xs_ctrl)
            cs_y = CubicSpline(t_knots, ys_ctrl)
        except Exception:
            # Fallback to linear interpolation if spline fails
            t_sample = np.linspace(0, 1, n_waypoints)
            xs = np.interp(t_sample, t_knots, xs_ctrl)
            ys = np.interp(t_sample, t_knots, ys_ctrl)
            angles_deg = t_sample * 90.0
            mask = (angles_deg > 0.5) & (angles_deg < 89.5)
            return np.column_stack((xs[mask], ys[mask], angles_deg[mask]))
        
        # Sample the spline at n_waypoints points
        t_sample = np.linspace(0, 1, n_waypoints)
        xs = cs_x(t_sample)
        ys = cs_y(t_sample)
        angles_deg = t_sample * 90.0
        
        # Exclude points too close to 0 or 90 to avoid duplicate/zero-step issues
        mask = (angles_deg > 0.5) & (angles_deg < 89.5)
        
        if not np.any(mask):
            return np.array([[-1.0, 0.1, 45.0]])
            
        return np.column_stack((xs[mask], ys[mask], angles_deg[mask]))

    def objective(coeffs, n_waypoints=25):
        """
        Objective function for minimization (scipy minimizes, so we negate area).
        """
        path = get_path_from_coeffs(coeffs, n_waypoints)
        area = evaluate_path_fast(path)
        return -area

    # 3. Optimization
    start_time = time.time()
    deadline = start_time + 580.0
    
    n_interior = N_KNOTS - 2
    
    # Initial guess: x goes from 0 to -2, y stays small
    # Interior x: [-0.4, -0.8, -1.2, -1.6]
    # Interior y: [0.1, 0.2, 0.2, 0.1]
    init_coeffs = np.array([-0.4, -0.8, -1.2, -1.6, 0.1, 0.2, 0.2, 0.1])
    
    best_coeffs = init_coeffs.copy()
    best_score = -objective(init_coeffs)
    
    # Try a few random perturbations to find a good starting point
    n_starts = 6
    rng = np.random.default_rng(42)
    for i in range(n_starts):
        if time.time() > deadline - 120:
            break
            
        if i == 0:
            coeffs = init_coeffs.copy()
        else:
            scale = 0.25 / (i + 1)
            coeffs = init_coeffs + rng.uniform(-scale, scale, size=2 * n_interior)
        
        score = -objective(coeffs)
        if score > best_score:
            best_score = score
            best_coeffs = coeffs.copy()
            
    # Run Nelder-Mead optimization
    if time.time() < deadline - 80:
        try:
            res = opt.minimize(objective, best_coeffs, method='Nelder-Mead', 
                               options={'maxiter': 400, 'xatol': 1e-4, 'fatol': 1e-6,
                                        'adaptive': True})
            
            if time.time() < deadline - 40:
                final_score = -res.fun
                if final_score > best_score:
                    best_score = final_score
                    best_coeffs = res.x
        except Exception as e:
            pass
    
    # Final polish: tighter optimization with more waypoints
    if time.time() < deadline - 30:
        def objective_polish(coeffs):
            return objective(coeffs, n_waypoints=40)
        
        try:
            res2 = opt.minimize(objective_polish, best_coeffs, method='Nelder-Mead', 
                                options={'maxiter': 150, 'xatol': 1e-5, 'fatol': 1e-7,
                                         'adaptive': True})
            
            if time.time() < deadline - 10:
                final_score2 = -res2.fun
                if final_score2 > best_score:
                    best_score = final_score2
                    best_coeffs = res2.x
        except Exception as e:
            pass
    
    # 4. Final Path Construction
    if best_coeffs is not None:
        final_path = get_path_from_coeffs(best_coeffs, n_waypoints=35)
    else:
        final_path = np.array([
            [-0.1, 0.1, 15.0],
            [-0.2, 0.2, 20.0],
            [-0.3, 0.3, 34.0],
            [-0.3, 0.3, 40.0],
            [-0.4, 0.3, 50.0],
            [-0.5, 0.2, 55.0],
            [-0.6, 0.2, 75.0],
            [-0.7, 0.3, 86.0],
        ], dtype=np.float64)
        
    # Ensure the path is valid (finite, correct shape)
    if not np.all(np.isfinite(final_path)) or len(final_path) == 0:
        final_path = np.array([
            [-0.1, 0.1, 15.0],
            [-0.2, 0.2, 20.0],
            [-0.3, 0.3, 34.0],
            [-0.3, 0.3, 40.0],
            [-0.4, 0.3, 50.0],
            [-0.5, 0.2, 55.0],
            [-0.6, 0.2, 75.0],
            [-0.7, 0.3, 86.0],
        ], dtype=np.float64)
        
    return final_path

# EVOLVE-BLOCK-END
