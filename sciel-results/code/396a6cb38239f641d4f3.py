import numpy as np
import time
from scipy.optimize import differential_evolution

def moving_sofa_2d() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 570  # Safety margin

    # Grid definition
    xs = np.arange(-3.0, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    n_points = X_flat.size

    # Start and End states added by evaluator
    P_start = np.array([0.0, 0.0, 0.0])
    P_end = np.array([-2.0, 0.0, 90.0])

    def densify_path(path):
        """
        Takes user path (n, 3), prepends start, appends end.
        Interpolates linearly.
        Returns array of (x, y, a) arrays.
        """
        full_path = np.vstack([P_start, path, P_end])
        n = len(full_path)
        
        # Pre-allocate list is safer for variable sizes, but we can estimate
        # Max steps per segment is ~90. Total segments n-1.
        # Let's just use a list append for simplicity, it's fast enough in Python for small n
        dense_points = [full_path[0]]
        for i in range(n - 1):
            p1 = full_path[i]
            p2 = full_path[i+1]
            delta_a = abs(p2[2] - p1[2])
            m = max(1, int(np.ceil(delta_a)))
            if m == 1:
                dense_points.append(p2)
            else:
                # Vectorized generation of interpolation points for this segment
                # t in [1/m, ..., 1]
                ts = np.arange(1, m + 1) / m
                # p_int = p1 + ts[:, None] * (p2 - p1)[None, :]
                p_ints = p1[None, :] + ts[:, None] * (p2 - p1)[None, :]
                dense_points.extend([p_ints[j] for j in range(len(p_ints))])
        
        return np.array(dense_points)

    def check_inside(placements):
        """
        placements: array of shape (M, 3)
        Returns count of grid points inside ALL placements.
        """
        # Start with all True
        valid_mask = np.ones(n_points, dtype=bool)
        
        for x, y, a_deg in placements:
            a_rad = np.deg2rad(a_deg)
            cos_a = np.cos(a_rad)
            sin_a = np.sin(a_rad)
            
            dx = X_flat - x
            dy = Y_flat - y
            
            # R(-a) * (dx, dy)
            u = dx * cos_a + dy * sin_a
            v = -dx * sin_a + dy * cos_a
            
            # Base corridor:
            # Union of {0 <= u <= 1, -3 <= v <= 1} and {-3 <= u <= 1, 0 <= v <= 1}
            
            in_rect1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
            in_rect2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
            inside = in_rect1 | in_rect2
            
            valid_mask &= inside
            
            # Early exit if no points remain
            if not np.any(valid_mask):
                return 0

        return np.count_nonzero(valid_mask)

    def score_path(path):
        dense = densify_path(path)
        count = check_inside(dense)
        return count * 0.0001

    # Objective function for Differential Evolution
    # Returns negative score (to minimize)
    def objective(params):
        # params layout:
        # [x0, y0, a0, x1, y1, a1, ..., xN-1, yN-1, aN-1]
        n_pts = len(params) // 3
        path = np.zeros((n_pts, 3))
        path[:, 0] = params[0::3]
        path[:, 1] = params[1::3]
        path[:, 2] = params[2::3]
        
        # Ensure angles are within reasonable bounds and sorted?
        # DE might produce unsorted angles, but the interpolator handles that.
        # However, for the sofa problem, angles generally should go 0 -> 90.
        # We'll rely on bounds and DE's exploration, but sorting is usually better 
        # to avoid backtracking in angle.
        # Let's sort the path by angle to ensure monotonic rotation, which is physically sensible.
        if n_pts > 1:
            path = path[np.argsort(path[:, 2])]
        
        s = score_path(path)
        return -s

    # Define number of waypoints
    # More waypoints allow more complex paths but increase optimization difficulty.
    # 5 to 8 seems good.
    n_waypoints = 6
    
    # Bounds for each parameter: [xmin, xmax] for x, y, a
    # x range: corridor goes from x=0 to x=-2. Path x should be between -1.5 and 0.5?
    # y range: corridor goes from y=-1.5 to 1.5. Path y between -0.5 and 1.5?
    # Let's be generous: x in [-2.0, 0.5], y in [-1.0, 2.0], a in [0, 90]
    
    bounds = []
    for _ in range(n_waypoints):
        bounds.append((-2.0, 0.5))   # x
        bounds.append((-1.0, 2.0))   # y
        bounds.append((0.0, 90.0))   # a

    # Seed the RNG for reproducibility in DE
    rng = np.random.default_rng(42)
    
    best_score = -np.inf
    best_params = None
    best_path = None

    # Initial heuristic path
    seed_path = np.array([
        [-0.2, 0.2, 15.0],
        [-0.4, 0.4, 30.0],
        [-0.6, 0.5, 45.0],
        [-0.8, 0.5, 60.0],
        [-1.0, 0.4, 75.0],
        [-1.2, 0.3, 85.0],
    ])
    
    # If seed has different length than n_waypoints, we can't use it directly for DE.
    # Let's just use DE with random init, or seed the DE with a good guess.
    # DE takes `init` parameter. If we don't provide, it's random.
    
    # Run Differential Evolution
    # maxiter and popsize control complexity.
    # popsize 15 is standard. maxiter 100-200.
    # tol controls stopping.
    
    try:
        # Use a time-based stopping? DE doesn't natively support time budget easily in older scipy,
        # but we can control maxiter.
        # Let's estimate iterations.
        # Each objective call is O(M * N_points). M ~ 100 placements, N ~ 150k.
        # 100 * 150k = 15M ops. Very fast (< 10ms).
        # popsize=20, maxiter=200 -> 4000 evals. 4000 * 10ms = 40s. 
        # We have ~500s. So we can do much more.
        
        # Let's increase population size and max iterations to use the time budget.
        # popsize = 40, maxiter = 200 -> 8000 evals.
        
        de_result = differential_evolution(
            objective,
            bounds,
            maxiter=150,
            popsize=30,
            tol=1e-4,
            seed=42,
            init='soles',  # 'soles' can be more robust, but 'latinhypercube' is standard. Let's use 'latinhypercube'
            # Actually 'soles' uses a more even distribution. Let's stick to default or LH.
            # Let's use 'latinhypercube'
        )
        
        if de_result.success or de_result.fun < -best_score:
            best_score = -de_result.fun
            best_params = de_result.x
    except Exception as e:
        print(f"DE failed: {e}")
        # Fallback: use seed path if DE fails completely
        if n_waypoints == len(seed_path):
            best_score = score_path(seed_path)
            # Construct params from seed
            best_params = seed_path.flatten()

    # Reconstruct path from best_params
    n_pts_out = len(best_params) // 3
    best_path = np.zeros((n_pts_out, 3))
    best_path[:, 0] = best_params[0::3]
    best_path[:, 1] = best_params[1::3]
    best_path[:, 2] = best_params[2::3]
    best_path = best_path[np.argsort(best_path[:, 2])]

    # Local refinement with Nelder-Mead?
    # DE is usually pretty good. NM can polish.
    # Let's try a quick NM if time remains.
    
    if time.time() < deadline - 20:
        try:
            from scipy.optimize import minimize
            
            nm_result = minimize(
                objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 50, 'xatol': 1e-5, 'fatol': 1e-6}
            )
            if nm_result.fun < -best_score:
                best_score = -nm_result.fun
                best_params = nm_result.x
                # Reconstruct path
                n_pts_out = len(best_params) // 3
                best_path = np.zeros((n_pts_out, 3))
                best_path[:, 0] = best_params[0::3]
                best_path[:, 1] = best_params[1::3]
                best_path[:, 2] = best_params[2::3]
                best_path = best_path[np.argsort(best_path[:, 2])]
        except Exception as e:
            pass

    # Final validation and return
    # Ensure path is valid
    if best_path is None or len(best_path) == 0:
        return np.array([[0.0, 0.0, 0.0]])
        
    return best_path
