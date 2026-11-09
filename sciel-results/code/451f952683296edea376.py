import numpy as np
from scipy.optimize import differential_evolution
import time

# Pre-compute grid
# i = 0..449, j = 0..299
# x = -3 + 0.01 * i
# y = -1.5 + 0.01 * j
# Note: The prompt says np.arange(-3, 1.5, 0.01) which gives 450 points.
# np.arange(-1.5, 1.5, 0.01) which gives 300 points.
# Total 135,000 points.

def moving_sofa_2d() -> np.ndarray:
    start_time = time.time()
    deadline = 600.0
    safety_margin = 10.0
    
    # Grid
    xs = np.arange(-3, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    XX, YY = np.meshgrid(xs, ys, indexing='xy')
    W = np.column_stack((XX.ravel(), YY.ravel()))
    N = W.shape[0]
    
    # Base corridor mask function
    def in_base_corridor(W_rot):
        u = W_rot[:, 0]
        v = W_rot[:, 1]
        # Part 1: 0 <= u <= 1, -3 <= v <= 1
        mask1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
        # Part 2: -3 <= u <= 1, 0 <= v <= 1
        mask2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
        return mask1 | mask2
    
    def densify_path(path):
        # path: (K, 3)
        # Full path with endpoints
        full_path = np.vstack([
            [[0.0, 0.0, 0.0]],
            path,
            [[-2.0, 0.0, 90.0]]
        ])
        
        dense_path = [full_path[0]]
        for i in range(len(full_path) - 1):
            p1 = full_path[i]
            p2 = full_path[i+1]
            da = abs(p2[2] - p1[2])
            m = max(1, int(np.ceil(da)))
            if m > 1:
                # m segments -> m-1 intermediate points
                ts = np.linspace(0, 1, m+1)[1:-1]
                for t in ts:
                    p_interp = p1 + t * (p2 - p1)
                    dense_path.append(p_interp)
            dense_path.append(p2)
        return np.array(dense_path)

    def compute_area(path):
        dense = densify_path(path)
        # Initialize mask with True
        mask = np.ones(N, dtype=bool)
        
        for p in dense:
            x, y, a_deg = p
            a_rad = np.deg2rad(a_deg)
            # Rotation matrix R(-a)
            c = np.cos(-a_rad)
            s = np.sin(-a_rad)
            R = np.array([[c, -s], [s, c]])
            
            # Translate
            W_trans = W - np.array([x, y])
            # Rotate
            W_rot = W_trans @ R.T
            
            # Check base corridor
            mask_in = in_base_corridor(W_rot)
            
            # Intersection
            mask &= mask_in
            
            # Early exit if mask is empty
            if not np.any(mask):
                return 0.0
        
        return np.sum(mask) * 0.0001

    # Optimization
    # We optimize K intermediate points.
    K = 10
    # Variables: x_1..x_K, y_1..y_K, theta_1..theta_K
    # theta_0 = 0, theta_{K+1} = 90
    # We enforce monotonicity by optimizing deltas?
    # Or just bounds: theta_i in [0, 90] and penalize if not sorted?
    # Better: Optimize theta_i directly with bounds [0, 90] and check sorting.
    # If not sorted, return 0.
    
    def objective(params):
        # params: 3*K
        x = params[0:K]
        y = params[K:2*K]
        theta = params[2*K:3*K]
        
        # Check monotonicity of theta
        # theta should be increasing
        if np.any(np.diff(theta) < 0):
            return -1.0 # Penalty
            
        # Also check bounds relative to start/end?
        # Start 0, End 90.
        # theta[0] should be > 0, theta[-1] < 90.
        if theta[0] <= 0 or theta[-1] >= 90:
            return -1.0
            
        path = np.column_stack((x, y, theta))
        return compute_area(path)

    # Bounds for DE
    # x, y: The corridor moves from (0,0) to (-2,0).
    # The corner should probably stay within a reasonable box.
    # Let's say x in [-4, 2], y in [-4, 4].
    bounds = [(-4.0, 2.0)] * K + [(-4.0, 4.0)] * K + [(0.1, 89.9)] * K
    
    # Initial population
    # Seed path points (scaled to K=10)
    seed_path = np.array([
        [-0.1, 0.1, 15.0],
        [-0.2, 0.2, 20.0],
        [-0.3, 0.3, 34.0],
        [-0.3, 0.3, 40.0],
        [-0.4, 0.3, 50.0],
        [-0.5, 0.2, 55.0],
        [-0.6, 0.2, 75.0],
        [-0.7, 0.3, 86.0],
    ])
    
    # Interpolate seed path to K points
    seed_angles = seed_path[:, 2]
    seed_x = seed_path[:, 0]
    seed_y = seed_path[:, 1]
    
    # We want K points between 0 and 90.
    # The seed has 8 points.
    # Let's just use the seed points if K==8, else interpolate.
    if K == len(seed_path):
        init_x = seed_x
        init_y = seed_y
        init_theta = seed_angles
    else:
        # Linear interpolation of angles?
        # Just create a smooth path
        init_theta = np.linspace(5, 85, K)
        init_x = np.linspace(-0.1, -0.7, K)
        init_y = np.linspace(0.1, 0.3, K)
        
    x0 = np.concatenate([init_x, init_y, init_theta])
    
    # Run DE
    # To save time, we can limit maxiter and popsize
    # And check time in the callback? DE doesn't support easy time-based stopping in all versions.
    # We can run a fixed number of iterations and then check time.
    
    # Let's try 50 iterations, popsize 20.
    # 50 * 20 * 30 * (cost of eval)
    # Cost of eval ~ 100 placements * 135k points = 13.5M ops.
    # 13.5M ops is ~ 0.05s in Python/NumPy? Maybe 0.1s.
    # 1000 evals * 0.1s = 100s.
    # So 50 iters * 20 pop = 1000 evals.
    # This should fit in 600s.
    
    try:
        result = differential_evolution(
            objective, 
            bounds, 
            x0=x0, 
            seed=42, 
            maxiter=50, 
            popsize=20,
            tol=0.01,
            mutation=(0.5, 1.0),
            recombination=0.7,
            polish=False # Polish is expensive and might fail due to non-smoothness
        )
    except Exception as e:
        # Fallback to seed
        return seed_path if K == len(seed_path) else np.column_stack((init_x, init_y, init_theta))

    if result.success or result.fun > 0:
        best_params = result.x
        best_x = best_params[0:K]
        best_y = best_params[K:2*K]
        best_theta = best_params[2*K:3*K]
        best_path = np.column_stack((best_x, best_y, best_theta))
    else:
        # If DE failed, use the seed or initial guess
        best_path = np.column_stack((init_x, init_y, init_theta))
        
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline - safety_margin:
        # Return best found so far
        pass
        
    return best_path

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
import time

# Pre-compute grid
# i = 0..449, j = 0..299
# x = -3 + 0.01 * i
# y = -1.5 + 0.01 * j

def moving_sofa_2d() -> np.ndarray:
    start_time = time.time()
    deadline = 600.0
    safety_margin = 10.0
    
    # Grid
    xs = np.arange(-3, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    XX, YY = np.meshgrid(xs, ys, indexing='xy')
    W = np.column_stack((XX.ravel(), YY.ravel()))
    N = W.shape[0]
    
    # Base corridor mask function
    def in_base_corridor(W_rot):
        u = W_rot[:, 0]
        v = W_rot[:, 1]
        # Part 1: 0 <= u <= 1, -3 <= v <= 1
        mask1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
        # Part 2: -3 <= u <= 1, 0 <= v <= 1
        mask2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
        return mask1 | mask2
    
    def densify_path(path):
        # path: (K, 3)
        # Full path with endpoints
        full_path = np.vstack([
            [[0.0, 0.0, 0.0]],
            path,
            [[-2.0, 0.0, 90.0]]
        ])
        
        dense_path = [full_path[0]]
        for i in range(len(full_path) - 1):
            p1 = full_path[i]
            p2 = full_path[i+1]
            da = abs(p2[2] - p1[2])
            m = max(1, int(np.ceil(da)))
            if m > 1:
                # m segments -> m-1 intermediate points
                ts = np.linspace(0, 1, m+1)[1:-1]
                for t in ts:
                    p_interp = p1 + t * (p2 - p1)
                    dense_path.append(p_interp)
            dense_path.append(p2)
        return np.array(dense_path)

    def compute_area(path):
        dense = densify_path(path)
        # Initialize mask with True
        mask = np.ones(N, dtype=bool)
        
        for p in dense:
            x, y, a_deg = p
            a_rad = np.deg2rad(a_deg)
            # Rotation matrix R(-a)
            c = np.cos(-a_rad)
            s = np.sin(-a_rad)
            R = np.array([[c, -s], [s, c]])
            
            # Translate
            W_trans = W - np.array([x, y])
            # Rotate
            W_rot = W_trans @ R.T
            
            # Check base corridor
            mask_in = in_base_corridor(W_rot)
            
            # Intersection
            mask &= mask_in
            
            # Early exit if mask is empty
            if not np.any(mask):
                return 0.0
        
        return np.sum(mask) * 0.0001

    # Optimization
    # We optimize K intermediate points.
    K = 10
    # Variables: x_1..x_K, y_1..y_K, theta_1..theta_K
    # theta_0 = 0, theta_{K+1} = 90
    
    def objective(params):
        # params: 3*K
        x = params[0:K]
        y = params[K:2*K]
        theta = params[2*K:3*K]
        
        # Check monotonicity of theta
        if np.any(np.diff(theta) < 0):
            return -1.0
            
        # Check bounds relative to start/end
        if theta[0] <= 0 or theta[-1] >= 90:
            return -1.0
            
        path = np.column_stack((x, y, theta))
        return compute_area(path)

    # Bounds for DE
    bounds = [(-4.0, 2.0)] * K + [(-4.0, 4.0)] * K + [(0.1, 89.9)] * K
    
    # Initial population
    # Seed path points
    seed_path = np.array([
        [-0.1, 0.1, 15.0],
        [-0.2, 0.2, 20.0],
        [-0.3, 0.3, 34.0],
        [-0.3, 0.3, 40.0],
        [-0.4, 0.3, 50.0],
        [-0.5, 0.2, 55.0],
        [-0.6, 0.2, 75.0],
        [-0.7, 0.3, 86.0],
    ])
    
    # Interpolate seed path to K points
    if K == len(seed_path):
        init_x = seed_path[:, 0]
        init_y = seed_path[:, 1]
        init_theta = seed_path[:, 2]
    else:
        init_theta = np.linspace(5, 85, K)
        init_x = np.linspace(-0.1, -0.7, K)
        init_y = np.linspace(0.1, 0.3, K)
        
    x0 = np.concatenate([init_x, init_y, init_theta])
    
    # Run DE
    try:
        result = differential_evolution(
            objective, 
            bounds, 
            x0=x0, 
            seed=42, 
            maxiter=50, 
            popsize=20,
            tol=0.01,
            mutation=(0.5, 1.0),
            recombination=0.7,
            polish=False
        )
    except Exception:
        return np.column_stack((init_x, init_y, init_theta))

    if result.success or result.fun > 0:
        best_params = result.x
        best_x = best_params[0:K]
        best_y = best_params[K:2*K]
        best_theta = best_params[2*K:3*K]
        best_path = np.column_stack((best_x, best_y, best_theta))
    else:
        best_path = np.column_stack((init_x, init_y, init_theta))
        
    return best_path

# EVOLVE-BLOCK-END
