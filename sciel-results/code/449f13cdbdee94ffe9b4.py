# EVOLVE-BLOCK-START
import numpy as np
import time

def moving_sofa_2d() -> np.ndarray:
    """
    Finds a path for moving a sofa through an L-shaped corridor.
    Returns a float array of shape (n, 3) where each row is (x, y, angle_degrees).
    """
    start_time = time.time()
    deadline = start_time + 590.0  # Safety margin
    
    # Grid points for sofa area calculation
    u = np.arange(-3.0, 1.5, 0.01)
    v = np.arange(-1.5, 1.5, 0.01)
    gu, gv = np.meshgrid(u, v, indexing='ij')
    grid_points = np.column_stack((gu.ravel(), gv.ravel()))
    N = grid_points.shape[0]
    
    def densify_path(path):
        """
        Prepend (0,0,0), append (-2,0,90), then insert linear interpolations
        between consecutive rows so that each segment spans <= 1 degree of rotation.
        """
        start = np.array([0.0, 0.0, 0.0])
        end = np.array([-2.0, 0.0, 90.0])
        full = np.vstack(([start], path, [end]))
        
        dense_points = [full[0]]
        for i in range(len(full) - 1):
            p1 = full[i]
            p2 = full[i + 1]
            da = p2[2] - p1[2]
            m = max(1, int(np.ceil(abs(da))))
            
            if m > 1:
                ts = np.linspace(1.0 / m, 1.0 - 1.0 / m, m - 1)
                new_pts = p1 + ts[:, np.newaxis] * (p2 - p1)
                dense_points.extend(new_pts)
            
            dense_points.append(p2)
        
        return np.array(dense_points)
    
    def score_path_fast(path):
        """
        Compute the area of the sofa defined by the path using vectorized operations.
        """
        dense = densify_path(path)
        n_dense = len(dense)
        
        # Precompute rotation matrices for all dense points
        a_rad = np.deg2rad(dense[:, 2])
        cos_a = np.cos(a_rad)
        sin_a = np.sin(a_rad)
        
        # For each grid point, check if it's inside all placements
        # Vectorize: for each dense placement, compute transformed coords for all grid points
        # Then check base corridor condition
        
        inside_all = np.ones(N, dtype=bool)
        
        # Process in batches to avoid memory issues
        batch_size = max(1, min(n_dense, 50))
        
        for batch_start in range(0, n_dense, batch_size):
            batch_end = min(batch_start + batch_size, n_dense)
            batch = dense[batch_start:batch_end]
            batch_cos = cos_a[batch_start:batch_end]
            batch_sin = sin_a[batch_start:batch_end]
            
            # For each point in batch, transform grid points
            # batch shape: (b, 3), grid_points shape: (N, 2)
            # wx, wy: (b, N)
            wx = grid_points[:, 0][np.newaxis, :] - batch[:, 0][:, np.newaxis]
            wy = grid_points[:, 1][np.newaxis, :] - batch[:, 1][:, np.newaxis]
            
            # Transform: ul = cos(wx) + sin(wy), vl = -sin(wx) + cos(wy)
            # batch_cos, batch_sin shape: (b,)
            ul = batch_cos[:, np.newaxis] * wx + batch_sin[:, np.newaxis] * wy
            vl = -batch_sin[:, np.newaxis] * wx + batch_cos[:, np.newaxis] * wy
            
            # Check base corridor: (0<=ul<=1, -3<=vl<=1) OR (-3<=ul<=1, 0<=vl<=1)
            in_r1 = (ul >= 0.0) & (ul <= 1.0) & (vl >= -3.0) & (vl <= 1.0)
            in_r2 = (ul >= -3.0) & (ul <= 1.0) & (vl >= 0.0) & (vl <= 1.0)
            in_base = in_r1 | in_r2
            
            # A point is inside all placements in this batch if it's inside all of them
            # in_base shape: (b, N), we need all True along axis 0
            inside_all &= in_base.all(axis=0)
            
            # Early exit if no points remain
            if not np.any(inside_all):
                break
        
        return np.sum(inside_all) * 0.0001
    
    # Initial path - try a smooth rotation
    best_path = np.array([
        [-0.05, 0.05, 10.0],
        [-0.1, 0.1, 20.0],
        [-0.15, 0.15, 30.0],
        [-0.2, 0.2, 40.0],
        [-0.25, 0.25, 50.0],
        [-0.3, 0.3, 60.0],
        [-0.4, 0.4, 70.0],
        [-0.5, 0.5, 80.0],
        [-0.6, 0.6, 88.0]
    ])
    
    best_score = score_path_fast(best_path)
    
    # Use Nelder-Mead for optimization
    def objective(flat_params):
        if time.time() > deadline - 15.0:
            return 0.0
        p = flat_params.reshape(-1, 3)
        # Clamp angles to [0, 90]
        p[:, 2] = np.clip(p[:, 2], 0, 90)
        s = score_path_fast(p)
        return -s  # Minimize negative score
    
    # Nelder-Mead doesn't support bounds, so we'll use a penalty approach
    # Or we can just rely on the clamping
    
    flat_path = best_path.flatten()
    
    # Use a simple iterative improvement strategy
    # Try different initial paths and keep the best
    
    initial_paths = [
        # Current best
        best_path,
        # More aggressive rotation
        np.array([
            [-0.1, 0.1, 15.0],
            [-0.2, 0.2, 25.0],
            [-0.3, 0.3, 35.0],
            [-0.4, 0.4, 45.0],
            [-0.5, 0.5, 55.0],
            [-0.6, 0.6, 65.0],
            [-0.7, 0.7, 75.0],
            [-0.8, 0.8, 85.0]
        ]),
        # Less aggressive
        np.array([
            [0.0, 0.0, 5.0],
            [-0.05, 0.05, 15.0],
            [-0.1, 0.1, 25.0],
            [-0.15, 0.15, 35.0],
            [-0.2, 0.2, 45.0],
            [-0.3, 0.3, 55.0],
            [-0.4, 0.4, 65.0],
            [-0.5, 0.5, 75.0],
            [-0.6, 0.6, 85.0]
        ]),
        # Different trajectory
        np.array([
            [-0.05, 0.0, 10.0],
            [-0.1, 0.05, 20.0],
            [-0.15, 0.1, 30.0],
            [-0.2, 0.15, 40.0],
            [-0.3, 0.2, 50.0],
            [-0.4, 0.25, 60.0],
            [-0.5, 0.3, 70.0],
            [-0.6, 0.35, 80.0],
            [-0.7, 0.4, 88.0]
        ])
    ]
    
    for init_path in initial_paths:
        if time.time() > deadline - 30.0:
            break
        s = score_path_fast(init_path)
        if s > best_score:
            best_score = s
            best_path = init_path
    
    # Local optimization with coordinate descent
    if time.time() < deadline - 20.0:
        current_path = best_path.copy()
        current_score = best_score
        n_points = current_path.shape[0]
        
        step_size = 0.1
        max_iter = 200
        iter_count = 0
        no_improve_count = 0
        
        while iter_count < max_iter and time.time() < deadline - 15.0:
            iter_count += 1
            improved = False
            
            for i in range(n_points):
                # Try x perturbation
                for dx in [-step_size, step_size]:
                    new_path = current_path.copy()
                    new_path[i, 0] += dx
                    s = score_path_fast(new_path)
                    if s > current_score + 1e-10:
                        current_path = new_path
                        current_score = s
                        improved = True
                
                # Try y perturbation
                for dy in [-step_size, step_size]:
                    new_path = current_path.copy()
                    new_path[i, 1] += dy
                    s = score_path_fast(new_path)
                    if s > current_score + 1e-10:
                        current_path = new_path
                        current_score = s
                        improved = True
                
                # Try angle perturbation
                for da in [-step_size*2, step_size*2]:
                    new_a = current_path[i, 2] + da
                    if 0 <= new_a <= 90:
                        new_path = current_path.copy()
                        new_path[i, 2] = new_a
                        s = score_path_fast(new_path)
                        if s > current_score + 1e-10:
                            current_path = new_path
                            current_score = s
                            improved = True
            
            if improved:
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= 3:
                    step_size *= 0.7
                    if step_size < 1e-3:
                        break
                no_improve_count = 0
        
        if current_score > best_score:
            best_path = current_path
            best_score = current_score
    
    return best_path

# EVOLVE-BLOCK-END
