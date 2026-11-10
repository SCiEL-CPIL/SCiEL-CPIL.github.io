# EVOLVE-BLOCK-START
import numpy as np
import time

def moving_sofa_2d() -> np.ndarray:
    """
    Finds a path for moving a sofa through an L-shaped corridor.
    Returns a float array of shape (n, 3) where each row is (x, y, angle_degrees).
    """
    start_time = time.time()
    deadline = start_time + 580.0  # Keep margin from 630s

    # Grid points for sofa area calculation
    u = np.arange(-3.0, 1.5, 0.01)
    v = np.arange(-1.5, 1.5, 0.01)
    gu, gv = np.meshgrid(u, v, indexing='ij')
    gx, gy = gu.ravel(), gv.ravel()
    N = len(gx)

    def densify_path(path):
        """
        Prepend (0,0,0), append (-2,0,90), then insert linear interpolations.
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

    def score_path_hard(path):
        """
        Compute the raw score (area).
        Optimized for speed: vectorized checks.
        """
        dense = densify_path(path)
        n_dense = len(dense)
        
        a_rad = np.deg2rad(dense[:, 2])
        cos_a = np.cos(a_rad)
        sin_a = np.sin(a_rad)
        
        inside_all = np.ones(N, dtype=bool)
        
        # Process in larger batches to reduce loop overhead, but keep memory safe
        # N=135000, n_dense~100-200. 135000*200*8 bytes = 216MB. Too big for one go.
        # Batch size 20 placements -> 135000*20*8 = 21MB. Safe.
        batch_size = 20
        
        for batch_start in range(0, n_dense, batch_size):
            batch_end = min(batch_start + batch_size, n_dense)
            b_cos = cos_a[batch_start:batch_end]
            b_sin = sin_a[batch_start:batch_end]
            b_x = dense[batch_start:batch_end, 0]
            b_y = dense[batch_start:batch_end, 1]
            
            # Vectorized rotation and check
            wx = gx - b_x[:, np.newaxis]
            wy = gy - b_y[:, np.newaxis]
            
            ul = b_cos[:, np.newaxis] * wx + b_sin[:, np.newaxis] * wy
            vl = -b_sin[:, np.newaxis] * wx + b_cos[:, np.newaxis] * wy
            
            in_r1 = (ul >= 0.0) & (ul <= 1.0) & (vl >= -3.0) & (vl <= 1.0)
            in_r2 = (ul >= -3.0) & (ul <= 1.0) & (vl >= 0.0) & (vl <= 1.0)
            in_base = in_r1 | in_r2
            
            inside_all &= in_base.all(axis=0)
            
            if not np.any(inside_all):
                break
        
        return np.sum(inside_all) * 0.0001

    # Initial paths
    initial_paths = [
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
        np.array([
            [-0.1, 0.1, 15.0],
            [-0.2, 0.2, 20.0],
            [-0.3, 0.3, 34.0],
            [-0.3, 0.3, 40.0],
            [-0.4, 0.3, 50.0],
            [-0.5, 0.2, 55.0],
            [-0.6, 0.2, 75.0],
            [-0.7, 0.3, 86.0]
        ])
    ]
    
    best_path = initial_paths[0]
    best_hard_score = score_path_hard(best_path)
    
    # Coordinate Ascent Optimization
    # This is robust and easy to time-bound.
    current_path = best_path.copy()
    current_score = best_hard_score
    n_points = current_path.shape[0]
    
    step_size = 0.1
    max_iter = 100
    
    for iter_idx in range(max_iter):
        if time.time() > deadline - 10.0:
            break
            
        improvement_found = False
        
        for i in range(n_points):
            if time.time() > deadline - 5.0:
                break
                
            # Try small perturbations in x, y, a
            # We can try multiple steps or just +/-
            for dim in range(3):
                orig_val = current_path[i, dim]
                
                # Try +step
                new_path = current_path.copy()
                new_path[i, dim] = orig_val + step_size
                if dim == 2:
                    new_path[i, dim] = np.clip(new_path[i, dim], 0, 90)
                
                s = score_path_hard(new_path)
                if s > current_score:
                    current_path = new_path
                    current_score = s
                    improvement_found = True
                    
                # Try -step
                new_path = current_path.copy()
                new_path[i, dim] = orig_val - step_size
                if dim == 2:
                    new_path[i, dim] = np.clip(new_path[i, dim], 0, 90)
                
                s = score_path_hard(new_path)
                if s > current_score:
                    current_path = new_path
                    current_score = s
                    improvement_found = True
        
        if not improvement_found:
            step_size *= 0.8
            if step_size < 0.005:
                break
        else:
            step_size *= 0.95
            
        if current_score > best_hard_score:
            best_path = current_path
            best_hard_score = current_score

    return best_path

# EVOLVE-BLOCK-END
