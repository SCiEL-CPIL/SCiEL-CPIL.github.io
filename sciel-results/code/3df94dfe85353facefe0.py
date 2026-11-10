# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def moving_sofa_2d() -> np.ndarray:
    """
    Finds a path for moving a sofa through an L-shaped corridor.
    Returns a float array of shape (n, 3) where each row is (x, y, angle_degrees).
    """
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin

    # Grid points for sofa area calculation
    u = np.arange(-3.0, 1.5, 0.01)
    v = np.arange(-1.5, 1.5, 0.01)
    gu, gv = np.meshgrid(u, v, indexing='ij')
    grid_points = np.column_stack((gu.ravel(), gv.ravel()))
    N = grid_points.shape[0]
    gx = grid_points[:, 0]
    gy = grid_points[:, 1]

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
        
        inside_all = np.ones(N, dtype=bool)
        
        # Process in batches to avoid memory issues
        # 50000 points * 2 floats * 8 bytes * 50 batches = 40MB, safe
        batch_size = 50
        
        for batch_start in range(0, n_dense, batch_size):
            batch_end = min(batch_start + batch_size, n_dense)
            batch_cos = cos_a[batch_start:batch_end]
            batch_sin = sin_a[batch_start:batch_end]
            batch_x = dense[batch_start:batch_end, 0]
            batch_y = dense[batch_start:batch_end, 1]
            
            # Transform: ul = cos(wx) + sin(wy), vl = -sin(wx) + cos(wy)
            # wx, wy: (b, N)
            wx = gx[np.newaxis, :] - batch_x[:, np.newaxis]
            wy = gy[np.newaxis, :] - batch_y[:, np.newaxis]
            
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

    # Initial path
    best_path = np.array([
        [-0.1, 0.1, 15.0],
        [-0.2, 0.2, 20.0],
        [-0.3, 0.3, 34.0],
        [-0.3, 0.3, 40.0],
        [-0.4, 0.3, 50.0],
        [-0.5, 0.2, 55.0],
        [-0.6, 0.2, 75.0],
        [-0.7, 0.3, 86.0]
    ])
    
    best_score = score_path_fast(best_path)
    
    # Optimization using scipy's L-BFGS-B
    # Flatten the path for optimization
    flat_path = best_path.flatten()
    
    def objective(flat_params):
        if time.time() > deadline - 10.0:
            return 0.0
        p = flat_params.reshape(-1, 3)
        # Ensure angles are in range [0, 90]
        p[:, 2] = np.clip(p[:, 2], 0, 90)
        s = score_path_fast(p)
        return -s  # Minimize negative score

    try:
        res = minimize(objective, flat_path, method='L-BFGS-B', 
                       options={'maxiter': 200, 'gtol': 1e-6, 'ftol': 1e-8})
        if res.fun < -best_score:
            best_path = res.x.reshape(-1, 3)
            best_score = -res.fun
    except Exception:
        pass
    
    # Local polish with coordinate ascent
    if time.time() < deadline - 20.0:
        current_path = best_path.copy()
        current_score = best_score
        n_points = current_path.shape[0]
        
        step_size = 0.05
        improvement_found = True
        max_polish_iter = 50
        polish_iter = 0
        
        while improvement_found and polish_iter < max_polish_iter:
            if time.time() > deadline - 10.0:
                break
            improvement_found = False
            polish_iter += 1
            
            for i in range(n_points):
                # Try x perturbation
                for dx in [-step_size, step_size]:
                    new_path = current_path.copy()
                    new_path[i, 0] += dx
                    s = score_path_fast(new_path)
                    if s > current_score:
                        current_path = new_path
                        current_score = s
                        improvement_found = True
                # Try y perturbation
                for dy in [-step_size, step_size]:
                    new_path = current_path.copy()
                    new_path[i, 1] += dy
                    s = score_path_fast(new_path)
                    if s > current_score:
                        current_path = new_path
                        current_score = s
                        improvement_found = True
                # Try angle perturbation
                for da in [-step_size*5, step_size*5]:
                    new_path = current_path.copy()
                    new_a = current_path[i, 2] + da
                    if 0 <= new_a <= 90:
                        new_path[i, 2] = new_a
                        s = score_path_fast(new_path)
                        if s > current_score:
                            current_path = new_path
                            current_score = s
                            improvement_found = True
            
            # If no improvement, reduce step size
            if not improvement_found:
                step_size *= 0.7
                if step_size < 1e-4:
                    break
            
        if current_score > best_score:
            best_path = current_path
            best_score = current_score

    return best_path

# EVOLVE-BLOCK-END
