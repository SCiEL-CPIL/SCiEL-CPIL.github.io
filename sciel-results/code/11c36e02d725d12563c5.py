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
    deadline = start_time + 575.0  # Safety margin
    
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
        
        a_rad = np.deg2rad(dense[:, 2])
        cos_a = np.cos(a_rad)
        sin_a = np.sin(a_rad)
        
        inside_all = np.ones(N, dtype=bool)
        
        batch_size = max(1, min(n_dense, 50))
        
        for batch_start in range(0, n_dense, batch_size):
            batch_end = min(batch_start + batch_size, n_dense)
            batch = dense[batch_start:batch_end]
            batch_cos = cos_a[batch_start:batch_end]
            batch_sin = sin_a[batch_start:batch_end]
            
            wx = gx[np.newaxis, :] - batch[:, 0][:, np.newaxis]
            wy = gy[np.newaxis, :] - batch[:, 1][:, np.newaxis]
            
            ul = batch_cos[:, np.newaxis] * wx + batch_sin[:, np.newaxis] * wy
            vl = -batch_sin[:, np.newaxis] * wx + batch_cos[:, np.newaxis] * wy
            
            in_r1 = (ul >= 0.0) & (ul <= 1.0) & (vl >= -3.0) & (vl <= 1.0)
            in_r2 = (ul >= -3.0) & (ul <= 1.0) & (vl >= 0.0) & (vl <= 1.0)
            in_base = in_r1 | in_r2
            
            inside_all &= in_base.all(axis=0)
            
            if not np.any(inside_all):
                break
        
        return np.sum(inside_all) * 0.0001

    def smooth_obj_and_grad(params, tau=0.03):
        """
        Smooth surrogate objective using log-sum-exp soft-min over placements.
        Returns (objective_value, gradient).
        Objective: -sum_over_grid_points(softmin_over_placements(margin))
        We minimize this (equivalently maximize the soft-min sum).
        """
        n_path = len(params) // 3
        path = params.reshape(n_path, 3)
        dense = densify_path(path)
        n_dense = len(dense)
        
        a_rad = np.deg2rad(dense[:, 2])
        cos_a = np.cos(a_rad)
        sin_a = np.sin(a_rad)
        
        # Collect margins for all (placement, grid_point) pairs
        # margin > 0 means inside the base corridor
        all_margins = []
        
        batch_size = max(1, min(n_dense, 25))
        
        for batch_start in range(0, n_dense, batch_size):
            batch_end = min(batch_start + batch_size, n_dense)
            batch = dense[batch_start:batch_end]
            batch_cos = cos_a[batch_start:batch_end]
            batch_sin = sin_a[batch_start:batch_end]
            
            wx = gx[np.newaxis, :] - batch[:, 0][:, np.newaxis]
            wy = gy[np.newaxis, :] - batch[:, 1][:, np.newaxis]
            
            ul = batch_cos[:, np.newaxis] * wx + batch_sin[:, np.newaxis] * wy
            vl = -batch_sin[:, np.newaxis] * wx + batch_cos[:, np.newaxis] * wy
            
            # Margin for each rectangle (distance to boundary, negative if outside)
            margin_r1 = np.minimum(
                np.minimum(ul, 1.0 - ul),
                np.minimum(vl + 3.0, 1.0 - vl)
            )
            margin_r2 = np.minimum(
                np.minimum(ul + 3.0, 1.0 - ul),
                np.minimum(vl, 1.0 - vl)
            )
            
            # Point is inside if max(margin_r1, margin_r2) > 0
            margin = np.maximum(margin_r1, margin_r2)
            # margin shape: (batch_size, N)
            
            all_margins.append(margin)
        
        # Stack: shape (n_dense, N)
        margins = np.concatenate(all_margins, axis=0)
        
        # Soft-min over placements (axis 0) for each grid point
        # softmin(m) = -tau * logsumexp(-m/tau)
        # We want to maximize sum of softmins, so minimize negative
        neg_margins_div_tau = -margins / tau
        # For numerical stability, subtract max
        max_val = np.max(neg_margins_div_tau, axis=0, keepdims=True)
        logsumexp_vals = max_val + np.log(np.sum(np.exp(neg_margins_div_tau - max_val), axis=0))
        softmin_vals = -tau * logsumexp_vals  # shape (N,)
        
        # Objective: minimize -sum(softmin_vals) * 0.0001
        obj_value = -np.sum(softmin_vals) * 0.0001
        
        # Gradient computation
        # d(softmin)/d(margins[i,j]) = -exp(-margins[i,j]/tau) / sum_k exp(-margins[k,j]/tau)
        # = -exp(neg_margins_div_tau[i,j] - logsumexp_vals[j])
        # Then chain rule through margins to params
        
        # This is complex. For L-BFGS-B, let's just return the objective
        # and let it use numerical gradients (approx_grad)
        # But that's slow. Let's use a simpler approach:
        # Just return the hard score and use L-BFGS-B with approx_grad
        # The smooth objective is too expensive for analytical gradients
        
        return obj_value

    def objective(params):
        """Objective for optimization: minimize negative score."""
        n_path = len(params) // 3
        path = params.reshape(n_path, 3)
        return -score_path_fast(path)

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
    
    initial_paths = [
        best_path,
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
        if time.time() > deadline - 60.0:
            break
        s = score_path_fast(init_path)
        if s > best_score:
            best_score = s
            best_path = init_path.copy()
    
    # L-BFGS-B optimization
    if time.time() < deadline - 30.0:
        for init_path in initial_paths:
            if time.time() > deadline - 20.0:
                break
            
            n_path = init_path.shape[0]
            x0 = init_path.ravel().copy()
            
            try:
                result = minimize(
                    objective,
                    x0,
                    method='L-BFGS-B',
                    options={'maxiter': 30, 'maxfun': 100, 'gtol': 1e-2, 'disp': False}
                )
                
                new_path = result.x.reshape(n_path, 3)
                new_score = score_path_fast(new_path)
                if new_score > best_score:
                    best_path = new_path
                    best_score = new_score
            except Exception:
                pass
    
    # Final coordinate descent polish
    if time.time() < deadline - 10.0:
        current_path = best_path.copy()
        current_score = best_score
        n_points = current_path.shape[0]
        
        step_size = 0.05
        max_iter = 100
        iter_count = 0
        no_improve_count = 0
        
        while iter_count < max_iter and time.time() < deadline - 5.0:
            iter_count += 1
            improved = False
            
            for i in range(n_points):
                for dx in [-step_size, step_size]:
                    new_path = current_path.copy()
                    new_path[i, 0] += dx
                    s = score_path_fast(new_path)
                    if s > current_score + 1e-10:
                        current_path = new_path
                        current_score = s
                        improved = True
                
                for dy in [-step_size, step_size]:
                    new_path = current_path.copy()
                    new_path[i, 1] += dy
                    s = score_path_fast(new_path)
                    if s > current_score + 1e-10:
                        current_path = new_path
                        current_score = s
                        improved = True
                
                for da in [-step_size, step_size]:
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
                    if step_size < 1e-4:
                        break
                no_improve_count = 0
        
        if current_score > best_score:
            best_path = current_path
            best_score = current_score
    
    return best_path

# EVOLVE-BLOCK-END
