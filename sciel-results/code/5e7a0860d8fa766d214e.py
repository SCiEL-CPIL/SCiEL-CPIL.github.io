import numpy as np
import time
import scipy.optimize as so

N = 200

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin

    def check_time():
        return time.time() > deadline

    # 1. Initialization: Hexagonal Lattice
    # Estimate spacing
    s = np.sqrt(2 / (np.sqrt(3) * N)) * 1.1  # slightly larger to ensure we have enough points
    # Generate hex grid
    points = []
    # Range of indices
    max_k = int(0.5 / (s * 0.866)) + 2
    for j in range(-max_k, max_k + 1):
        for i in range(-max_k, max_k + 1):
            x = s * (i + 0.5 * (j % 2))
            y = s * j * np.sqrt(3) / 2
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                points.append([x, y])
    
    points = np.array(points)
    
    # If we have more than N, randomly select N
    if len(points) > N:
        indices = np.random.choice(len(points), N, replace=False)
        points = points[indices]
    elif len(points) < N:
        # If too few, add random points inside
        while len(points) < N:
            p = np.random.uniform(-0.5, 0.5, 2)
            points = np.vstack([points, p])
            
    # Add small random noise to break symmetry
    points += np.random.normal(0, 0.001, points.shape)
    # Clip to bounds
    points = np.clip(points, -0.5, 0.5)

    # 2. Optimization
    # Objective: Minimize sum of inverse squared distances (smooth)
    # x is flattened array of shape (2*N,)
    
    def objective_and_grad(x):
        if check_time():
            # Return a large value to signal stopping? 
            # Actually, better to let it finish current eval and check in loop if we were doing manual opt.
            # Since we use scipy, we can't easily break. 
            # We rely on maxiter and ftol.
            pass
            
        pts = x.reshape(N, 2)
        
        # Pairwise distances
        # diff[i, j] = pts[i] - pts[j]
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        
        # Avoid division by zero
        dist_sq = np.maximum(dist_sq, 1e-12)
        
        # Potential: 1 / dist_sq^2
        # We only want i < j
        iu = np.triu_indices(N, 1)
        pair_dist_sq = dist_sq[iu]
        
        # Sum of potentials
        val = np.sum(1.0 / (pair_dist_sq**2))
        
        # Wall potential
        # Distance to walls: 0.5 - |x|, 0.5 - |y|
        # We use 1 / (dist_to_wall + eps)^2
        # To keep it smooth and inside, we can just use the coordinates directly
        # If a point is outside, the potential should be huge.
        # But L-BFGS-B keeps it inside bounds.
        # So we just need to penalize getting close to walls.
        wall_dist_x_pos = 0.5 - pts[:, 0]
        wall_dist_x_neg = 0.5 + pts[:, 0]
        wall_dist_y_pos = 0.5 - pts[:, 1]
        wall_dist_y_neg = 0.5 + pts[:, 1]
        
        # Ensure positive
        wall_dist_x_pos = np.maximum(wall_dist_x_pos, 1e-12)
        wall_dist_x_neg = np.maximum(wall_dist_x_neg, 1e-12)
        wall_dist_y_pos = np.maximum(wall_dist_y_pos, 1e-12)
        wall_dist_y_neg = np.maximum(wall_dist_y_neg, 1e-12)
        
        wall_val = np.sum(1.0 / (wall_dist_x_pos**2) + 
                          1.0 / (wall_dist_x_neg**2) + 
                          1.0 / (wall_dist_y_pos**2) + 
                          1.0 / (wall_dist_y_neg**2))
        
        total_val = val + 10.0 * wall_val  # Weight wall repulsion
        
        # Gradient
        # d/dx_i (1/d^4) = -2 * (x_i - x_j) / d^6
        # Wait, potential is 1/d^4? No, I used 1/dist_sq^2 which is 1/d^4.
        # d(1/d^4)/dx = -4/d^5 * dx/dd = -4/d^5 * (x_i-x_j)/d = -4 (x_i-x_j) / d^6.
        # My dist_sq is d^2. So potential is 1/dist_sq^2.
        # d(1/dist_sq^2)/d(pts[i]) = -2 * (1/dist_sq^3) * d(dist_sq)/d(pts[i])
        # d(dist_sq)/d(pts[i]) = 2 * (pts[i] - pts[j]) for j != i.
        # So grad = -2 * (1/dist_sq^3) * 2 * (pts[i] - pts[j]) = -4 * (pts[i] - pts[j]) / dist_sq^3.
        
        # Vectorized gradient for pairs
        # We need to sum contributions from all j for each i.
        # grad_pts = zeros(N, 2)
        # For each pair (i, j), grad_pts[i] += -4 * (pts[i] - pts[j]) / dist_sq[i, j]^3
        #             grad_pts[j] += -4 * (pts[j] - pts[i]) / dist_sq[i, j]^3 = 4 * (pts[i] - pts[j]) / dist_sq[i, j]^3
        
        # Using the diff matrix
        # diff[i, j] = pts[i] - pts[j]
        # dist_sq[i, j]
        # term = 4 * diff / dist_sq^3
        # But we must exclude diagonal.
        dist_sq_cubed = dist_sq**3
        # Avoid div by zero on diagonal
        dist_sq_cubed = np.where(dist_sq > 1e-12, dist_sq_cubed, 1e-12)
        
        term = 4.0 * diff / dist_sq_cubed[:, :, np.newaxis]
        
        # Sum over j (axis 1)
        # grad_pts[i] = sum_j term[i, j]
        # But term[i, i] is 0/0 handled by where, so it's 0.
        grad_pts = np.sum(term, axis=1)
        
        # Wall gradient
        # d(1/w^2)/dx = -2/w^3 * dw/dx
        # w = 0.5 - x, dw/dx = -1 -> grad = 2/w^3
        # w = 0.5 + x, dw/dx = 1 -> grad = -2/w^3
        
        gw_x_pos = 2.0 * 10.0 / (wall_dist_x_pos**3)
        gw_x_neg = -2.0 * 10.0 / (wall_dist_x_neg**3)
        gw_y_pos = 2.0 * 10.0 / (wall_dist_y_pos**3)
        gw_y_neg = -2.0 * 10.0 / (wall_dist_y_neg**3)
        
        grad_pts[:, 0] += gw_x_pos + gw_x_neg
        grad_pts[:, 1] += gw_y_pos + gw_y_neg
        
        return total_val, grad_pts.flatten()

    x0 = points.flatten()
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    # Run optimization
    try:
        res = so.minimize(objective_and_grad, x0, method='L-BFGS-B', 
                          jac=True, bounds=bounds, 
                          options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
    except Exception:
        # If it fails, return initial
        pass

    x_opt = res.x if hasattr(res, 'x') else x0
    pts_opt = x_opt.reshape(N, 2)
    
    # Ensure validity
    pts_opt = np.clip(pts_opt, -0.5, 0.5)
    
    # Check time
    if check_time():
        return pts_opt
        
    # Optional: Second pass with higher power potential to sharpen min distance
    # But L-BFGS-B on 400 vars is already heavy. 
    # Let's try one more pass with a steeper potential (power 6 instead of 4)
    
    def objective_and_grad_2(x):
        pts = x.reshape(N, 2)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        dist_sq = np.maximum(dist_sq, 1e-12)
        
        iu = np.triu_indices(N, 1)
        pair_dist_sq = dist_sq[iu]
        
        # Power 6: 1/d^12
        val = np.sum(1.0 / (pair_dist_sq**3))
        
        wall_dist_x_pos = np.maximum(0.5 - pts[:, 0], 1e-12)
        wall_dist_x_neg = np.maximum(0.5 + pts[:, 0], 1e-12)
        wall_dist_y_pos = np.maximum(0.5 - pts[:, 1], 1e-12)
        wall_dist_y_neg = np.maximum(0.5 + pts[:, 1], 1e-12)
        
        wall_val = np.sum(1.0 / (wall_dist_x_pos**3) + 
                          1.0 / (wall_dist_x_neg**3) + 
                          1.0 / (wall_dist_y_pos**3) + 
                          1.0 / (wall_dist_y_neg**3))
        
        total_val = val + 10.0 * wall_val
        
        # Gradient for 1/d^12: d(1/d^12)/dx = -12/d^13 * dx/dd = -12 (x_i-x_j)/d^14
        # In terms of dist_sq (d^2): 1/dist_sq^3.
        # d(1/dist_sq^3)/d(pts[i]) = -3 * (1/dist_sq^4) * 2 * (pts[i] - pts[j]) = -6 * (pts[i] - pts[j]) / dist_sq^4
        
        dist_sq_4 = dist_sq**4
        dist_sq_4 = np.where(dist_sq > 1e-12, dist_sq_4, 1e-12)
        
        term = 6.0 * diff / dist_sq_4[:, :, np.newaxis]
        grad_pts = np.sum(term, axis=1)
        
        # Wall grad for 1/w^6: -6/w^7 * dw/dx
        gw_x_pos = 6.0 * 10.0 / (wall_dist_x_pos**7)
        gw_x_neg = -6.0 * 10.0 / (wall_dist_x_neg**7)
        gw_y_pos = 6.0 * 10.0 / (wall_dist_y_pos**7)
        gw_y_neg = -6.0 * 10.0 / (wall_dist_y_neg**7)
        
        grad_pts[:, 0] += gw_x_pos + gw_x_neg
        grad_pts[:, 1] += gw_y_pos + gw_y_neg
        
        return total_val, grad_pts.flatten()

    if not check_time():
        try:
            res2 = so.minimize(objective_and_grad_2, x_opt, method='L-BFGS-B', 
                               jac=True, bounds=bounds, 
                               options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10})
            if hasattr(res2, 'x'):
                x_opt = res2.x
                pts_opt = x_opt.reshape(N, 2)
                pts_opt = np.clip(pts_opt, -0.5, 0.5)
        except Exception:
            pass

    return pts_opt
