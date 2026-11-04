import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    start_time = time.time()
    timeout = 140.0  # Safety margin
    
    best_result = None
    best_score = -np.inf
    
    # Optimization function
    def objective(params):
        # params: [x0, y0, r0, x1, y1, r1, ..., x20, y20, r20, W]
        if len(params) != 3*N + 1:
            return 0, np.zeros_like(params)
        
        W = params[-1]
        H = 2.0 - W
        
        if W <= 0 or H <= 0:
            return 1e6, np.zeros_like(params)
            
        xy_r = params[:3*N].reshape(N, 3)
        x = xy_r[:, 0]
        y = xy_r[:, 1]
        r = xy_r[:, 2]
        
        # Ensure radii are positive
        r = np.maximum(r, 1e-6)
        
        sum_r = np.sum(r)
        
        # Overlap penalty
        # d_ij
        diff_x = x[:, None] - x[None, :]
        diff_y = y[:, None] - y[None, :]
        d = np.sqrt(diff_x**2 + diff_y**2)
        np.fill_diagonal(d, np.inf)
        
        r_sum = r[:, None] + r[None, :]
        
        overlap = np.maximum(0, r_sum - d)
        overlap_penalty = np.sum(overlap**2)
        
        # Boundary penalty
        # x_i - r_i >= 0
        # x_i + r_i <= W
        # y_i - r_i >= 0
        # y_i + r_i <= H
        
        bound_x_low = np.maximum(0, r - x)
        bound_x_high = np.maximum(0, x + r - W)
        bound_y_low = np.maximum(0, r - y)
        bound_y_high = np.maximum(0, y + r - H)
        
        bound_penalty = np.sum(bound_x_low**2) + np.sum(bound_x_high**2) + np.sum(bound_y_low**2) + np.sum(bound_y_high**2)
        
        # Objective: Maximize sum_r, Minimize penalties
        # Loss = -sum_r + lambda_overlap * overlap_penalty + lambda_bound * bound_penalty
        lambda_overlap = 10.0
        lambda_bound = 10.0
        
        loss = -sum_r + lambda_overlap * overlap_penalty + lambda_bound * bound_penalty
        
        # Gradient
        grad = np.zeros_like(params)
        
        # d(loss)/d(r_i) = -1 + 2*lambda_overlap * sum_j overlap_ij * sign(r_i+r_j-d_ij) + 2*lambda_bound * (bound terms)
        # This is getting complex. Let's use numerical gradient or just use finite differences if needed, 
        # but L-BFGS-B with finite differences is slow.
        # Let's try to derive it.
        
        # Actually, let's just use scipy.optimize.minimize with finite differences first to see if it works, 
        # or implement the gradient carefully.
        
        # For now, let's return the loss and let scipy handle gradients? 
        # No, finite differences on 43 variables is okay for a few iterations, but might be slow for many restarts.
        
        # Let's implement the gradient.
        
        # d(overlap_penalty)/d(x_i)
        # overlap_ij = max(0, r_i+r_j - d_ij)
        # d(overlap_ij^2)/d(x_i) = 2 * overlap_ij * d(overlap_ij)/d(x_i)
        # d(overlap_ij)/d(x_i) = -d(d_ij)/d(x_i) = -(x_i - x_j) / d_ij
        # So term is 2 * overlap_ij * (-(x_i - x_j) / d_ij)
        
        # We need to sum over j.
        
        # This is vectorizable.
        
        # d(d_ij)/d(x_i) = (x_i - x_j) / d_ij
        # Let M = overlap_matrix (N x N)
        # M[i,j] = max(0, r_i+r_j - d_ij)
        
        # Grad_x[i] = 2 * lambda_overlap * sum_j M[i,j] * (-(x_i - x_j) / d_ij)
        
        # Let's compute the matrix of gradients.
        
        # d_ij is already computed.
        # We need to avoid division by zero.
        with np.errstate(divide='ignore', invalid='ignore'):
            grad_d_dx = (x[:, None] - x[None, :]) / d
            grad_d_dy = (y[:, None] - y[None, :]) / d
        
        # M is N x N
        # Sum over j for each i
        # grad_x[i] = -2 * lambda_overlap * sum_j M[i,j] * grad_d_dx[i,j]
        
        # Note: M[i,i] = 0, d[i,i] = inf, so grad_d_dx[i,i] = 0. Safe.
        
        grad_x = -2 * lambda_overlap * np.sum(M * grad_d_dx, axis=1)
        grad_y = -2 * lambda_overlap * np.sum(M * grad_d_dy, axis=1)
        
        # d(overlap_penalty)/d(r_i)
        # d(M_ij)/d(r_i) = 1 if M_ij > 0 else 0
        # So d(M_ij^2)/d(r_i) = 2 * M_ij * (1 if M_ij > 0 else 0)
        # Sum over j
        grad_r_overlap = 2 * lambda_overlap * np.sum(M * (M > 0), axis=1)
        
        # Boundary gradients
        # bound_x_low = max(0, r_i - x_i)
        # d(bound_x_low^2)/d(r_i) = 2 * bound_x_low * 1
        # d(bound_x_low^2)/d(x_i) = 2 * bound_x_low * (-1)
        
        # bound_x_high = max(0, x_i + r_i - W)
        # d(bound_x_high^2)/d(r_i) = 2 * bound_x_high * 1
        # d(bound_x_high^2)/d(x_i) = 2 * bound_x_high * 1
        
        # Similarly for y.
        
        grad_r_bound = 2 * lambda_bound * (bound_x_low + bound_x_high + bound_y_low + bound_y_high)
        grad_x_bound = 2 * lambda_bound * (-bound_x_low + bound_x_high)
        grad_y_bound = 2 * lambda_bound * (-bound_y_low + bound_y_high)
        
        # Total gradient
        # d(loss)/d(x_i) = grad_x[i] + grad_x_bound[i]
        # d(loss)/d(y_i) = grad_y[i] + grad_y_bound[i]
        # d(loss)/d(r_i) = -1 + grad_r_overlap[i] + grad_r_bound[i]
        
        # d(loss)/d(W)
        # H = 2 - W
        # bound_y_high = max(0, y_i + r_i - H) = max(0, y_i + r_i - (2-W)) = max(0, y_i + r_i - 2 + W)
        # d(bound_y_high^2)/d(W) = 2 * bound_y_high * 1
        # bound_x_high = max(0, x_i + r_i - W)
        # d(bound_x_high^2)/d(W) = 2 * bound_x_high * (-1)
        
        grad_W = 2 * lambda_bound * (np.sum(bound_y_high) - np.sum(bound_x_high))
        
        # Assemble gradient
        grad[:N*3:N*3] = grad_x + grad_x_bound
        grad[1:N*3:N*3] = grad_y + grad_y_bound
        grad[2:N*3:N*3] = -1 + grad_r_overlap + grad_r_bound
        grad[-1] = grad_W
        
        return loss, grad

    # Define the optimization loop
    for restart in range(50):
        if time.time() - start_time > timeout:
            break
            
        # Initialization
        # Grid initialization
        side = int(np.ceil(np.sqrt(N)))
        # 5x5 grid = 25 cells, use 21
        # Or 4x6 = 24 cells
        # Let's use a 5x5 grid, drop 4 random ones? Or just use first 21.
        
        # Better: Random initialization within the box
        W_init = 1.0
        H_init = 1.0
        
        # Random positions
        x0 = np.random.rand(N) * W_init
        y0 = np.random.rand(N) * H_init
        
        # Initial radii: small, or based on grid
        # Let's use a grid-based initial radius
        # Approximate radius for N circles in 1x1 square is ~ 1/sqrt(N)
        r0 = np.ones(N) * (0.5 / np.sqrt(N))
        
        # Perturb
        x0 += np.random.normal(0, 0.01, N)
        y0 += np.random.normal(0, 0.01, N)
        r0 += np.random.normal(0, 0.01, N)
        r0 = np.maximum(r0, 0.01)
        
        # Clip to box
        x0 = np.clip(x0, r0, W_init - r0)
        y0 = np.clip(y0, r0, H_init - r0)
        
        params0 = np.concatenate([np.column_stack([x0, y0, r0]).flatten(), [W_init]])
        
        # Optimize
        try:
            res = minimize(objective, params0, method='L-BFGS-B', jac=True, options={'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-8})
            
            if res.success or res.fun < best_score:
                # Extract result
                W_opt = res.x[-1]
                H_opt = 2.0 - W_opt
                
                if W_opt > 0 and H_opt > 0:
                    xy_r_opt = res.x[:3*N].reshape(N, 3)
                    x_opt = xy_r_opt[:, 0]
                    y_opt = xy_r_opt[:, 1]
                    r_opt = xy_r_opt[:, 2]
                    
                    # Check validity
                    valid = True
                    # Check boundaries
                    if np.any(x_opt < r_opt - 1e-5) or np.any(x_opt + r_opt > W_opt + 1e-5):
                        valid = False
                    if np.any(y_opt < r_opt - 1e-5) or np.any(y_opt + r_opt > H_opt + 1e-5):
                        valid = False
                    
                    # Check overlaps
                    diff_x = x_opt[:, None] - x_opt[None, :]
                    diff_y = y_opt[:, None] - y_opt[None, :]
                    d = np.sqrt(diff_x**2 + diff_y**2)
                    np.fill_diagonal(d, np.inf)
                    r_sum = r_opt[:, None] + r_opt[None, :]
                    if np.any(r_sum > d + 1e-5):
                        valid = False
                    
                    if valid:
                        score = np.sum(r_opt)
                        if score > best_score:
                            best_score = score
                            best_result = np.column_stack([x_opt, y_opt, r_opt])
                            
        except Exception as e:
            pass

    if best_result is None:
        # Fallback to grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_result
