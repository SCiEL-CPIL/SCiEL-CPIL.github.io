import numpy as np
from scipy.optimize import minimize

N = 21

def construct():
    # Deadline guard
    import time
    start_time = time.time()
    deadline = 110.0  # 10 seconds margin
    
    # Variables: x, y, r for N circles, and W (width). H = 2 - W.
    # Total variables: 3*N + 1 = 64
    
    def get_H(W):
        return 2.0 - W

    def smooth_max(u, eps):
        return 0.5 * (u + np.sqrt(u**2 + eps**2))

    def objective_and_grad(params, eps):
        # params: [x1, y1, r1, ..., xN, yN, rN, W]
        xy = params[:3*N].reshape(N, 3)
        x = xy[:, 0]
        y = xy[:, 1]
        r = xy[:, 2]
        W = params[3*N]
        H = get_H(W)
        
        # Penalty weights
        w_overlap = 1000.0
        w_bound = 1000.0
        
        obj = -np.sum(r)
        grad = np.zeros_like(params)
        
        # Overlap penalties
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx**2 + dy**2)
                if d < 1e-10:
                    d = 1e-10
                # Overlap amount
                u = r[i] + r[j] - d
                s = smooth_max(u, eps)
                obj += w_overlap * s
                
                # Gradient of s wrt u
                ds_du = 0.5 * (1 + u / np.sqrt(u**2 + eps**2))
                
                # du/dx_i, etc.
                du_dx_i = -dx / d
                du_dy_i = -dy / d
                du_dr_i = 1.0
                
                grad[3*i] += w_overlap * ds_du * du_dx_i
                grad[3*i+1] += w_overlap * ds_du * du_dy_i
                grad[3*i+2] += w_overlap * ds_du * du_dr_i
                
                grad[3*j] += w_overlap * ds_du * (-du_dx_i)
                grad[3*j+1] += w_overlap * ds_du * (-du_dy_i)
                grad[3*j+2] += w_overlap * ds_du * du_dr_i

        # Boundary penalties
        for i in range(N):
            # x bounds
            u_x_min = r[i] - x[i]
            s_x_min = smooth_max(u_x_min, eps)
            obj += w_bound * s_x_min
            ds_du = 0.5 * (1 + u_x_min / np.sqrt(u_x_min**2 + eps**2))
            grad[3*i] -= w_bound * ds_du
            grad[3*i+2] += w_bound * ds_du
            
            u_x_max = x[i] - (W - r[i])
            s_x_max = smooth_max(u_x_max, eps)
            obj += w_bound * s_x_max
            ds_du = 0.5 * (1 + u_x_max / np.sqrt(u_x_max**2 + eps**2))
            grad[3*i] += w_bound * ds_du
            grad[3*i+2] -= w_bound * ds_du
            # d(W-r_i)/dW = 1, so d(x_i - (W-r_i))/dW = -1
            grad[3*N] -= w_bound * ds_du

            # y bounds
            u_y_min = r[i] - y[i]
            s_y_min = smooth_max(u_y_min, eps)
            obj += w_bound * s_y_min
            ds_du = 0.5 * (1 + u_y_min / np.sqrt(u_y_min**2 + eps**2))
            grad[3*i+1] -= w_bound * ds_du
            grad[3*i+2] += w_bound * ds_du
            
            u_y_max = y[i] - (H - r[i])
            # H = 2 - W, so H - r_i = 2 - W - r_i
            # u_y_max = y_i - (2 - W - r_i) = y_i - 2 + W + r_i
            s_y_max = smooth_max(u_y_max, eps)
            obj += w_bound * s_y_max
            ds_du = 0.5 * (1 + u_y_max / np.sqrt(u_y_max**2 + eps**2))
            grad[3*i+1] += w_bound * ds_du
            grad[3*i+2] += w_bound * ds_du
            grad[3*N] += w_bound * ds_du

        return obj, grad

    def run_optimization(init_params, eps_schedule):
        current_params = init_params.copy()
        for eps in eps_schedule:
            res = minimize(
                lambda p: objective_and_grad(p, eps)[0],
                current_params,
                jac=lambda p: objective_and_grad(p, eps)[1],
                method='L-BFGS-B',
                bounds=[(1e-6, None)] * (3*N) + [(0.5, 1.5)], # W bounds
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            current_params = res.x
            if time.time() > deadline:
                break
        return current_params

    # Initial guess: Grid
    side = int(np.ceil(np.sqrt(N)))
    # Try a few different aspect ratios and initial layouts
    
    best_params = None
    best_score = -np.inf

    # Layout 1: Square-ish grid
    for W_init in [1.0, 1.2, 0.8]:
        H_init = 2.0 - W_init
        init = np.zeros(3*N + 1)
        init[3*N] = W_init
        
        idx = 0
        for i in range(side):
            for j in range(side):
                if idx >= N: break
                x = (i + 0.5) * (W_init / side)
                y = (j + 0.5) * (H_init / side)
                r = 0.4 * min(W_init, H_init) / side
                init[3*idx] = x
                init[3*idx+1] = y
                init[3*idx+2] = r
                idx += 1
        
        # Random perturbation
        init[:3*N] += np.random.normal(0, 0.01, 3*N)
        
        params = run_optimization(init, [0.1, 0.01, 0.001, 0.0001])
        
        # Calculate score
        xy = params[:3*N].reshape(N, 3)
        score = np.sum(xy[:, 2])
        
        if score > best_score:
            best_score = score
            best_params = params.copy()
            
        if time.time() > deadline:
            break

    # Final polish with very small epsilon
    if best_params is not None and time.time() < deadline - 5:
        params = run_optimization(best_params, [1e-5, 1e-6, 1e-7])
        xy = params[:3*N].reshape(N, 3)
        score = np.sum(xy[:, 2])
        if score > best_score:
            best_score = score
            best_params = params.copy()

    # Extract result
    xy = best_params[:3*N].reshape(N, 3)
    # Ensure validity (clip to bounds if necessary, though optimizer should handle it)
    W = best_params[3*N]
    H = 2.0 - W
    for i in range(N):
        xy[i, 0] = np.clip(xy[i, 0], xy[i, 2], W - xy[i, 2])
        xy[i, 1] = np.clip(xy[i, 1], xy[i, 2], H - xy[i, 2])
        
    return xy
