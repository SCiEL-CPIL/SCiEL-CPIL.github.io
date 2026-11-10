import numpy as np
import scipy.optimize as opt
import time

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    # Initial Keich construction
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)

    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = keich[i_flat]
    base_y = keich[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    
    # Optimization
    start_time = time.time()
    deadline = start_time + 600.0 # 10 minutes
    
    # Flatten positions to 1D vector of 128
    p0 = positions.flatten()
    
    # Define the smooth volume function
    def volume_loss(p):
        # Reshape to (64, 2)
        pos = p.reshape(64, 2)
        x = pos[:, 0]
        y = pos[:, 1]
        
        # Tube definitions
        # At height c, square k is [x_k + c*i_k/8, x_k + 1/8 + c*i_k/8] x [y_k + c*j_k/8, y_k + 1/8 + c*j_k/8]
        
        # We will integrate over c in [0, 1] using 10 steps
        # And over x, y using a grid
        # To avoid boundary issues, we define a fixed grid large enough
        # Or we can compute the bounding box and use it.
        # Let's use a fixed grid for speed and stability, covering [-2, 3] x [-2, 3]
        # The Keich positions are in [0, 1].
        # With shear, max x is 1 + 7/8 = 1.875.
        # So [0, 2] is safe. Let's use [0, 2.5] to be safe.
        
        x_min, x_max = 0.0, 2.5
        y_min, y_max = 0.0, 2.5
        
        # Grid size 40x40 for x,y
        gx = np.linspace(x_min, x_max, 40)
        gy = np.linspace(y_min, y_max, 40)
        dx = gx[1] - gx[0]
        dy = gy[1] - gy[0]
        
        # c steps
        c_steps = 10
        dc = 1.0 / c_steps
        
        total_vol = 0.0
        
        # Pre-compute grid points
        X, Y = np.meshgrid(gx, gy, indexing='ij')
        
        for c_idx in range(c_steps):
            c = (c_idx + 0.5) * dc
            
            # Compute square boundaries for all 64 tubes
            # x_start = x_k + c * i_k / 8
            # x_end = x_start + 1/8
            # y_start = y_k + c * j_k / 8
            # y_end = y_start + 1/8
            
            xs = x + c * i_flat / 8.0
            xe = xs + 1.0 / 8.0
            ys = y + c * j_flat / 8.0
            ye = ys + 1.0 / 8.0
            
            # For each grid point, check if inside any square
            # To make it smooth, we use a soft max of the "inside" indicators
            # Indicator for square k: 1 if X in [xs_k, xe_k] and Y in [ys_k, ye_k]
            # Let's compute distance to boundary for each square
            # d_x = min(X - xs_k, xe_k - X)
            # d_y = min(Y - ys_k, ye_k - Y)
            # d = min(d_x, d_y)
            # I_k = sigmoid(d / eps)
            
            # This is expensive: 64 squares * 1600 grid points = 102k ops per c step.
            # 10 c steps = 1M ops.
            # This is fast.
            
            # Vectorized computation
            # X is 40x40, xs is (64,)
            # We need to compute d for all k and grid points.
            # Shape: (64, 40, 40)
            
            # d_x_k = min(X - xs_k, xe_k - X)
            # We can compute this as:
            # X_exp = X[None, :, :] # (1, 40, 40)
            # xs_exp = xs[:, None, None] # (64, 1, 1)
            # xe_exp = xe[:, None, None]
            
            # dx = np.minimum(X_exp - xs_exp, xe_exp - X_exp)
            # dy = np.minimum(Y_exp - ys_exp, ye_exp - Y_exp)
            # d = np.minimum(dx, dy)
            
            # I = 1 / (1 + np.exp(-d / eps))
            # Union = 1 - np.prod(1 - I, axis=0) ? No, union is max.
            # Soft max: np.logsumexp(beta * I) / beta
            # Or just np.max(I, axis=0) if we accept non-smoothness.
            # L-BFGS-B can handle non-smoothness if we use finite differences, but it's better to be smooth.
            # Let's use soft max.
            
            X_exp = X[None, :, :]
            Y_exp = Y[None, :, :]
            xs_exp = xs[:, None, None]
            xe_exp = xe[:, None, None]
            ys_exp = ys[:, None, None]
            ye_exp = ye[:, None, None]
            
            dx = np.minimum(X_exp - xs_exp, xe_exp - X_exp)
            dy = np.minimum(Y_exp - ys_exp, ye_exp - Y_exp)
            d = np.minimum(dx, dy)
            
            eps = 0.05
            I = 1.0 / (1.0 + np.exp(-d / eps))
            
            # Soft max approximation of union
            beta = 10.0
            # logsumexp trick
            I_max = np.max(I, axis=0)
            union = I_max + np.log(np.mean(np.exp(I - I_max))) / beta
            # Actually, soft max is: (1/beta) * log(sum(exp(beta * I)))
            # This is better:
            union = np.log(np.sum(np.exp(beta * I), axis=0)) / beta
            
            area = np.sum(union) * dx * dy
            total_vol += area * dc
            
        return total_vol

    # Optimize
    best_p = p0.copy()
    best_vol = volume_loss(p0)
    
    # Use L-BFGS-B
    # We might need to restart or use multiple initializations if it gets stuck.
    # Given the deadline, we will just run one long optimization.
    
    try:
        res = opt.minimize(volume_loss, p0, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-8})
        if res.fun < best_vol:
            best_vol = res.fun
            best_p = res.x
    except Exception:
        pass
    
    # Check time
    if time.time() < deadline - 60:
        # Try a second pass with a different method or more iterations
        try:
            res2 = opt.minimize(volume_loss, best_p, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
            if res2.fun < best_vol:
                best_vol = res2.fun
                best_p = res2.x
        except Exception:
            pass
            
    return best_p.reshape(64, 2)
