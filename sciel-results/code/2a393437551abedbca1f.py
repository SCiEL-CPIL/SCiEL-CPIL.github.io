import numpy as np
import time

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    w = 1.0 / N  # tube width = 0.125
    start_time = time.time()
    deadline = 630.0  # evaluator kills at 630s
    
    # Precompute grids
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten().astype(float)
    j_flat = j_grid.flatten().astype(float)
    
    def compute_volume(xk, yk, num_samples=50000):
        """Fast volume estimator for local search. xk, yk are shape (64,)."""
        # Compute bounding box
        X0 = np.min(xk)
        X1 = np.max(xk + i_flat / N)
        Y0 = np.min(yk)
        Y1 = np.max(yk + j_flat / N)
        
        area = (X1 - X0) * (Y1 - Y0)
        if area <= 1e-9:
            return float('inf')
        
        rng = np.random.default_rng(12345)
        a = rng.uniform(X0, X1, num_samples)
        b = rng.uniform(Y0, Y1, num_samples)
        c = rng.uniform(0, 1, num_samples)
        
        # Vectorized check
        c_col = c[:, np.newaxis]
        x_left = xk[np.newaxis, :] + c_col * (i_flat / N)[np.newaxis, :]
        y_left = yk[np.newaxis, :] + c_col * (j_flat / N)[np.newaxis, :]
        
        in_x = (a[:, np.newaxis] >= x_left) & (a[:, np.newaxis] <= x_left + w)
        in_y = (b[:, np.newaxis] >= y_left) & (b[:, np.newaxis] <= y_left + w)
        in_any = np.any(in_x & in_y, axis=1)
        
        fraction = np.mean(in_any)
        volume = fraction * area
        return volume
    
    def compute_volume_full(xk, yk, num_samples=500000):
        """Full precision volume estimator for final evaluation."""
        X0 = np.min(xk)
        X1 = np.max(xk + i_flat / N)
        Y0 = np.min(yk)
        Y1 = np.max(yk + j_flat / N)
        
        area = (X1 - X0) * (Y1 - Y0)
        if area <= 1e-9:
            return float('inf')
        
        rng = np.random.default_rng(12345)
        a = rng.uniform(X0, X1, num_samples)
        b = rng.uniform(Y0, Y1, num_samples)
        c = rng.uniform(0, 1, num_samples)
        
        c_col = c[:, np.newaxis]
        x_left = xk[np.newaxis, :] + c_col * (i_flat / N)[np.newaxis, :]
        y_left = yk[np.newaxis, :] + c_col * (j_flat / N)[np.newaxis, :]
        
        in_x = (a[:, np.newaxis] >= x_left) & (a[:, np.newaxis] <= x_left + w)
        in_y = (b[:, np.newaxis] >= y_left) & (b[:, np.newaxis] <= y_left + w)
        in_any = np.any(in_x & in_y, axis=1)
        
        fraction = np.mean(in_any)
        volume = fraction * area
        return volume
    
    # Parametric deformation field approach:
    # x_k = f(i) + g(j) where f, g are low-order polynomials
    # y_k = h(i) + k(j) where h, k are low-order polynomials
    # This reduces 128 DOF to ~24 parameters
    
    def params_to_positions(params):
        """Convert 24 parameters to 64 positions.
        params[0:4] = coefficients for f(i) (cubic)
        params[4:8] = coefficients for g(j) (cubic)
        params[8:12] = coefficients for h(i) (cubic)
        params[12:16] = coefficients for k(j) (cubic)
        params[16:20] = quadratic correction for x: a*i^2*b + b*j^2*a + c*i*j
        params[20:24] = quadratic correction for y
        """
        # Actually, let's use a simpler but effective parametric form:
        # x_k = a0 + a1*i + a2*j + a3*i*j + a4*i^2 + a5*j^2
        # y_k = b0 + b1*i + b2*j + b3*i*j + b4*i^2 + b5*j^2
        # That's 12 parameters total
        
        a0, a1, a2, a3, a4, a5 = params[0:6]
        b0, b1, b2, b3, b4, b5 = params[6:12]
        
        xk = a0 + a1 * i_flat + a2 * j_flat + a3 * i_flat * j_flat + a4 * i_flat**2 + a5 * j_flat**2
        yk = b0 + b1 * i_flat + b2 * j_flat + b3 * i_flat * j_flat + b4 * i_flat**2 + b5 * j_flat**2
        return xk, yk
    
    def compute_volume_from_params(params, num_samples=50000):
        xk, yk = params_to_positions(params)
        return compute_volume(xk, yk, num_samples)
    
    # Initial parameters: try symmetric configuration
    # x_k = -alpha * i, y_k = -alpha * j
    alpha_init = 0.0625
    params0 = np.array([
        0.0, -alpha_init, 0.0, 0.0, 0.0, 0.0,  # x: a0, a1, a2, a3, a4, a5
        0.0, 0.0, -alpha_init, 0.0, 0.0, 0.0   # y: b0, b1, b2, b3, b4, b5
    ])
    
    # Also try other initial configurations
    candidates = [
        # Symmetric linear
        np.array([0.0, -alpha_init, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -alpha_init, 0.0, 0.0, 0.0]),
        # With cross term
        np.array([0.0, -alpha_init, -alpha_init, 0.0, 0.0, 0.0, 0.0, -alpha_init, -alpha_init, 0.0, 0.0, 0.0]),
        # Pure x depends on i, y depends on j with different scales
        np.array([0.0, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.05, 0.0, 0.0, 0.0]),
        np.array([0.0, -0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.0, 0.0]),
        # Quadratic terms
        np.array([0.0, -0.08, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, -0.08, 0.0, 0.0, 0.01]),
        np.array([0.0, -0.08, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, -0.08, 0.01, 0.0, 0.0]),
    ]
    
    # Evaluate all candidates
    best_vol = float('inf')
    best_params = params0.copy()
    for p in candidates:
        vol = compute_volume_from_params(p, 50000)
        if vol < best_vol:
            best_vol = vol
            best_params = p.copy()
    
    # Gradient-free optimization using Nelder-Mead
    from scipy.optimize import minimize
    
    def objective(params):
        return compute_volume_from_params(params, 50000)
    
    # Nelder-Mead optimization
    options = {
        'maxiter': 2000,
        'xatol': 1e-8,
        'fatol': 1e-10,
        'adaptive': True
    }
    
    result = minimize(objective, best_params, method='Nelder-Mead', options=options)
    
    if result.fun < best_vol:
        best_params = result.x
        best_vol = result.fun
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline - 30:
        # Not enough time for refinement, return current best
        xk, yk = params_to_positions(best_params)
        return np.stack([xk, yk], axis=1)
    
    # Refinement: use the parametric result as initial guess for full 128-DOF search
    xk, yk = params_to_positions(best_params)
    best_xk, best_yk = xk.copy(), yk.copy()
    best_vol = compute_volume(xk, yk, 50000)
    
    # Coordinate descent with adaptive step size
    step_size = 0.005
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() - start_time < deadline - 60:
        improved = False
        for k in range(64):
            if time.time() - start_time > deadline - 60:
                break
            
            # Try perturbations in a pattern
            for dx in [-step_size, step_size]:
                for dy in [-step_size, step_size]:
                    xk_new = xk.copy()
                    yk_new = yk.copy()
                    xk_new[k] += dx
                    yk_new[k] += dy
                    
                    vol = compute_volume(xk_new, yk_new, 50000)
                    if vol < best_vol - 1e-12:
                        xk, yk = xk_new, yk_new
                        best_vol = vol
                        improved = True
                        no_improve_count = 0
                        break
                if improved:
                    break
            
            if not improved:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    break
        
        if not improved:
            step_size *= 0.7
            no_improve_count = 0
            if step_size < 1e-7:
                break
        else:
            no_improve_count = 0
    
    # Final polish with smaller steps
    if time.time() - start_time < deadline - 30:
        step_size = 1e-5
        for _ in range(3):
            for k in range(64):
                if time.time() - start_time > deadline - 30:
                    break
                for dx in [-step_size, step_size]:
                    for dy in [-step_size, step_size]:
                        xk_new = xk.copy()
                        yk_new = yk.copy()
                        xk_new[k] += dx
                        yk_new[k] += dy
                        vol = compute_volume(xk_new, yk_new, 50000)
                        if vol < best_vol - 1e-14:
                            xk, yk = xk_new, yk_new
                            best_vol = vol
                            break
                    if vol < best_vol + 1e-14:
                        break
            step_size *= 0.5
    
    # Return best configuration
    positions = np.stack([xk, yk], axis=1)
    return positions
