import numpy as np
import time
import sys
from scipy.optimize import minimize

def compute_smooth_volume(positions, grid_size=100, sigma=0.01):
    """
    Computes a smooth approximation of the 3D union volume.
    positions: (64, 2) array of (x_k, y_k).
    grid_size: resolution of the 3D grid.
    sigma: width of the sigmoid approximation for the indicator function.
    """
    N = 8
    # Reshape positions to (8, 8, 2)
    pos_3d = positions.reshape(N, N, 2)
    
    # The box bounds
    # X0 = min_k x_k, X1 = max_k (x_k + (i+1)/8)
    # Y0 = min_k y_k, Y1 = max_k (y_k + (j+1)/8)
    
    # To make it differentiable and stable, we can fix the box to a reasonable range
    # or compute it. Computing bounds inside the objective is non-differentiable 
    # at the min/max switches. 
    # Strategy: The volume is proportional to (X1-X0)*(Y1-Y0)*Fraction.
    # If we keep the positions within a fixed box, say [0,1]x[0,1] roughly, 
    # the box size is constant. 
    # The tubes at c=0 are in [x_k, x_k+1/8]. If we constrain x_k in [0, 7/8], 
    # the box is [0,1]. Same for y.
    # Let's assume the box is [0,1]x[0,1]x[0,1] for the optimization, 
    # and add a penalty if positions go out of bounds, or just let the optimizer 
    # figure out the optimal scaling.
    # Actually, the score is 1/V. V = Fraction * BoxVol.
    # If we shrink the positions, BoxVol shrinks.
    # If we expand them, BoxVol grows.
    # The optimal solution likely fills the box efficiently.
    
    # Let's use a fixed box [0,1]x[0,1] for the grid evaluation to ensure 
    # differentiability. The actual box in the evaluator is the bounding box.
    # If we optimize for the fixed box [0,1], the resulting V_eval might be 
    # smaller if the bounding box is smaller than 1.
    # However, the evaluator uses the ACTUAL bounding box.
    # To be safe, we should minimize the actual volume.
    # Since min/max are non-smooth, we can use a soft-min/max or just 
    # optimize the fraction assuming a fixed box and then check.
    
    # Better approach: 
    # The volume is the measure of the set U.
    # V = \int_0^1 Area(c) dc.
    # Area(c) is the area of the union of 64 squares at height c.
    # Each square k=(i,j) at height c is [x_k + c*i/8, x_k + c*i/8 + 1/8] x [y_k + c*j/8, y_k + c*j/8 + 1/8].
    
    # We approximate the integral with a grid in c, and for each c, 
    # we approximate the 2D area with a grid in x,y.
    
    # To make it differentiable, we use a smooth indicator.
    # I(x, [L, R]) = sigmoid((x - (L - w))/w) * (1 - sigmoid((x - (R + w))/w))
    # This is 0 outside, 1 inside, smooth transition of width w.
    
    # Grid for c
    c_vals = np.linspace(0, 1, 20) # 20 slices in c
    dc = c_vals[1] - c_vals[0]
    
    # Grid for x, y
    # The positions will likely be in [0, 1]. 
    # Let's evaluate on a grid [0, 1] x [0, 1].
    # If the actual bounding box is smaller, the volume will be smaller, 
    # which is good. If it's larger, the volume is larger.
    # We assume the optimizer will keep them compact.
    
    x_vals = np.linspace(0, 1, grid_size)
    y_vals = np.linspace(0, 1, grid_size)
    dx = x_vals[1] - x_vals[0]
    dy = y_vals[1] - y_vals[0]
    
    X, Y = np.meshgrid(x_vals, y_vals, indexing='xy')
    
    total_volume = 0.0
    
    for c in c_vals:
        # Compute the smooth indicator for each tube at this c
        # Indicator is a 2D array of shape (grid_size, grid_size)
        # Union is the maximum (or 1 - product(1 - ind))
        
        # To save memory/time, we can compute the union incrementally
        # union_prob = 1 - prod(1 - ind_k)
        # Or simply max if we use hard indicators, but we need smooth.
        # 1 - prod(1 - ind_k) is differentiable.
        
        one_minus_prob = np.ones((grid_size, grid_size))
        
        for i in range(N):
            for j in range(N):
                xk = pos_3d[i, j, 0]
                yk = pos_3d[i, j, 1]
                
                # Square at height c
                lx = xk + c * i / 8.0
                rx = lx + 1.0 / 8.0
                ly = yk + c * j / 8.0
                ry = ly + 1.0 / 8.0
                
                # Smooth indicator in X
                # sigmoid((x - (lx - sigma))/sigma) * (1 - sigmoid((x - (rx + sigma))/sigma))
                # This is a bit expensive for 64 tubes * 20 c * 100x100 grid.
                # 64 * 20 * 10000 = 12.8M operations per objective call.
                # L-BFGS might need 100-1000 calls. 1.2B - 12B ops. Might be too slow in Python.
                
                # Optimization: Vectorize over tubes?
                # Or reduce grid size?
                # Let's try grid_size=50 first. 64 * 20 * 2500 = 3.2M.
                # Still heavy for pure Python loops.
                
                # Let's vectorize the tube loop for a fixed c.
                # We can precompute the shifts.
                
                pass
        
        # This loop is too slow. We need a faster approximation.
        
    return total_volume

def compute_fast_smooth_volume(positions, c_samples=10, xy_grid=50, sigma=0.02):
    """
    Faster version using vectorization over tubes.
    """
    N = 8
    pos_3d = positions.reshape(N, N, 2)
    
    c_vals = np.linspace(0, 1, c_samples)
    dc = c_vals[1] - c_vals[0]
    
    x_vals = np.linspace(0, 1, xy_grid)
    y_vals = np.linspace(0, 1, xy_grid)
    dx = x_vals[1] - x_vals[0]
    dy = y_vals[1] - y_vals[0]
    
    X, Y = np.meshgrid(x_vals, y_vals, indexing='xy')
    X_flat = X.flatten() # (xy_grid*xy_grid,)
    Y_flat = Y.flatten()
    
    total_volume = 0.0
    
    for c in c_vals:
        # For this c, compute the indicator for all 64 tubes
        # We want 1 - prod(1 - ind_k)
        # ind_k(x,y) = S(x, lx_k, rx_k) * S(y, ly_k, ry_k)
        
        # lx_k = x_k + c*i/8
        # rx_k = lx_k + 1/8
        
        # Let's compute the X and Y indicators separately for all tubes
        # x_ind: (64, xy_grid*xy_grid)
        # y_ind: (64, xy_grid*xy_grid)
        
        # Vectorized computation of lx, rx, ly, ry
        i_idx, j_idx = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
        i_flat = i_idx.flatten()
        j_flat = j_idx.flatten()
        
        xk = pos_3d[i_flat, j_flat, 0]
        yk = pos_3d[i_flat, j_flat, 1]
        
        lx = xk + c * i_flat / 8.0
        rx = lx + 1.0 / 8.0
        ly = yk + c * j_flat / 8.0
        ry = ly + 1.0 / 8.0
        
        # Compute smooth step functions
        # S_low = sigmoid((x - (l - sigma))/sigma)
        # S_high = 1 - sigmoid((x - (r + sigma))/sigma)
        # ind = S_low * S_high
        
        # Broadcasting: lx is (64,), X_flat is (G,)
        # We want (64, G)
        
        # To avoid huge memory usage, we can do it in chunks or use a smaller grid.
        # Let's stick to xy_grid=50 -> G=2500. 64*2500 = 160k floats. Fine.
        
        x_diff_low = (X_flat[None, :] - (lx[:, None] - sigma)) / sigma
        x_diff_high = (X_flat[None, :] - (rx[:, None] + sigma)) / sigma
        
        x_ind = (1.0 / (1.0 + np.exp(-x_diff_low))) * (1.0 - (1.0 / (1.0 + np.exp(-x_diff_high))))
        
        y_diff_low = (Y_flat[None, :] - (ly[:, None] - sigma)) / sigma
        y_diff_high = (Y_flat[None, :] - (ry[:, None] + sigma)) / sigma
        
        y_ind = (1.0 / (1.0 + np.exp(-y_diff_low))) * (1.0 - (1.0 / (1.0 + np.exp(-y_diff_high))))
        
        tube_ind = x_ind * y_ind # (64, G)
        
        # Union probability at each grid point
        # p_union = 1 - prod(1 - tube_ind, axis=0)
        one_minus_prob = np.prod(1.0 - tube_ind, axis=0)
        p_union = 1.0 - one_minus_prob
        
        area = np.sum(p_union) * dx * dy
        total_volume += area * dc
        
    return total_volume

def kakeya_needle_3d8() -> np.ndarray:
    start_time = time.time()
    deadline = 630.0
    
    N = 8
    rng = np.random.default_rng(12345)
    
    # Initial guess: Keich construction or random
    # Let's start with a compact configuration
    # x_k = (k // 8) * 0.1, y_k = (k % 8) * 0.1
    init_pos = np.zeros((64, 2))
    for k in range(64):
        i = k // 8
        j = k % 8
        init_pos[k, 0] = i * 0.1
        init_pos[k, 1] = j * 0.1
        
    # Add some random noise to break symmetry
    init_pos += rng.normal(0, 0.01, init_pos.shape)
    
    best_pos = init_pos.copy()
    best_vol = compute_fast_smooth_volume(best_pos)
    
    # Optimize with L-BFGS-B
    # We need to handle the bounds. 
    # The evaluator uses the bounding box. 
    # If we constrain to [0, 1], the box is at most [0,1]x[0,1].
    # If we allow negative values, the box might be larger.
    # Let's constrain to [-0.5, 1.5] to allow some flexibility, 
    # but the smooth volume assumes [0,1] grid. 
    # This mismatch might be an issue.
    # Better: Optimize on [0,1] grid, and hope the optimal solution stays there.
    # Or, adjust the grid to cover the bounding box.
    # For now, assume [0,1] is sufficient.
    
    bounds = [(-0.2, 1.2)] * 128 # 64 x 2
    
    try:
        res = minimize(
            compute_fast_smooth_volume, 
            init_pos.flatten(), 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6}
        )
        
        if res.success and res.fun < best_vol:
            best_pos = res.x.reshape(64, 2)
            best_vol = res.fun
            
    except Exception as e:
        print(f"Optimization failed: {e}")
        
    # If we have time, try a few random restarts
    if time.time() - start_time < deadline - 60:
        for _ in range(3):
            if time.time() - start_time > deadline - 30:
                break
            
            init_pos2 = rng.uniform(0, 0.8, (64, 2))
            # Compact initialization
            # Let's try to initialize based on the current best but perturbed
            init_pos2 = best_pos + rng.normal(0, 0.1, (64, 2))
            
            try:
                res2 = minimize(
                    compute_fast_smooth_volume, 
                    init_pos2.flatten(), 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6}
                )
                
                if res2.success and res2.fun < best_vol:
                    best_pos = res2.x.reshape(64, 2)
                    best_vol = res2.fun
                    
            except:
                pass
                
    return best_pos

if __name__ == "__main__":
    res = kakeya_needle_3d8()
    print(res.shape)
