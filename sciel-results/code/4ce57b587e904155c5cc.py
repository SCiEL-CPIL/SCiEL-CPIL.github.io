# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_volume_mc(x, y, n_samples=20000, seed=12345):
    """
    Compute the Monte Carlo estimate of the union volume.
    x: array of shape (8,) base x positions for i=0..7
    y: array of shape (8,) base y positions for j=0..7
    """
    N = 8
    # Construct full 64 positions
    # i = k // 8, j = k % 8
    # x_k = x[i], y_k = y[j]
    
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = x[i_flat]
    base_y = y[j_flat]
    
    # Determine bounding box
    # X range: min(x_i + c*i/N) to max(x_i + 1/N + c*i/N) for c in [0,1]
    # Min x is min(x_i), Max x is max(x_i + 1/N + i/N)
    # Actually, x_i + c*i/N. If i>0, it increases. If i=0, constant.
    # So min X0 = min(x). Max X1 = max(x + 1/N + i/N).
    
    # We need to be careful with the bounding box for MC.
    # The task defines X0 = min_k x_k, X1 = max_k (x_k + (i+1)/8).
    # Note: the task description says "x_k + (i+1)/8". 
    # Wait, the tube is [x_k + c*i/8, x_k + c*i/8 + 1/8].
    # So the max x extent for tube k is x_k + 1/8 + i/8 = x_k + (i+1)/8.
    # This matches the task description.
    
    X0 = np.min(base_x)
    X1 = np.max(base_x + (i_flat + 1) / N)
    Y0 = np.min(base_y)
    Y1 = np.max(base_y + (j_flat + 1) / N)
    
    if X1 - X0 < 1e-9 or Y1 - Y0 < 1e-9:
        return 1e10 # Invalid or tiny, penalize heavily
        
    rng = np.random.default_rng(seed)
    a = rng.uniform(X0, X1, n_samples)
    b = rng.uniform(Y0, Y1, n_samples)
    c = rng.uniform(0, 1, n_samples)
    
    # Check if point (a, b, c) is in any tube
    # Tube k (i, j) contains (a, b, c) if:
    # x_k + c*i/N <= a <= x_k + c*i/N + 1/N
    # y_k + c*j/N <= b <= y_k + c*j/N + 1/N
    
    # Vectorized check
    # a is (M,), base_x is (64,). We need (M, 64) comparison.
    # This might be memory heavy if M is large. 
    # 20000 * 64 * 8 bytes = 10 MB. OK.
    
    # left_x = base_x + c[:, None] * i_flat[None, :] / N
    # right_x = left_x + 1/N
    
    # To save memory, we can check in chunks or just use broadcasting carefully.
    
    cx = c[:, None]
    ix = i_flat[None, :]
    jx = j_flat[None, :]
    bx = base_x[None, :]
    by = base_y[None, :]
    
    # Shape: (M, 64)
    lx = bx + cx * ix / N
    rx = lx + 1.0 / N
    ly = by + cx * jx / N
    ry = ly + 1.0 / N
    
    in_x = (a[:, None] >= lx) & (a[:, None] <= rx)
    in_y = (b[:, None] >= ly) & (b[:, None] <= ry)
    
    in_tube = in_x & in_y
    
    # If any tube contains the point
    hit = np.any(in_tube, axis=1)
    
    fraction = np.sum(hit) / n_samples
    volume = fraction * (X1 - X0) * (Y1 - Y0)
    
    return volume

def _keich_offsets(N=8):
    """Keich binary construction for N=2^k."""
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    return keich

def _optimize_coordinate(x0, y0, deadline):
    """
    Coordinate descent: alternate optimizing x and y.
    """
    N = 8
    x = x0.copy()
    y = y0.copy()
    
    # Use a moderate number of MC samples for speed during optimization
    n_samples = 10000 
    
    best_x = x.copy()
    best_y = y.copy()
    best_vol = _compute_volume_mc(x, y, n_samples, seed=12345)
    
    for iteration in range(20):
        if time.time() > deadline:
            break
            
        # Optimize x given y
        def cost_x(x):
            return _compute_volume_mc(x, y, n_samples, seed=12345)
            
        try:
            res_x = minimize(cost_x, x, method='Nelder-Mead', 
                             options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-8})
            if res_x.fun < best_vol:
                x = res_x.x
                best_x = x.copy()
                best_vol = res_x.fun
        except Exception:
            pass
            
        if time.time() > deadline:
            break
            
        # Optimize y given x
        def cost_y(y):
            return _compute_volume_mc(x, y, n_samples, seed=12345)
            
        try:
            res_y = minimize(cost_y, y, method='Nelder-Mead', 
                             options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-8})
            if res_y.fun < best_vol:
                y = res_y.x
                best_y = y.copy()
                best_vol = res_y.fun
        except Exception:
            pass
            
        # Check convergence
        if best_vol < 1e-4: # Very small volume
            break

    return best_x, best_y

def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.
    Returns positions of shape (64, 2).
    """
    deadline = time.time() + 580  # Safety margin before 600s limit
    
    N = 8
    keich = _keich_offsets(N)
    
    # Initial guess
    x0 = keich.copy()
    y0 = keich.copy()
    
    # Coordinate Descent
    x_opt, y_opt = _optimize_coordinate(x0, y0, deadline)
    
    # Final refinement with more samples to ensure we aren't stuck in a noise local min?
    # Or just use the result.
    
    # Build full 64 positions
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = x_opt[i_flat]
    base_y = y_opt[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    
    return positions

# EVOLVE-BLOCK-END
