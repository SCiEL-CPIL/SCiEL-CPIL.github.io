# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize, LinearConstraint
import time


def _compute_volume(positions, n_samples=200000):
    """
    Compute the Monte Carlo estimate of the union volume of the 64 tubes.
    positions: array of shape (64, 2)
    """
    N = 8
    pos = positions.reshape(N, N, 2)
    # i = row index (0..N-1), j = col index (0..N-1)
    i_idx = np.arange(N)[:, None]  # (N, 1)
    j_idx = np.arange(N)[None, :]  # (1, N)
    
    # Base positions
    x0 = pos[:, :, 0]  # (N, N)
    y0 = pos[:, :, 1]  # (N, N)
    
    # At height c, tube (i, j) occupies:
    # x in [x0[i,j] + c*i/N, x0[i,j] + c*i/N + 1/N]
    # y in [y0[i,j] + c*j/N, y0[i,j] + c*j/N + 1/N]
    
    # Compute bounding box
    x_max_top = x0 + 1.0 / N  # base x + 1/N (the top offset in x is c*i/N, max at c=1 is i/N, but the square width is 1/N)
    # Actually, the x-interval at height c is [x0 + c*i/N, x0 + c*i/N + 1/N]
    # So min x over all tubes and c is min(x0) + 0 = min(x0)
    # Max x over all tubes and c is max(x0 + i/N + 1/N) over i,j
    x_min = np.min(x0)
    x_max = np.max(x0 + i_idx / N + 1.0 / N)
    
    y_min = np.min(y0)
    y_max = np.max(y0 + j_idx / N + 1.0 / N)
    
    box_area = (x_max - x_min) * (y_max - y_min)
    if box_area <= 1e-9 or not np.isfinite(box_area):
        return np.inf
    
    # Use a fixed RNG for consistency in the objective
    rng = np.random.default_rng(12345)
    a = rng.uniform(x_min, x_max, n_samples)
    b = rng.uniform(y_min, y_max, n_samples)
    c = rng.uniform(0, 1, n_samples)
    
    # Check if point (a, b, c) is inside any tube
    # Tube (i, j): x in [x0[i,j] + c*i/N, x0[i,j] + c*i/N + 1/N]
    #               y in [y0[i,j] + c*j/N, y0[i,j] + c*j/N + 1/N]
    
    # Vectorized check: for each tube, check if a, b are in the interval at height c
    # a >= x0[i,j] + c*i/N and a <= x0[i,j] + c*i/N + 1/N
    # b >= y0[i,j] + c*j/N and b <= y0[i,j] + c*j/N + 1/N
    
    # Shape: a is (S,), x0 is (N,N), c is (S,)
    # We need to check for all i, j, s: 
    # a[s] >= x0[i,j] + c[s]*i/N and a[s] <= x0[i,j] + c[s]*i/N + 1/N
    # b[s] >= y0[i,j] + c[s]*j/N and b[s] <= y0[i,j] + c[s]*j/N + 1/N
    
    # Let's broadcast: 
    # a: (S, 1, 1), c: (S, 1, 1), x0: (1, N, N), i_idx: (1, N, 1), j_idx: (1, 1, N)
    a_b = a[:, None, None]
    b_b = b[:, None, None]
    c_b = c[:, None, None]
    x0_b = x0[None, :, :]
    y0_b = y0[None, :, :]
    i_b = i_idx[None, :, 0]  # (1, N, 1)
    j_b = j_idx[None, 0, :]  # (1, 1, N)
    
    # x_low = x0 + c * i / N
    x_low = x0_b + c_b * i_b / N
    x_high = x_low + 1.0 / N
    y_low = y0_b + c_b * j_b / N
    y_high = y_low + 1.0 / N
    
    # inside: (a >= x_low) & (a <= x_high) & (b >= y_low) & (b <= y_high)
    inside = (a_b >= x_low) & (a_b <= x_high) & (b_b >= y_low) & (b_b <= y_high)
    
    # Any tube covers the point
    covered = np.any(inside, axis=(1, 2))  # (S,)
    
    fraction = np.mean(covered)
    return fraction * box_area


def kakeya_needle_3d8() -> np.ndarray:
    """
    Optimize the 64 base positions to minimize the union volume of the sheared tubes.
    """
    N = 8
    start_time = time.time()
    deadline = start_time + 600  # 10 minutes
    
    # Initial guess: use the Keich construction as a starting point
    pos_keich = int(np.log2(N))
    j_terms = np.arange(1, pos_keich + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos_keich - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos_keich)
    
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = keich[i_flat]
    base_y = keich[j_flat]
    positions_init = np.stack([base_x, base_y], axis=1)
    
    # Compute initial volume
    vol_init = _compute_volume(positions_init, n_samples=200000)
    
    # Define the objective function for the optimizer
    # The optimizer works on a 128-dimensional vector
    def objective_flat(flat_vec):
        positions = flat_vec.reshape(N, N, 2)
        vol = _compute_volume(positions, n_samples=50000)  # Fewer samples for speed during optimization
        if not np.isfinite(vol):
            return 1e10
        return vol
    
    # Set up bounds for the optimizer
    # We need to ensure the bounding box is valid.
    # Let's set reasonable bounds. The Keich construction has values in [0, 1).
    # Let's allow a range of [-1, 2] for both x and y coordinates.
    bounds = [(-1.0, 2.0)] * (N * N * 2)
    
    # Run differential evolution
    best_positions = positions_init
    best_vol = vol_init
    
    try:
        result = differential_evolution(
            objective_flat,
            bounds,
            seed=12345,
            maxiter=50,
            popsize=15,
            tol=1e-4,
            mutation=(0.3, 1.0),
            recombination=0.9,
            polish=False,
            callback=lambda x, convergence: None
        )
        
        if result.success and result.fun < best_vol:
            best_positions = result.x.reshape(N, N, 2)
            best_vol = result.fun
            # Recompute with more samples for accuracy
            best_vol = _compute_volume(best_positions, n_samples=200000)
            
        # Check time
        if time.time() > deadline - 60:
            return best_positions.reshape(N * N, 2)
        
        # Polish with Nelder-Mead
        if np.isfinite(best_vol):
            result2 = minimize(
                objective_flat,
                best_positions.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            
            if result2.fun < best_vol:
                best_positions = result2.x.reshape(N, N, 2)
                best_vol = result2.fun
                # Recompute with more samples for accuracy
                best_vol = _compute_volume(best_positions, n_samples=200000)
    
    except Exception as e:
        # If optimization fails, return the initial guess
        pass
    
    # Final polish with coordinate descent
    # We can try to optimize each coordinate separately using a 1D minimizer
    if time.time() < deadline - 30:
        for iter in range(3):  # A few passes of coordinate descent
            for k in range(N * N * 2):
                # Create a 1D objective function for coordinate k
                def obj_1d(val, k=k):
                    vec = best_positions.flatten().copy()
                    vec[k] = val
                    return objective_flat(vec)
                
                # Find the current value
                current_val = best_positions.flatten()[k]
                
                # Use a bounded scalar minimizer
                # We need to find reasonable bounds for this coordinate
                # Let's use [current_val - 0.5, current_val + 0.5]
                bounds_1d = (current_val - 0.5, current_val + 0.5)
                
                try:
                    res = minimize(
                        obj_1d,
                        current_val,
                        method='bounded',
                        bounds=[bounds_1d],
                        options={'xatol': 1e-6}
                    )
                    
                    if res.fun < best_vol:
                        best_positions.flatten()[k] = res.x[0]
                        best_vol = res.fun
                except:
                    pass
            
            # Recompute the volume with more samples after each pass
            best_vol = _compute_volume(best_positions, n_samples=200000)
            
            # Check time
            if time.time() > deadline - 60:
                break
    
    # Ensure the result is valid
    final_positions = best_positions.reshape(N * N, 2)
    final_vol = _compute_volume(final_positions, n_samples=200000)
    
    if not np.isfinite(final_vol) or final_vol <= 0:
        # Return the initial guess if the optimized result is invalid
        final_positions = positions_init
    
    return final_positions


# EVOLVE-BLOCK-END
