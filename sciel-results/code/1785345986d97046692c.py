# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def kakeya_needle_3d8() -> np.ndarray:
    """
    Optimizes the 3D Kakeya set positions using L-BFGS-B on a smoothed volume objective.
    
    The problem is to find base positions (x_k, y_k) for 64 tubes (i, j) where i, j in [0, 7].
    Tube (i, j) at height c is [x_i + c*i/8, x_i + c*i/8 + 1/8] x [y_j + c*j/8, y_j + c*j/8 + 1/8].
    
    We model the volume as the integral over c of the area of the union of the 2D rectangles.
    Area of union of rectangles R_k(c) = sum Area(R_k) - sum overlaps + ...
    Instead of exact inclusion-exclusion (expensive), we use a Monte-Carlo-like grid integration
    for the objective function value, but we need gradients.
    
    Alternative: Use a differentiable proxy for the union volume.
    The volume is V = ∫_0^1 Area(Union_k R_{i,j}(c)) dc.
    
    Let's define the indicator function for a point (u, v) at height c to be inside tube (i,j):
    I_{i,j}(u, v, c) = 1 if u in [x_i + c*i/8, x_i + c*i/8 + 1/8] AND v in [y_j + c*j/8, y_j + c*j/8 + 1/8].
    
    The union indicator is max_{i,j} I_{i,j}.
    This max is non-differentiable. We can approximate it with a smooth max (e.g., log-sum-exp)
    or simply use a fixed grid of sample points (u, v, c) and compute the probability that a point
    is covered, then scale by the bounding box volume.
    
    To get gradients, we can define the objective as the sum of smoothed indicators over a grid.
    Let G be a grid of points (u_g, v_g, c_g).
    Objective(params) = (1/|G|) * sum_g [ max_{i,j} SmoothedIndicator_{i,j}(u_g, v_g, c_g) ] * BoxVolume
    
    SmoothedIndicator can be a product of two smooth step functions.
    SmoothedStep(x) = 1 / (1 + exp(-x/epsilon))
    
    Indicator for x-interval [L, R]:
    S((u - L)/eps) * (1 - S((u - R)/eps))
    
    This is differentiable everywhere.
    
    The box volume depends on params (X0, X1, Y0, Y1).
    X0 = min_i x_i
    X1 = max_i (x_i + (i+1)/8)
    Y0 = min_j y_j
    Y1 = max_j (y_j + (j+1)/8)
    
    BoxVolume = (X1 - X0) * (Y1 - Y0)
    
    Note: The Monte Carlo estimate in the task uses a specific rng and box.
    However, the score is 1/V where V is the MC estimate.
    Minimizing the true volume (or a good approximation) should minimize the MC estimate.
    
    We will use a grid of points for the objective function to ensure differentiability.
    """
    N = 8
    start_time = time.time()
    deadline = start_time + 590.0  # Leave margin for safety

    # Parameters: 16 values. x_0..x_7, y_0..y_7
    # Initial guess: Keich construction
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    
    x_init = keich.copy()
    y_init = keich.copy()
    params_init = np.concatenate([x_init, y_init])
    
    # Grid for integration
    # We need enough points to estimate the volume well, but not too many for speed.
    # 20x20x20 = 8000 points.
    # 30x30x30 = 27000 points.
    # Let's start with 25x25x25 = 15625 points.
    grid_size = 25
    u_grid = np.linspace(0, 1, grid_size)
    v_grid = np.linspace(0, 1, grid_size)
    c_grid = np.linspace(0, 1, grid_size)
    
    # Create meshgrid
    U, V, C = np.meshgrid(u_grid, v_grid, c_grid, indexing='ij')
    U_flat = U.flatten()
    V_flat = V.flatten()
    C_flat = C.flatten()
    
    num_points = len(U_flat)
    
    # Epsilon for smoothing
    eps = 0.01
    
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))
    
    def objective(params):
        if time.time() > deadline:
            return 1e10 # Penalty for timeout
        
        x = params[:N]
        y = params[N:]
        
        # Compute box bounds
        # X0 = min x_i
        # X1 = max (x_i + (i+1)/8)
        # Note: The tube width is 1/8.
        # At height c, x range is [x_i + c*i/N, x_i + c*i/N + 1/N]
        # Max x extent is at c=1: x_i + i/N + 1/N = x_i + (i+1)/N
        # Min x extent is at c=0: x_i
        
        x_offsets = np.arange(N) / N
        x_max_offsets = (np.arange(N) + 1) / N
        
        X0 = np.min(x)
        X1 = np.max(x + x_max_offsets)
        
        y_offsets = np.arange(N) / N
        y_max_offsets = (np.arange(N) + 1) / N
        
        Y0 = np.min(y)
        Y1 = np.max(y + y_max_offsets)
        
        box_vol = (X1 - X0) * (Y1 - Y0)
        
        if box_vol <= 1e-9:
            return 1e10
            
        # Normalize grid points to box coordinates?
        # The grid U, V, C are in [0,1].
        # We need to map them to the actual box [X0, X1] x [Y0, Y1] x [0,1]
        # Or rather, the tube positions are defined in absolute coordinates.
        # The MC estimator samples from the box [X0, X1] x [Y0, Y1] x [0,1].
        # So we should sample from this box.
        
        # Let's generate the sample points in the box for this objective.
        # To make it differentiable, we can treat the box bounds as fixed during the gradient step?
        # No, the box bounds depend on params.
        # This makes the objective highly non-linear and potentially ill-conditioned if the box changes.
        
        # Alternative: Fix the box to the initial bounds?
        # No, the task defines the box dynamically.
        
        # Let's use the current box bounds for the grid mapping.
        # u_actual = X0 + U_flat * (X1 - X0)
        # v_actual = Y0 + V_flat * (Y1 - Y0)
        # c_actual = C_flat
        
        u_act = X0 + U_flat * (X1 - X0)
        v_act = Y0 + V_flat * (Y1 - Y0)
        c_act = C_flat
        
        # Compute indicators for all tubes
        # Tube (i, j)
        # x_start = x[i] + c_act * i / N
        # x_end = x_start + 1/N
        # y_start = y[j] + c_act * j / N
        # y_end = y_start + 1/N
        
        # We need max over all i, j of I_{i,j}
        # I_{i,j} = SmoothedStep((u_act - x_start)/eps) * (1 - SmoothedStep((u_act - x_end)/eps)) * 
        #           SmoothedStep((v_act - y_start)/eps) * (1 - SmoothedStep((v_act - y_end)/eps))
        
        # To do this efficiently, we can vectorize over i and j.
        # Shape of u_act: (num_points,)
        # Shape of x: (N,)
        # We want an array of shape (num_points, N, N)
        
        # This might be memory heavy. 15625 * 8 * 8 = ~1M floats. OK.
        
        # x_starts: (1, N, 1) + (c_act, 1, 1) * (i/N)
        # i_vals = np.arange(N) / N
        # j_vals = np.arange(N) / N
        
        i_vals = np.arange(N) / N
        j_vals = np.arange(N) / N
        
        # x_start[i] = x[i] + c * i/N
        # x_start shape (N,) -> broadcast to (1, N, 1) ? No.
        # Let's construct the arrays carefully.
        
        # c_act is (num_points,)
        # i_vals is (N,)
        # x is (N,)
        
        # x_starts[i] for all points:
        # x_start = x[i] + c_act * i_vals[i]
        # We want a matrix of shape (num_points, N) for x_starts
        # x_starts = c_act[:, None] * i_vals[None, :] + x[None, :]
        
        # Similarly for x_ends
        # x_ends = x_starts + 1/N
        
        # y_starts = c_act[:, None] * j_vals[None, :] + y[None, :]
        # y_ends = y_starts + 1/N
        
        x_starts = c_act[:, None] * i_vals[None, :] + x[None, :]
        x_ends = x_starts + 1.0 / N
        
        y_starts = c_act[:, None] * j_vals[None, :] + y[None, :]
        y_ends = y_starts + 1.0 / N
        
        # Now we have x_starts (num_points, N), x_ends (num_points, N)
        # y_starts (num_points, N), y_ends (num_points, N)
        
        # We need to combine i and j.
        # For a given point p, the indicator for tube (i, j) is:
        # I_x(p, i) * I_y(p, j)
        
        # I_x(p, i) = S((u_act[p] - x_starts[p, i])/eps) * (1 - S((u_act[p] - x_ends[p, i])/eps))
        # I_y(p, j) = S((v_act[p] - y_starts[p, j])/eps) * (1 - S((v_act[p] - y_ends[p, j])/eps))
        
        # Let's compute I_x (num_points, N) and I_y (num_points, N)
        
        dx_start = (u_act[:, None] - x_starts) / eps
        dx_end = (u_act[:, None] - x_ends) / eps
        
        I_x = sigmoid(dx_start) * (1.0 - sigmoid(dx_end))
        
        dy_start = (v_act[:, None] - y_starts) / eps
        dy_end = (v_act[:, None] - y_ends) / eps
        
        I_y = sigmoid(dy_start) * (1.0 - sigmoid(dy_end))
        
        # Now we need max over i, j of I_x[p, i] * I_y[p, j]
        # This is equivalent to (max_i I_x[p, i]) * (max_j I_y[p, j])?
        # No, max_{i,j} (A_i B_j) = (max_i A_i) * (max_j B_j) if A_i, B_j >= 0.
        # Yes, because max_{i,j} A_i B_j = max_i (A_i * max_j B_j) = max_i (A_i * M_j) = M_i * M_j.
        
        max_I_x = np.max(I_x, axis=1)
        max_I_y = np.max(I_y, axis=1)
        
        union_indicator = max_I_x * max_I_y
        
        # Volume estimate
        # The MC estimator is: fraction * BoxVol
        # Our grid estimate is: mean(union_indicator) * BoxVol
        
        vol = np.mean(union_indicator) * box_vol
        
        return vol

    # Run L-BFGS-B
    best_params = params_init.copy()
    best_vol = objective(best_params)
    
    # We can try a few restarts or just one long run
    # L-BFGS-B is local, so good initialization is key.
    # We can also try to anneal epsilon?
    # Start with larger eps for smoothness, then decrease?
    # For simplicity, let's run with eps=0.01 and see.
    
    # To improve robustness, we can run multiple iterations with different initializations
    # or slightly perturbed Keich.
    
    # Let's try to optimize.
    # Time check
    if time.time() < deadline - 10:
        try:
            res = minimize(
                objective, 
                params_init, 
                method='L-BFGS-B', 
                options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-8}
            )
            if res.fun < best_vol:
                best_params = res.x
                best_vol = res.fun
        except Exception:
            pass
    
    # If time remains, try a second pass with smaller epsilon for refinement?
    # Or just use the result.
    
    x_opt = best_params[:N]
    y_opt = best_params[N:]
    
    # Construct the 64 positions
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = x_opt[i_flat]
    base_y = y_opt[j_flat]
    
    positions = np.stack([base_x, base_y], axis=1)
    
    return positions

# EVOLVE-BLOCK-END
