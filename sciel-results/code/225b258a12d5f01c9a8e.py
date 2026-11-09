"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    t_start = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    
    n = min(x.size, y.size)
    if n == 0:
        return np.zeros(6, dtype=np.float64)
    
    x = x[:n]
    y = y[:n]
    
    # Remove non-finite values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = x.size
    
    if n == 0:
        return np.zeros(6, dtype=np.float64)
    
    # Strategy:
    # 1. Estimate the conditional expectation E[Y|X] using an ensemble of
    #    Local Weighted Quadratic Regressions (LOESS-like) with multiple bandwidths.
    #    This avoids the bias of a single fixed bandwidth and the high variance
    #    of raw data. Averaging over scales provides a robust denoiser.
    # 2. Fit a degree-5 polynomial to these ensemble-averaged pseudo-labels.
    
    # Define a grid for evaluating the conditional mean
    x_min, x_max = np.min(x), np.max(x)
    # Add buffer to ensure coverage if validation x is slightly outside training range
    # but not too far to avoid wild extrapolation.
    buffer = 0.1 * (x_max - x_min) if x_max > x_min else 1.0
    x_min -= buffer
    x_max += buffer
    
    # Use a sufficiently dense grid for accurate polynomial fitting
    num_grid = 400
    x_grid = np.linspace(x_min, x_max, num_grid)
    
    # Sort training data for potential vectorization or just for stability
    # Actually, for weighted least squares, sorting isn't strictly necessary but helps if we were doing local windows.
    # Here we use global weighted solves per grid point? No, that's too slow.
    # We need an efficient way to compute weighted quadratic fits at x_grid points.
    
    # Efficient approach:
    # For each bandwidth h, compute weights w_i = exp(-0.5 * (x_i - x_g)^2 / h^2) for all i, for a given grid point x_g.
    # Then solve for quadratic coefficients: (x_i - x_g) is the local variable?
    # Or just fit global quadratic with weights.
    # Fitting global quadratic with weights is equivalent to local linear/quadratic if weights are localized.
    
    # To speed up, we can subsample training points if n is very large.
    # But weighted LS is O(n * num_params^3) per grid point? No, O(n * num_params) for normal equations if we precompute moments?
    # Let's precompute moments for each bandwidth?
    # The weights depend on x_grid, so moments change.
    
    # Alternative: Use a fixed set of "centers" or just evaluate directly.
    # With n~10k and grid~400, and say 10 bandwidths, that's 400 * 10 * 10k = 40M operations.
    # Weighted LS for quadratic (3 params) is fast.
    
    # Let's subsample to max 5000 points to ensure speed.
    max_train_pts = 5000
    if n > max_train_pts:
        # Random subsampling
        idx = np.random.choice(n, max_train_pts, replace=False)
        x_sub = x[idx]
        y_sub = y[idx]
    else:
        x_sub = x
        y_sub = y
        max_train_pts = n
        
    n_sub = x_sub.size
    
    # Bandwidths to try
    # Range of x
    x_range = x_max - x_min
    if x_range < 1e-6:
        x_range = 1.0
        
    # Heuristic bandwidths: from very local to more global
    # Typical scale of structure in Lorenz-96 forcing is related to the fast scale dynamics.
    # Let's try log-spaced bandwidths.
    h_min = x_range / 50.0
    h_max = x_range / 2.0
    if h_min < 1e-6: h_min = 1e-6
    
    num_bandwidths = 15
    bandwidths = np.geomspace(h_min, h_max, num_bandwidths)
    
    y_ensemble = np.zeros(num_grid)
    
    # Precompute x_sub for broadcasting
    # We will loop over bandwidths, then over grid points?
    # Or loop over grid points, then over bandwidths?
    # Vectorizing over grid points for a fixed bandwidth is better.
    
    # For a fixed bandwidth h:
    # We want to compute weighted quadratic fit at each x_g.
    # This is hard to fully vectorize over x_g because the design matrix depends on x_g (if using local coords) or weights depend on x_g.
    # If we use global quadratic basis [1, x, x^2] with weights w_i(x_g):
    # The normal equations are:
    # [sum w_i, sum w_i x_i, sum w_i x_i^2] [a0]   [sum w_i y_i]
    # [sum w_i x_i, sum w_i x_i^2, sum w_i x_i^3] [a1] = [sum w_i y_i x_i]
    # [sum w_i x_i^2, sum w_i x_i^3, sum w_i x_i^4] [a2]   [sum w_i y_i x_i^2]
    
    # We can precompute powers of x_sub: x1, x2, x3, x4.
    x1 = x_sub
    x2 = x_sub * x_sub
    x3 = x2 * x_sub
    x4 = x2 * x2
    
    # y is y_sub
    
    # For each bandwidth h:
    #   Weights W (num_grid, n_sub) = exp(-0.5 * (x_grid[:,None] - x_sub[None,:] )**2 / h^2)
    #   This matrix might be large: 400 * 5000 = 2M floats. 16MB. OK.
    #   Then compute moments M (num_grid, 5)
    #   M[:, 0] = sum(W, axis=1)
    #   M[:, 1] = sum(W * x1, axis=1)
    #   M[:, 2] = sum(W * x2, axis=1)
    #   M[:, 3] = sum(W * x3, axis=1)
    #   M[:, 4] = sum(W * x4, axis=1)
    #   RHS (num_grid, 3)
    #   R[:, 0] = sum(W * y, axis=1)
    #   R[:, 1] = sum(W * y * x1, axis=1)
    #   R[:, 2] = sum(W * y * x2, axis=1)
    
    #   Then solve 3x3 system for each grid point.
    #   We can vectorize the solve using np.linalg.solve on a batch of matrices.
    
    # Let's implement this loop over bandwidths.
    
    for h in bandwidths:
        if time.time() - t_start > deadline - 10.0:
            break
            
        # Compute distances squared
        # x_grid: (G,), x_sub: (N,)
        # dist2: (G, N)
        diff = x_grid[:, None] - x_sub[None, :]
        dist2 = diff * diff
        
        # Weights
        W = np.exp(-0.5 * dist2 / (h * h))
        
        # Avoid zero weights leading to singular matrices (should be rare with exp, but safe)
        # Sum weights
        sum_w = np.sum(W, axis=1)
        
        # Compute moments
        # M: (G, 5)
        # We can use matrix multiplication if we stack powers?
        # W @ x1 is (G, N) @ (N,) -> (G,)
        # W @ x2 -> (G,)
        # etc.
        
        m0 = np.sum(W, axis=1)
        m1 = W @ x1
        m2 = W @ x2
        m3 = W @ x3
        m4 = W @ x4
        
        r0 = W @ y_sub
        r1 = W @ (y_sub * x1)
        r2 = W @ (y_sub * x2)
        
        # Construct the 3x3 matrices A and vector b for each grid point
        # A_i = [[m0_i, m1_i, m2_i],
        #        [m1_i, m2_i, m3_i],
        #        [m2_i, m3_i, m4_i]]
        # b_i = [r0_i, r1_i, r2_i]
        
        # Create array of matrices
        # Shape (G, 3, 3)
        A_batch = np.zeros((num_grid, 3, 3))
        A_batch[:, 0, 0] = m0
        A_batch[:, 0, 1] = m1
        A_batch[:, 0, 2] = m2
        A_batch[:, 1, 0] = m1
        A_batch[:, 1, 1] = m2
        A_batch[:, 1, 2] = m3
        A_batch[:, 2, 0] = m2
        A_batch[:, 2, 1] = m3
        A_batch[:, 2, 2] = m4
        
        b_batch = np.column_stack((r0, r1, r2))
        
        # Add regularization to diagonal to handle potential ill-conditioning
        # The trace of A is m0 + m2 + m4.
        # Regularization: lambda * I.
        # If weights are small, m0 is small.
        # Let's use a small relative regularization.
        # trace = np.trace(A_batch, axis1=1, axis2=2)
        # reg = 1e-6 * trace
        # A_batch[:, 0, 0] += reg
        # A_batch[:, 1, 1] += reg
        # A_batch[:, 2, 2] += reg
        
        # Solve
        try:
            # np.linalg.solve supports batched inputs
            coeff_batch = np.linalg.solve(A_batch, b_batch)
        except np.linalg.LinAlgError:
            # Fallback to lstsq if singular
            try:
                # Reshape for lstsq? lstsq doesn't support batched 3x3 directly in same way?
                # Actually, np.linalg.lstsq works on 2D. We need to loop or use scipy.
                # Let's just use lstsq on the flattened batch? No.
                # Let's use a loop for robustness in this rare case, or just skip this bandwidth.
                # Given the exp kernel, singularity is unlikely unless all weights are ~0.
                # If m0 is very small, the fit is unstable.
                # Let's set the contribution to 0 for this bandwidth if solve fails.
                coeff_batch = np.zeros((num_grid, 3))
            except Exception:
                coeff_batch = np.zeros((num_grid, 3))
                
        # Evaluate quadratic at x_grid
        # y_est = a0 + a1*x + a2*x^2
        # coeff_batch is (G, 3) -> a0, a1, a2
        a0 = coeff_batch[:, 0]
        a1 = coeff_batch[:, 1]
        a2 = coeff_batch[:, 2]
        
        y_est_h = a0 + a1 * x_grid + a2 * x_grid * x_grid
        
        # Add to ensemble
        y_ensemble += y_est_h
        
    if time.time() - t_start > deadline - 5.0:
        # Fallback: simple global polynomial fit on raw data
        A = np.vander(x, 6, increasing=True)
        coeff_asc, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        return coeff_asc
        
    # Average the ensemble
    y_pseudo = y_ensemble / num_bandwidths
    
    # Check time
    if time.time() - t_start > deadline - 2.0:
        A = np.vander(x_grid, 6, increasing=True)
        coeff_asc, _, _, _ = np.linalg.lstsq(A, y_pseudo, rcond=None)
        return coeff_asc

    # Fit degree-5 polynomial to the pseudo-labels
    # Weight the fit? Uniform is fine since grid is uniform.
    A = np.vander(x_grid, 6, increasing=True)
    try:
        coeff_asc, residuals, rank, sv = np.linalg.lstsq(A, y_pseudo, rcond=None)
        
        if not np.all(np.isfinite(coeff_asc)):
            # Fallback to lower degree
            A_lin = np.vander(x_grid, 3, increasing=True)
            coeff_lin, _, _, _ = np.linalg.lstsq(A_lin, y_pseudo, rcond=None)
            coeff_asc = np.zeros(6)
            coeff_asc[:3] = coeff_lin
             
    except Exception:
        try:
            A_lin = np.vander(x, 3, increasing=True)
            coeff_lin, _, _, _ = np.linalg.lstsq(A_lin, y, rcond=None)
            coeff_asc = np.zeros(6)
            coeff_asc[:3] = coeff_lin
        except Exception:
            return np.zeros(6, dtype=np.float64)
            
    return coeff_asc
# EVOLVE-BLOCK-END
