import numpy as np
import time

def _local_weighted_poly(x_train, y_train, x_eval, bandwidth, degree=3):
    """Local weighted polynomial regression with Gaussian kernel."""
    n = len(x_train)
    if n == 0:
        return np.zeros(len(x_eval))
    
    # Compute weights for each evaluation point
    # For efficiency, we'll process in batches
    predictions = np.zeros(len(x_eval))
    
    # Pre-compute scaled x for training
    x_min = np.min(x_train)
    x_max = np.max(x_train)
    x_range = x_max - x_min
    if x_range < 1e-12:
        x_range = 1.0
    
    # Build Vandermonde matrix for training
    V_train = np.vander(x_train, degree + 1, increasing=True)
    
    # Process evaluation points in batches for memory efficiency
    batch_size = min(1000, len(x_eval))
    for start in range(0, len(x_eval), batch_size):
        end = min(start + batch_size, len(x_eval))
        x_batch = x_eval[start:end]
        
        # Compute Gaussian weights: w_i = exp(-0.5 * ((x_eval - x_train_i) / bandwidth)^2)
        diff = x_batch[:, np.newaxis] - x_train[np.newaxis, :]
        weights = np.exp(-0.5 * (diff / bandwidth) ** 2)
        
        # Weighted least squares: min sum w_i * (y_i - V_i @ theta)^2
        # Equivalent to: (V^T W V) theta = V^T W y
        sqrt_w = np.sqrt(weights)
        V_weighted = V_train * sqrt_w.T  # Each row scaled by sqrt(w_i)
        y_weighted = y_train * np.sum(weights, axis=0)  # This is wrong, let me fix
        
        # Actually, weighted LS: (V^T W V) theta = V^T W y
        # W is diagonal with weights[i] on diagonal
        # V^T W V = (V * sqrt(W))^T (V * sqrt(W))
        # V^T W y = (V * sqrt(W))^T (sqrt(W) * y)
        
        Vw = V_train * sqrt_w.T  # (n, degree+1), each row i multiplied by sqrt(w_i)
        yw = y_train * weights.T  # Wait, this is also wrong
        
        # Let me redo this properly
        # For each eval point j, we have weights w_ji for training point i
        # We want to solve: (V^T W_j V) theta_j = V^T W_j y
        # where W_j is diagonal with entries w_ji
        
        # This is expensive to do for each eval point separately
        # Let's use a different approach: for each eval point, compute weighted LS
        
        for j in range(end - start):
            w = weights[j]
            # Skip if all weights are essentially zero
            if np.sum(w) < 1e-10:
                predictions[start + j] = np.mean(y_train)
                continue
            
            # Weighted normal equations
            Vw = V_train * w[:, np.newaxis]  # (n, degree+1)
            G = Vw.T @ V_train  # (degree+1, degree+1)
            g = Vw.T @ y_train  # (degree+1,)
            
            # Add small regularization for stability
            G += 1e-10 * np.eye(degree + 1)
            
            try:
                theta = np.linalg.solve(G, g)
                predictions[start + j] = np.vander(x_batch[j], degree + 1, increasing=True) @ theta
            except np.linalg.LinAlgError:
                predictions[start + j] = np.mean(y_train)
    
    return predictions

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)

    # Remove non-finite values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    n = len(x)
    if n == 0:
        return np.zeros(6, dtype=np.float64)

    if n < 10:
        mean_y = np.mean(y) if n > 0 else 0.0
        return np.array([mean_y, 0, 0, 0, 0, 0], dtype=np.float64)

    # Scale x to [-1, 1] for better conditioning
    x_min = np.min(x)
    x_max = np.max(x)
    x_range = x_max - x_min
    if x_range < 1e-12:
        x_range = 1.0
        x_min = x_min - 0.5
        x_max = x_min + 0.5

    # Linear map: x_scaled = A * x + B
    A_scale = 2.0 / x_range
    B_scale = - (x_max + x_min) / x_range
    x_scaled = A_scale * x + B_scale

    # Cross-validation to select bandwidth and degree
    k_folds = 5
    rng = np.random.RandomState(42)
    indices = rng.permutation(n)
    fold_size = n // k_folds
    folds = [indices[i * fold_size: (i + 1) * fold_size] for i in range(k_folds)]
    if len(folds[-1]) < fold_size:
        folds[-1] = np.concatenate([folds[-1], indices[k_folds * fold_size:]])

    # Grid of bandwidths to try (in scaled units)
    bandwidths = np.logspace(-2, 1, 15)
    degrees = [2, 3, 4]

    best_cv_mse = np.inf
    best_bandwidth = 0.3
    best_degree = 3

    for deg in degrees:
        if time.time() > deadline:
            break

        for bw in bandwidths:
            if time.time() > deadline:
                break

            cv_mse_sum = 0.0
            for i in range(k_folds):
                if time.time() > deadline:
                    break

                val_idx = folds[i]
                train_idx = np.concatenate([folds[j] for j in range(k_folds) if j != i])

                x_train_cv = x_scaled[train_idx]
                y_train_cv = y[train_idx]
                x_val_cv = x_scaled[val_idx]
                y_val_cv = y[val_idx]

                y_pred = _local_weighted_poly(x_train_cv, y_train_cv, x_val_cv, bw, deg)
                mse = np.mean((y_pred - y_val_cv) ** 2)
                cv_mse_sum += mse

            avg_cv_mse = cv_mse_sum / k_folds
            if avg_cv_mse < best_cv_mse:
                best_cv_mse = avg_cv_mse
                best_bandwidth = bw
                best_degree = deg

    # Fit final model using all data with best parameters
    if time.time() > deadline:
        # Return a simple fallback
        return np.array([np.mean(y), 0, 0, 0, 0, 0], dtype=np.float64)

    # Now we need to convert the local weighted regression to a global polynomial
    # We'll do this by evaluating the local model on a grid and fitting a global polynomial
    x_grid = np.linspace(x_scaled.min() - 0.1, x_scaled.max() + 0.1, 200)
    y_grid = _local_weighted_poly(x_scaled, y, x_grid, best_bandwidth, best_degree)

    # Fit a degree-5 polynomial to the grid points
    V_grid = np.vander(x_grid, 6, increasing=True)
    try:
        theta_scaled, residuals, rank, sv = np.linalg.lstsq(V_grid, y_grid, rcond=None)
    except np.linalg.LinAlgError:
        theta_scaled = np.zeros(6)

    # Convert from scaled basis to original basis
    result_poly = np.zeros(6, dtype=np.float64)
    base_poly = np.array([B_scale, A_scale], dtype=np.float64)

    current_power = np.array([1.0], dtype=np.float64)

    for i in range(6):
        if len(current_power) < 6:
            cp_padded = np.pad(current_power, (0, 6 - len(current_power)), mode='constant')
        else:
            cp_padded = current_power[:6]

        result_poly += theta_scaled[i] * cp_padded

        if i < 5:
            current_power = np.polynomial.polynomial.polymul(current_power, base_poly)
            if len(current_power) > 6:
                current_power = current_power[:6]

    # Ensure finite
    if not np.all(np.isfinite(result_poly)):
        result_poly = np.array([np.mean(y), 0, 0, 0, 0, 0], dtype=np.float64)

    return result_poly
