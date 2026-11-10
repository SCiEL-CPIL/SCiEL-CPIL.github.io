import numpy as np
import time
import sys

def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial approximation to the unresolved fast-scale forcing.
    
    Args:
        training_x: Array of slow-state values.
        training_subgrid_tendency: Array of corresponding unresolved tendencies.
        
    Returns:
        Array of 6 finite real polynomial coefficients in ASCENDING powers of x.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

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

    # Scale x to [-1, 1] for Chebyshev fitting
    x_min = np.min(x)
    x_max = np.max(x)
    x_range = x_max - x_min
    if x_range < 1e-12:
        x_range = 1.0
        x_min = x_min - 0.5
        x_max = x_min + 0.5

    # Linear map: x_scaled = A * x + B
    A = 2.0 / x_range
    B = - (x_max + x_min) / x_range
    x_scaled = A * x + B

    # Set up cross-validation folds
    k_folds = 5
    rng = np.random.RandomState(42)
    indices = rng.permutation(n)
    fold_size = n // k_folds
    folds = [indices[i * fold_size: (i + 1) * fold_size] for i in range(k_folds)]
    if len(folds[-1]) < fold_size:
        folds[-1] = np.concatenate([folds[-1], indices[k_folds * fold_size:]])

    # Degrees to consider (0 to 5)
    degrees = [0, 1, 2, 3, 4, 5]
    # Regularization parameters for Ridge
    # We use a wide range to allow for both near-OLS and strong regularization
    alphas = np.logspace(-5, 5, 50)

    best_cv_mse = np.inf
    best_degree = 0
    best_alpha = 1.0
    best_theta_cheb = np.zeros(6)

    # Pre-compute Chebyshev basis for max degree 5
    # T_k(x) for k=0..5
    # Using recurrence: T_0=1, T_1=x, T_{k+1}=2xT_k - T_{k-1}
    V_cheb_full = np.zeros((n, 6))
    V_cheb_full[:, 0] = 1.0
    V_cheb_full[:, 1] = x_scaled
    for k in range(2, 6):
        V_cheb_full[:, k] = 2.0 * x_scaled * V_cheb_full[:, k-1] - V_cheb_full[:, k-2]

    for deg in degrees:
        if time.time() > deadline:
            break
            
        d = deg + 1  # number of basis functions
        V_cheb = V_cheb_full[:, :d]
        
        alpha_mse_sums = np.zeros(len(alphas))
        
        for i in range(k_folds):
            if time.time() > deadline:
                break

            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(k_folds) if j != i])

            V_train = V_cheb[train_idx]
            y_train = y[train_idx]
            V_val = V_cheb[val_idx]
            y_val = y[val_idx]

            # Precompute V_train.T @ V_train and V_train.T @ y_train
            G = V_train.T @ V_train
            g = V_train.T @ y_train
            
            # Solve for multiple alphas at once using diagonal loading
            # (G + alpha*I) theta = g
            # We can use eigen-decomposition of G for efficiency since G is small (6x6 max)
            try:
                eigvals, eigvecs = np.linalg.eigh(G)
            except np.linalg.LinAlgError:
                continue
            
            eigvecs_T_g = eigvecs.T @ g
            # denom shape (d, len(alphas))
            denom = eigvals[:, np.newaxis] + alphas[np.newaxis, :]
            # theta_batch shape (d, len(alphas))
            theta_batch = eigvecs @ (eigvecs_T_g[:, np.newaxis] / denom)
            
            # Predict on validation set
            # y_pred_batch shape (n_val, len(alphas))
            y_pred_batch = V_val @ theta_batch
            residuals = y_pred_batch - y_val[:, np.newaxis]
            mse_per_alpha = np.mean(residuals ** 2, axis=0)
            alpha_mse_sums += mse_per_alpha

        if time.time() > deadline:
            break

        min_idx = np.argmin(alpha_mse_sums)
        avg_cv_mse = alpha_mse_sums[min_idx] / k_folds

        if avg_cv_mse < best_cv_mse:
            best_cv_mse = avg_cv_mse
            best_degree = deg
            best_alpha = alphas[min_idx]
            # We don't store theta here, we refit at the end to ensure it's on full data

    # Refit on full data with best parameters
    d = best_degree + 1
    V_cheb = V_cheb_full[:, :d]
    
    G = V_cheb.T @ V_cheb
    g = V_cheb.T @ y
    
    # Add regularization
    reg_matrix = np.eye(d) * best_alpha
    A_mat = G + reg_matrix
    b_vec = g
    
    try:
        theta_cheb = np.linalg.solve(A_mat, b_vec)
    except np.linalg.LinAlgError:
        theta_cheb = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
    
    # Pad theta_cheb to length 6 if best_degree < 5
    if len(theta_cheb) < 6:
        theta_cheb_padded = np.zeros(6)
        theta_cheb_padded[:len(theta_cheb)] = theta_cheb
    else:
        theta_cheb_padded = theta_cheb[:6]
        
    # Convert Chebyshev coefficients to Monomial coefficients
    # np.polynomial.chebyshev.cheb2poly converts from Chebyshev basis to power basis
    # Note: cheb2poly expects coefficients in ascending order of Chebyshev degree
    try:
        theta_poly_scaled = np.polynomial.chebyshev.cheb2poly(theta_cheb_padded)
    except Exception:
        # Fallback: simple OLS in monomial basis if conversion fails
        V_mono = np.vander(x_scaled, 6, increasing=True)
        try:
            theta_poly_scaled = np.linalg.lstsq(V_mono, y, rcond=None)[0]
        except:
            theta_poly_scaled = np.array([np.mean(y), 0, 0, 0, 0, 0])
            
    # theta_poly_scaled is now in powers of x_scaled
    # We need to convert to powers of original x
    # x_scaled = A * x + B
    # We want coefficients c_i such that sum(c_i * x^i) = sum(d_i * (A*x+B)^i)
    
    result_poly = np.zeros(6, dtype=np.float64)
    
    # Base polynomial for (A*x + B) is [B, A]
    base_poly = np.array([B, A], dtype=np.float64)
    
    current_power = np.array([1.0], dtype=np.float64)  # (Ax+B)^0
    
    for i in range(6):
        coeff = theta_poly_scaled[i] if i < len(theta_poly_scaled) else 0.0
        if coeff != 0.0:
            # Multiply current_power by coeff and add to result
            # Ensure current_power is padded to length 6
            cp_padded = np.pad(current_power, (0, 6 - len(current_power)), mode='constant')
            result_poly += coeff * cp_padded
        
        # Update current_power for next iteration
        if i < 5:
            current_power = np.polynomial.polynomial.polymul(current_power, base_poly)
            if len(current_power) > 6:
                current_power = current_power[:6]
                
    # Ensure finite
    if not np.all(np.isfinite(result_poly)):
        result_poly = np.array([np.mean(y), 0, 0, 0, 0, 0], dtype=np.float64)
        
    return result_poly
