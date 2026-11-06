"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)
    
    n_samples = len(x)
    if n_samples < 10:
        return np.zeros(6)
    
    # Split data for model selection
    # Use a fixed shuffle for reproducibility if needed, but random is fine here
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    split_idx = int(0.8 * n_samples)
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]
    
    x_train = x[train_idx]
    y_train = y[train_idx]
    x_val = x[val_idx]
    y_val = y[val_idx]
    
    best_mse = np.inf
    best_coeff = np.zeros(6)
    best_degree = 1
    best_lambda = 1e-6
    
    # Regularization strengths to try
    # We want to cover a range that balances bias and variance
    lambdas = [0.0, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0, 100.0]
    
    for degree in range(1, 6):
        if time.time() > deadline:
            break
            
        # Build Vandermonde matrix for training set
        # X_v[:, i] = x_train ** i
        V_train = np.vander(x_train, N=degree + 1, increasing=True)
        V_val = np.vander(x_val, N=degree + 1, increasing=True)
        
        for lam in lambdas:
            if time.time() > deadline:
                break
                
            # Ridge Regression: (V^T V + lambda * I) c = V^T y
            # Note: We usually don't regularize the bias term (intercept), 
            # but for simplicity and since x is centered-ish in L96, 
            # regularizing all terms is often acceptable or we can adjust.
            # Let's regularize all terms for simplicity, or just the higher ones?
            # Standard Ridge regularizes all coefficients.
            
            try:
                # Add small jitter to diagonal for stability if lam=0
                reg_matrix = V_train.T @ V_train
                if lam > 0:
                    reg_matrix = reg_matrix + lam * np.eye(degree + 1)
                else:
                    # If lam is 0, use lstsq or add tiny jitter
                    reg_matrix = reg_matrix + 1e-12 * np.eye(degree + 1)
                    
                rhs = V_train.T @ y_train
                
                coeff_candidate = np.linalg.solve(reg_matrix, rhs)
                
                # Evaluate on validation set
                pred_val = V_val @ coeff_candidate
                mse_val = np.mean((pred_val - y_val) ** 2)
                
                if mse_val < best_mse:
                    best_mse = mse_val
                    best_degree = degree
                    best_lambda = lam
                    # Store temporary best coeff (will refit later)
                    best_coeff_temp = coeff_candidate.copy()
            except np.linalg.LinAlgError:
                continue
                
    # Refit with the best degree and lambda on the FULL training data
    if time.time() < deadline:
        V_full = np.vander(x, N=best_degree + 1, increasing=True)
        reg_matrix = V_full.T @ V_full
        if best_lambda > 0:
            reg_matrix = reg_matrix + best_lambda * np.eye(best_degree + 1)
        else:
            reg_matrix = reg_matrix + 1e-12 * np.eye(best_degree + 1)
        
        rhs = V_full.T @ y
        
        try:
            final_coeff = np.linalg.solve(reg_matrix, rhs)
        except np.linalg.LinAlgError:
            # Fallback to linear fit if something went wrong
            final_coeff = np.zeros(6)
            final_coeff[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
            
        # Pad to length 6
        result = np.zeros(6)
        result[:best_degree + 1] = final_coeff
        return result
    else:
        # Time out, return the best found so far (padded)
        result = np.zeros(6)
        result[:best_degree + 1] = best_coeff_temp
        return result
# EVOLVE-BLOCK-END
