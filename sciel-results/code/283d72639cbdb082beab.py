"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = 150.0  # Safety margin for the 150s limit
    
    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)
    
    n = len(x)
    if n == 0:
        return np.zeros(6)
    
    # Create a validation split from training data to select hyperparameters
    # Use a deterministic split to ensure reproducibility if random seed is fixed, 
    # but here we just split by index for simplicity and speed.
    split_idx = int(n * 0.8)
    x_train, x_val = x[:split_idx], x[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # We will test a few regularization strengths
    lambdas = [0.0, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0]
    
    # Candidate 1: Odd powers only (c1, c3, c5)
    # This aligns with the "Knowledge" that antisymmetry improves generalization.
    # Design matrix for odd powers: [x, x^3, x^5]
    A_train_odd = np.column_stack([x_train, x_train**3, x_train**5])
    A_val_odd = np.column_stack([x_val, x_val**3, x_val**5])
    
    # Candidate 2: Full powers (c1..c5, c0=0 usually, but let's allow c0..c5? 
    # The prompt says "sum(c_i*x**i)". Usually c0 is constant. 
    # However, the previous best used odd powers. Let's stick to the proven winner family 
    # but maybe allow a constant term if it helps? 
    # The knowledge says "restricting the fit to odd powers... significantly improving".
    # So we prioritize odd-only.
    
    best_val_mse = float('inf')
    best_coeff_odd = np.zeros(3)
    best_lambda_odd = 0.0
    
    # Evaluate odd-only models
    for lam in lambdas:
        if time.time() > deadline - 5.0:
            break
            
        # Regularized least squares: (A^T A + lam*I) c = A^T y
        # Note: We don't regularize the intercept if we had one, but here we don't.
        # All terms are regularized.
        try:
            A_t_A = A_train_odd.T @ A_train_odd
            A_t_y = A_train_odd.T @ y_train
            reg_matrix = A_t_A + lam * np.eye(3)
            c_odd = np.linalg.solve(reg_matrix, A_t_y)
            
            # Calculate validation MSE
            pred_val = A_val_odd @ c_odd
            mse = np.mean((pred_val - y_val)**2)
            
            if mse < best_val_mse:
                best_val_mse = mse
                best_coeff_odd = c_odd
                best_lambda_odd = lam
        except np.linalg.LinAlgError:
            continue

    # Candidate 3: Linear only (subset of odd)
    # Sometimes higher order terms are pure noise.
    A_train_lin = np.column_stack([x_train])
    A_val_lin = np.column_stack([x_val])
    best_val_mse_lin = float('inf')
    best_coeff_lin = 0.0
    best_lambda_lin = 0.0
    
    for lam in lambdas:
        if time.time() > deadline - 5.0:
            break
        try:
            A_t_A = A_train_lin.T @ A_train_lin
            A_t_y = A_train_lin.T @ y_train
            reg_matrix = A_t_A + lam * np.eye(1)
            c_lin = np.linalg.solve(reg_matrix, A_t_y)[0]
            
            pred_val = A_val_lin @ c_lin
            mse = np.mean((pred_val - y_val)**2)
            
            if mse < best_val_mse_lin:
                best_val_mse_lin = mse
                best_coeff_lin = c_lin
                best_lambda_lin = lam
        except np.linalg.LinAlgError:
            continue

    # Compare best odd-only vs best linear
    # We need to compare on the same validation set
    # best_val_mse is for odd-only, best_val_mse_lin is for linear
    if best_val_mse_lin < best_val_mse:
        # Use linear model
        # Refit on full training data with the best lambda
        try:
            A_full_lin = np.column_stack([x])
            A_t_A = A_full_lin.T @ A_full_lin
            A_t_y = A_full_lin.T @ y
            reg_matrix = A_t_A + best_lambda_lin * np.eye(1)
            c_final = np.linalg.solve(reg_matrix, A_t_y)[0]
            coeff = np.zeros(6)
            coeff[1] = c_final
            return coeff
        except:
            # Fallback to zero
            return np.zeros(6)
    else:
        # Use odd-only model
        # Refit on full training data with the best lambda
        try:
            A_full_odd = np.column_stack([x, x**3, x**5])
            A_t_A = A_full_odd.T @ A_full_odd
            A_t_y = A_full_odd.T @ y
            reg_matrix = A_t_A + best_lambda_odd * np.eye(3)
            c_final = np.linalg.solve(reg_matrix, A_t_y)
            coeff = np.zeros(6)
            coeff[1] = c_final[0]
            coeff[3] = c_final[1]
            coeff[5] = c_final[2]
            return coeff
        except:
            # Fallback to zero
            return np.zeros(6)

# EVOLVE-BLOCK-END
