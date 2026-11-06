import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = 140.0  # Safety margin for 150s limit
    
    x = np.asarray(training_x, dtype=float)
    y = np.asarray(training_subgrid_tendency, dtype=float)
    
    # Handle edge cases
    if len(x) == 0:
        return np.zeros(6)
    
    n = len(x)
    
    # Build Vandermonde matrix for degree 5
    V = np.vander(x, N=6, increasing=True)
    
    # Split for lambda selection
    indices = np.arange(n)
    np.random.shuffle(indices)
    split = int(0.8 * n)
    train_idx = indices[:split]
    val_idx = indices[split:]
    
    V_train = V[train_idx]
    y_train = y[train_idx]
    V_val = V[val_idx]
    y_val = y[val_idx]
    
    # Regularization: exclude intercept from penalty
    # Penalty matrix: diag([0, 1, 1, 1, 1, 1])
    P = np.diag([0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    
    best_lambda = 0.0
    best_mse = np.inf
    best_coeff = None
    
    # Log-spaced lambdas for smoother selection
    lambdas = np.logspace(-10, 8, 50)
    
    for lam in lambdas:
        if time.time() > deadline:
            break
            
        # Solve (V^T V + lam * P) c = V^T y
        try:
            A = V_train.T @ V_train + lam * P
            b = V_train.T @ y_train
            c = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            continue
            
        if not np.all(np.isfinite(c)):
            continue
            
        # Calculate validation MSE
        pred_val = V_val @ c
        mse = np.mean((pred_val - y_val)**2)
        
        if mse < best_mse:
            best_mse = mse
            best_lambda = lam
            best_coeff = c.copy()
    
    # Refit on full data with best lambda
    if time.time() < deadline - 5.0:
        try:
            A = V.T @ V + best_lambda * P
            b = V.T @ y
            coeff = np.linalg.solve(A, b)
        except:
            coeff = best_coeff if best_coeff is not None else np.zeros(6)
    else:
        coeff = best_coeff if best_coeff is not None else np.zeros(6)
        
    # Ensure finite
    if not np.all(np.isfinite(coeff)):
        coeff = np.zeros(6)
        
    return coeff
