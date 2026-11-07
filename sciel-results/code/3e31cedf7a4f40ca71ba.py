import numpy as np
import time

# EVOLVE-BLOCK-START

def fit(X_train, y_train):
    start_time = time.time()
    TIME_LIMIT = 300.0
    SAFETY_MARGIN = 5.0
    
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    
    # Ensure 2D
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
        
    n, d = X_train.shape
    
    # Normalize X to [-1, 1] based on training data
    # This helps with numerical stability for polynomials and exponentials
    x_min = X_train.min(axis=0)
    x_max = X_train.max(axis=0)
    x_range = x_max - x_min
    # Avoid division by zero
    x_range[x_range == 0] = 1.0
    
    def normalize(X):
        return (X - x_min) / x_range * 2.0 - 1.0
    
    X_norm = normalize(X_train)
    
    # Build rich basis functions on normalized input u
    # u is in [-1, 1]
    u = X_norm[:, 0] if d == 1 else X_norm[:, 0] # Task says 1 input
    
    cols = [np.ones(n)]
    
    # Polynomial terms up to degree 8
    # Using normalized u, high degrees are stable
    for k in range(1, 9):
        cols.append(u ** k)
        
    # Trigonometric terms
    # Since u is in [-1, 1], sin/cos are well-behaved
    cols.append(np.sin(u))
    cols.append(np.cos(u))
    cols.append(np.sin(2 * u))
    cols.append(np.cos(2 * u))
    cols.append(np.sin(0.5 * u))
    cols.append(np.cos(0.5 * u))
    
    # Exponential terms
    # e^u for u in [-1, 1] is in [0.36, 2.71], very stable
    cols.append(np.exp(u))
    cols.append(np.exp(-u))
    
    # Logarithmic terms
    # ln|u| is problematic at 0. We can use ln(1+|u|) or ln(u^2 + 1e-6)
    # Or just skip if risky. Let's add ln(u^2 + 1) which is smooth
    cols.append(np.log(u**2 + 1.0))
    
    # Rational-like
    cols.append(1.0 / (u**2 + 1.0))
    
    B = np.column_stack(cols)
    
    # Standardize columns of B to have mean 0 and std 1
    # This ensures the regularization penalty is applied fairly
    B_mean = B.mean(axis=0)
    B_std = B.std(axis=0)
    B_std[B_std == 0] = 1.0
    B_stdized = (B - B_mean) / B_std
    
    # Ridge Regression
    # Solve (B^T B + lambda I) beta = B^T y
    # Choose lambda. With 200 points and ~20 features, a small lambda is good.
    # We can try a few lambdas and pick best training MSE, or just a reasonable fixed one.
    # Let's try a small set of lambdas.
    
    best_beta = None
    best_mse = np.inf
    lambdas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    
    # Precompute B_stdized^T B_stdized and B_stdized^T y for speed if needed, but n is small
    BtB = B_stdized.T @ B_stdized
    Bty = B_stdized.T @ y_train
    
    for lam in lambdas:
        # Add regularization
        A = BtB + lam * np.eye(B_stdized.shape[1])
        try:
            beta = np.linalg.solve(A, Bty)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(B_stdized, y_train, rcond=None)[0]
            
        # Calculate training MSE
        y_pred = B_stdized @ beta
        mse = np.mean((y_pred - y_train) ** 2)
        if mse < best_mse:
            best_mse = mse
            best_beta = beta
            
    if best_beta is None:
        # Fallback to simple poly 2 if something went wrong
        def predict(X):
            X = np.asarray(X, dtype=np.float64)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            X_norm = normalize(X)
            u = X_norm[:, 0]
            # Simple fallback
            return u**2 + 1.0
            
        return predict
    
    # Store parameters for prediction
    # We need to return a callable that:
    # 1. Normalizes X
    # 2. Builds B
    # 3. Standardizes B
    # 4. Computes B @ beta
    
    def predict(X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        X_norm = normalize(X)
        u = X_norm[:, 0]
        
        # Rebuild basis
        cols = [np.ones(len(u))]
        for k in range(1, 9):
            cols.append(u ** k)
        cols.append(np.sin(u))
        cols.append(np.cos(u))
        cols.append(np.sin(2 * u))
        cols.append(np.cos(2 * u))
        cols.append(np.sin(0.5 * u))
        cols.append(np.cos(0.5 * u))
        cols.append(np.exp(u))
        cols.append(np.exp(-u))
        cols.append(np.log(u**2 + 1.0))
        cols.append(1.0 / (u**2 + 1.0))
        
        B_test = np.column_stack(cols)
        
        # Standardize using training stats
        B_test_stdized = (B_test - B_mean) / B_std
        
        y_pred = B_test_stdized @ best_beta
        
        # Ensure finite
        y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=1e10, neginf=-1e10)
        
        return y_pred
    
    return predict

# EVOLVE-BLOCK-END
