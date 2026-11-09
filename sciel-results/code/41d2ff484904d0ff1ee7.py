import numpy as np
import time
from scipy.optimize import minimize

def fit(X_train, y_train):
    t0 = time.time()
    DEADLINE = 325.0  # Safety margin
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape
    
    # Standardize
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-12] = 1.0
    
    def make_features(Xraw):
        Xr = np.asarray(Xraw, dtype=np.float64)
        Xrn = (Xr - mu) / sd
        cols = [np.ones(len(Xr))]
        
        # Linear
        for j in range(d):
            cols.append(Xrn[:, j])
            
        # Polynomial terms up to degree 4
        # Squares
        for j in range(d):
            cols.append(Xrn[:, j] ** 2)
            
        # Pairwise products (i < j)
        for i in range(d):
            for j in range(i + 1, d):
                cols.append(Xrn[:, i] * Xrn[:, j])
                
        # Cubes
        for j in range(d):
            cols.append(Xrn[:, j] ** 3)
            
        # Triple products (i < j < k)
        for i in range(d):
            for j in range(i + 1, d):
                for k in range(j + 1, d):
                    cols.append(Xrn[:, i] * Xrn[:, j] * Xrn[:, k])
                    
        # Quartics (just powers of 4 for simplicity, and some mixed)
        for j in range(d):
            cols.append(Xrn[:, j] ** 4)
            
        # Trig functions
        for j in range(d):
            cols.append(np.sin(Xrn[:, j]))
            cols.append(np.cos(Xrn[:, j]))
            
        # Exp functions (careful with overflow, but standardized X is usually safe)
        # Clip to avoid extreme values if standardization fails or range is large
        Xrn_clipped = np.clip(Xrn, -10, 10)
        for j in range(d):
            cols.append(np.exp(Xrn_clipped[:, j]))
            cols.append(np.exp(-Xrn_clipped[:, j]))
            
        return np.column_stack(cols)
    
    A = make_features(X_train)
    
    # Grid search for lambda
    lambdas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    best_w = None
    best_val_err = np.inf
    
    # Simple K-Fold CV (K=5)
    k = 5
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    folds = np.array_split(indices, k)
    
    for lam in lambdas:
        val_errors = []
        for i in range(k):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
            
            A_tr = A[train_idx]
            y_tr = y[train_idx]
            A_val = A[val_idx]
            y_val = y[val_idx]
            
            # Solve (A^T A + lam * I) w = A^T y
            # Use lstsq for robustness
            # Or solve explicitly
            try:
                # Add regularization to diagonal
                # A is n x p
                # A^T A is p x p
                # p might be > n, so A^T A is singular without reg.
                # We solve the regularized system.
                # Efficient way: np.linalg.lstsq with added noise? 
                # Better: solve (A.T @ A + lam * np.eye(A.shape[1])) @ w = A.T @ y
                p = A.shape[1]
                ATA = A_tr.T @ A_tr
                ATy = A_tr.T @ y_tr
                reg = lam * np.eye(p)
                w = np.linalg.solve(ATA + reg, ATy)
                
                y_pred_val = A_val @ w
                mse = np.mean((y_pred_val - y_val) ** 2)
                val_errors.append(mse)
            except np.linalg.LinAlgError:
                # Fallback to lstsq if solve fails (shouldn't happen with reg)
                w, *_ = np.linalg.lstsq(A_tr, y_tr, rcond=None)
                y_pred_val = A_val @ w
                mse = np.mean((y_pred_val - y_val) ** 2)
                val_errors.append(mse)
                
        mean_val_err = np.mean(val_errors)
        if mean_val_err < best_val_err:
            best_val_err = mean_val_err
            # Refit on all data with best lambda
            p = A.shape[1]
            ATA = A.T @ A
            ATy = A.T @ y
            reg = lam * np.eye(p)
            try:
                best_w = np.linalg.solve(ATA + reg, ATy)
            except np.linalg.LinAlgError:
                best_w, *_ = np.linalg.lstsq(A, y, rcond=None)
                
            # Check time
            if time.time() > DEADLINE:
                break
                
    # If no lambda was selected (e.g. due to time), use a default
    if best_w is None:
        p = A.shape[1]
        ATA = A.T @ A
        ATy = A.T @ y
        reg = 1e-3 * np.eye(p)
        best_w = np.linalg.solve(ATA + reg, ATy)
        
    # Check if the linear model is already good enough
    y_pred_train = A @ best_w
    train_mse = np.mean((y_pred_train - y) ** 2)
    y_var = np.var(y)
    normalized_mse = train_mse / y_var if y_var > 1e-12 else train_mse
    
    # If the linear model fits well, return it
    if normalized_mse < 0.01 and time.time() < DEADLINE - 100:
        def predict(X):
            X = np.asarray(X, dtype=np.float64)
            A_pred = make_features(X)
            return A_pred @ best_w
        return predict
    
    # Try non-linear optimization on the coefficients
    # We'll use a subset of the most important features or all if d is small
    # Let's try optimizing the weights directly with L2 regularization
    
    def objective(w):
        y_pred = A @ w
        mse = np.mean((y_pred - y) ** 2)
        l2_reg = 1e-4 * np.sum(w ** 2)
        return mse + l2_reg
    
    # Start from the regularized solution
    w0 = best_w.copy()
    
    # Try Nelder-Mead optimization
    try:
        result = minimize(objective, w0, method='Nelder-Mead', 
                          options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
        
        if result.success and result.fun < objective(w0):
            best_w = result.x
            y_pred_train = A @ best_w
            train_mse = np.mean((y_pred_train - y) ** 2)
            normalized_mse = train_mse / y_var if y_var > 1e-12 else train_mse
            # Only accept if it significantly improved
            if normalized_mse < 0.01:
                pass # Good
    except Exception:
        pass
    
    def predict(X):
        X = np.asarray(X, dtype=np.float64)
        A_pred = make_features(X)
        return A_pred @ best_w
        
    return predict
