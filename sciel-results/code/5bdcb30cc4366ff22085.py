import numpy as np
import time
from scipy.optimize import minimize_scalar

def _build_features(X):
    """
    Build a comprehensive set of features for symbolic regression.
    Includes polynomials, trigonometric, exponential, and interaction terms.
    """
    n, d = X.shape
    cols = [np.ones(n)]
    
    # Base features for each dimension
    bases = []
    for j in range(d):
        xj = X[:, j]
        # Polynomials
        for p in range(1, 6):
            bases.append(xj ** p)
        # Trig
        bases.append(np.sin(xj))
        bases.append(np.cos(xj))
        # Exp (damped to avoid overflow)
        bases.append(np.exp(-xj))
        # Log (shifted to avoid log(0) and negative)
        # xj is in [1, 10] usually, so log is safe
        bases.append(np.log(xj))
        # Reciprocal
        bases.append(1.0 / (xj + 1e-8))
        # Sqrt
        bases.append(np.sqrt(xj))

    # Add all individual bases
    for b in bases:
        cols.append(b)
        
    # Add pairwise interactions (products)
    # To keep dimensionality manageable, we limit interactions
    # If d is small, we can be more generous
    num_bases = len(bases)
    if d <= 2:
        # Include all pairwise products
        for i in range(num_bases):
            for j in range(i, num_bases):
                cols.append(bases[i] * bases[j])
    elif d == 3:
        # Include only products of bases from different dimensions or high-value bases
        # To avoid explosion, we'll just take products of the first few strong bases
        # Or simply limit to degree 2 polynomials of the raw inputs + interactions of trig/exp
        # Let's add products of raw inputs up to degree 3
        for i in range(d):
            for j in range(i, d):
                for k in range(j, d):
                    cols.append(X[:, i] * X[:, j] * X[:, k])
        # And some key interactions like sin(x1)*x2
        for i in range(d):
            for j in range(d):
                if i != j:
                    cols.append(np.sin(X[:, i]) * X[:, j])
                    cols.append(np.cos(X[:, i]) * X[:, j])
                    cols.append(np.exp(-X[:, i]) * X[:, j])
    else:
        # Higher dimensions: stick to lower order polynomials
        for i in range(d):
            for j in range(i, d):
                cols.append(X[:, i] * X[:, j])
                cols.append(X[:, i] ** 2)

    F = np.column_stack(cols)
    
    # Handle any NaN/Inf from numerical issues (e.g., exp overflow)
    F = np.nan_to_num(F, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return F

def _solve_lasso(X, y, lam):
    """
    Coordinate descent for Lasso regression.
    X: (n, p), y: (n,)
    lam: regularization parameter
    """
    n, p = X.shape
    w = np.zeros(p)
    
    # Pre-compute column norms squared
    col_norms_sq = np.sum(X ** 2, axis=0)
    
    # For stability, we'll run for a fixed number of iterations or until convergence
    max_iter = 100
    tol = 1e-4
    
    for _ in range(max_iter):
        w_old = w.copy()
        for j in range(p):
            if col_norms_sq[j] == 0:
                w[j] = 0
                continue
                
            # Compute residual without j
            # r = y - X @ w + X[:, j] * w[j]
            # To be efficient, we can update r incrementally, but for p~200 and n=200, 
            # direct calculation is fast enough.
            r = y - X @ w + X[:, j] * w[j]
            
            # rho_j = (1/n) * X_j^T * r
            rho_j = np.dot(X[:, j], r) / n
            # z_j = (1/n) * X_j^T * X_j
            z_j = col_norms_sq[j] / n
            
            # Soft threshold
            if rho_j > lam:
                w_new = (rho_j - lam) / z_j
            elif rho_j < -lam:
                w_new = (rho_j + lam) / z_j
            else:
                w_new = 0.0
                
            w[j] = w_new
            
        # Check convergence
        if np.max(np.abs(w - w_old)) < tol:
            break
            
    return w

def fit(X_train, y_train):
    t0 = time.time()
    deadline = t0 + 290.0  # 40s margin for 10 equations (total 330s)
    
    # Standardize inputs and output for better numerical stability in Lasso
    X_mean = np.mean(X_train, axis=0)
    X_std = np.std(X_train, axis=0)
    X_std[X_std < 1e-8] = 1.0
    
    y_mean = np.mean(y_train)
    y_std = np.std(y_train)
    y_std[y_std < 1e-8] = 1.0
    
    X_std_train = (X_train - X_mean) / X_std
    y_std_train = (y_train - y_mean) / y_std
    
    # Build features on standardized data
    F_train = _build_features(X_std_train)
    
    # Center features (remove mean) for Lasso
    F_mean = np.mean(F_train, axis=0)
    F_train_centered = F_train - F_mean
    
    n_features = F_train_centered.shape[1]
    
    # Try a range of lambda values
    # We want the "simplest" model that fits well.
    # So we start with high lambda (strong regularization, sparse) and decrease.
    
    # Estimate scale of lambda
    # Max correlation
    max_corr = np.max(np.abs(np.mean(F_train_centered * y_std_train, axis=0)))
    if max_corr < 1e-8:
        max_corr = 1.0
        
    # Lambda grid from large to small
    # Lasso lambda range: typically [0, max_corr]
    # We want to find the largest lambda that still gives low error.
    
    best_w = None
    best_mse = np.inf
    best_lambda = None
    
    # Generate lambda values
    # We'll try a geometric progression
    lambdas = np.logspace(np.log10(max_corr * 0.5), np.log10(max_corr * 1e-6), 30)
    # Sort descending (largest lambda first)
    lambdas = np.sort(lambdas)[::-1]
    
    for lam in lambdas:
        if time.time() > deadline:
            break
            
        try:
            w = _solve_lasso(F_train_centered, y_std_train, lam)
            
            # Calculate MSE on training data (standardized)
            pred = F_train_centered @ w
            mse = np.mean((pred - y_std_train) ** 2)
            
            # We want the first (largest lambda) that achieves a very low MSE
            # Since data is noiseless, we can expect MSE to be close to 0 if the model is correct.
            # However, if the true function is not in our basis, MSE won't be 0.
            # Strategy: Keep the model with the largest lambda (simplest) that has MSE < threshold.
            # If none, keep the one with lowest MSE.
            
            if mse < 1e-4: # Threshold for "perfect" fit on standardized data
                best_w = w
                best_mse = mse
                best_lambda = lam
                break # Found a simple perfect fit, stop
            else:
                if mse < best_mse:
                    best_mse = mse
                    best_w = w
                    best_lambda = lam
                    
        except Exception:
            continue
            
    if best_w is None:
        # Fallback: Ridge regression with strong regularization
        # Or just OLS if Lasso failed completely
        try:
            # Use a fixed ridge lambda
            lam_ridge = 1.0
            A = F_train_centered.T @ F_train_centered + lam_ridge * np.eye(n_features)
            b = F_train_centered.T @ y_std_train
            best_w = np.linalg.solve(A, b)
        except:
            best_w = np.zeros(n_features)
            
    # Create the predict function
    # It needs to:
    # 1. Standardize X
    # 2. Build features
    # 3. Center features
    # 4. Predict in standardized space
    # 5. Inverse standardize y
    
    def predict(X):
        # Standardize
        X_s = (X - X_mean) / X_std
        # Build features
        F = _build_features(X_s)
        # Center features (use same mean as training)
        F_c = F - F_mean
        # Predict
        y_pred_std = F_c @ best_w
        # Inverse standardize
        y_pred = y_pred_std * y_std + y_mean
        
        # Ensure finite
        y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=1e10, neginf=-1e10)
        
        return y_pred
        
    return predict
