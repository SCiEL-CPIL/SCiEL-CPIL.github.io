"""
Hybrid Chebyshev-RBF Symbolic Regression.

Strategy:
1. Global Structure: Fit a Chebyshev polynomial series. This provides a stable
   global approximation that captures the main trends and oscillations of smooth
   1D functions without the Runge phenomenon or local overfitting of splines.
2. Local Refinement: Fit an RBF (Radial Basis Function) model to the residuals
   of the Chebyshev fit. This allows the model to capture high-frequency details
   or localized features that the global polynomial might miss.
3. Model Selection: Use a hold-out validation set to select the best combination
   of Chebyshev degree and RBF bandwidth.
4. Robustness: Handle numerical issues (overflow, singular matrices) gracefully.
"""

# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

# Global deadline guard: 330s wall-clock limit, use 290s for safety
_DEADLINE = time.time() + 290.0

def _chebyshev_vandermonde(x, degree):
    """
    Compute the Chebyshev Vandermonde matrix for input x.
    x: array of shape (N,)
    degree: int
    Returns: array of shape (N, degree + 1)
    """
    # Map x to [-1, 1] for standard Chebyshev polynomials
    # We need to know the range of x. 
    # For the Vandermonde construction, we assume x is already normalized or we pass min/max.
    # To keep it simple and generic, we'll normalize inside the fit function and pass normalized x here.
    
    # T_k(x) = cos(k * arccos(x))
    # Using recursion is more stable and faster:
    # T_0(x) = 1
    # T_1(x) = x
    # T_{k+1}(x) = 2*x*T_k(x) - T_{k-1}(x)
    
    N = len(x)
    V = np.zeros((N, degree + 1))
    V[:, 0] = 1.0
    if degree >= 1:
        V[:, 1] = x
        for k in range(2, degree + 1):
            V[:, k] = 2.0 * x * V[:, k-1] - V[:, k-2]
    return V

def _fit_chebyshev(x_norm, y, degree, lam=0.0):
    """
    Fit Chebyshev polynomial of given degree to y.
    x_norm: normalized x in [-1, 1]
    lam: L2 regularization parameter
    Returns: coefficients c
    """
    V = _chebyshev_vandermonde(x_norm, degree)
    # Regularized least squares: (V^T V + lam * I) c = V^T y
    # Adding a small ridge to ensure invertibility even if lam=0
    A = V.T @ V
    b = V.T @ y
    if lam > 0:
        A += lam * np.eye(degree + 1)
    else:
        A += 1e-10 * np.eye(degree + 1)
    
    try:
        c = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        # Fallback to least squares if singular
        c, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return c

def _eval_chebyshev(x_norm, c):
    """Evaluate Chebyshev polynomial with coefficients c at x_norm."""
    degree = len(c) - 1
    V = _chebyshev_vandermonde(x_norm, degree)
    return V @ c

def _fit_rbf(x, y, sigma, lam=1e-6):
    """
    Fit RBF model with Gaussian kernel.
    x: raw x values
    y: target values
    sigma: bandwidth
    Returns: weights w
    """
    N = len(x)
    diff = x[:, np.newaxis] - x[np.newaxis, :]
    D = np.exp(-(diff ** 2) / (2 * sigma ** 2))
    D += lam * np.eye(N)
    
    try:
        w = np.linalg.solve(D, y)
    except np.linalg.LinAlgError:
        w, _, _, _ = np.linalg.lstsq(D, y, rcond=None)
    return w

def _eval_rbf(x_query, x_centers, w, sigma):
    """Evaluate RBF model."""
    diff = x_query[:, np.newaxis] - x_centers[np.newaxis, :]
    D = np.exp(-(diff ** 2) / (2 * sigma ** 2))
    return D @ w

def fit(X_train, y_train):
    global _DEADLINE
    
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    
    x_train = X_train[:, 0]
    y_train = y_train
    
    # Normalize x to [-1, 1] for Chebyshev stability
    x_min = np.min(x_train)
    x_max = np.max(x_train)
    x_range = x_max - x_min
    if x_range < 1e-10:
        # Degenerate case: all x are the same
        y_mean = np.mean(y_train)
        def predict(X):
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            return np.full(X.shape[0], y_mean)
        return predict
    
    x_norm_train = 2.0 * (x_train - x_min) / x_range - 1.0
    
    # Split for Validation
    n = len(x_train)
    n_val = max(20, int(n * 0.2)) # 20% validation
    n_train_cv = n - n_val
    
    indices = np.arange(n)
    np.random.shuffle(indices)
    val_idx = indices[:n_val]
    train_cv_idx = indices[n_val:]
    
    x_cv = x_train[train_cv_idx]
    y_cv = y_train[train_cv_idx]
    x_val = x_train[val_idx]
    y_val = y_train[val_idx]
    
    x_norm_cv = 2.0 * (x_cv - x_min) / x_range - 1.0
    x_norm_val = 2.0 * (x_val - x_min) / x_range - 1.0
    
    # Candidate configurations
    # Chebyshev degrees
    cheb_degrees = [5, 10, 15, 20, 30, 40, 50, 60, 80, 100]
    # RBF sigmas (relative to x_range)
    rbf_sigmas = [0.05, 0.1, 0.2, 0.5]
    
    best_val_mse = np.inf
    best_config = None
    
    # Time budget for search
    search_start = time.time()
    search_time_limit = 60.0 # Spend max 60s on model selection
    
    # We will evaluate combinations of (Chebyshev Degree, RBF Sigma)
    # For each combo:
    # 1. Fit Cheb on CV train
    # 2. Compute residuals on CV train
    # 3. Fit RBF on residuals (using CV train x, y_res)
    # 4. Predict on Val set (Cheb_pred + RBF_pred)
    # 5. Calculate MSE
    
    for deg in cheb_degrees:
        if time.time() - search_start > search_time_limit or time.time() > _DEADLINE - 10:
            break
            
        # Fit Cheb on CV
        c_cv = _fit_chebyshev(x_norm_cv, y_cv, deg)
        
        # Predict Cheb on CV and Val
        y_cheb_cv = _eval_chebyshev(x_norm_cv, c_cv)
        y_cheb_val = _eval_chebyshev(x_norm_val, c_cv)
        
        # Residuals on CV
        res_cv = y_cv - y_cheb_cv
        
        for sigma_rel in rbf_sigmas:
            if time.time() - search_start > search_time_limit or time.time() > _DEADLINE - 10:
                break
                
            sigma = sigma_rel * x_range
            if sigma <= 0:
                continue
            
            # Fit RBF on residuals
            try:
                w_rbf = _fit_rbf(x_cv, res_cv, sigma)
                
                # Predict RBF on Val
                y_rbf_val = _eval_rbf(x_val, x_cv, w_rbf, sigma)
                
                # Total prediction
                y_pred_val = y_cheb_val + y_rbf_val
                
                mse = np.mean((y_pred_val - y_val) ** 2)
                
                if mse < best_val_mse:
                    best_val_mse = mse
                    best_config = (deg, sigma)
                    
            except Exception:
                continue

    # If no config found, fallback to simple Chebyshev
    if best_config is None:
        best_config = (10, 0.0) # No RBF
        
    # Refit best model on FULL training data
    deg, sigma = best_config
    
    # 1. Fit Cheb on full data
    c_full = _fit_chebyshev(x_norm_train, y_train, deg)
    
    # 2. Compute residuals on full data
    y_cheb_full = _eval_chebyshev(x_norm_train, c_full)
    res_full = y_train - y_cheb_full
    
    # 3. Fit RBF on full data residuals
    w_full = None
    if sigma > 0:
        try:
            w_full = _fit_rbf(x_train, res_full, sigma)
        except Exception:
            w_full = None
            
    # Construct predict function
    def predict(X):
        try:
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            x_test = X[:, 0]
            
            # Normalize test x
            x_norm_test = 2.0 * (x_test - x_min) / x_range - 1.0
            
            # Cheb prediction
            y_pred = _eval_chebyshev(x_norm_test, c_full)
            
            # RBF prediction
            if w_full is not None:
                y_rbf = _eval_rbf(x_test, x_train, w_full, sigma)
                y_pred += y_rbf
            
            # Ensure finite
            y_pred = np.where(np.isfinite(y_pred), y_pred, np.mean(y_train))
            return y_pred
        except Exception:
            return np.full(X.shape[0], np.mean(y_train))
    
    return predict

# EVOLVE-BLOCK-END
