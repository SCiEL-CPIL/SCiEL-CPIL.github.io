import numpy as np
import time
from scipy.spatial.distance import cdist

def fit(X_train, y_train):
    """
    Fit a Kernel Ridge Regression model with Matern 5/2 kernel.
    Uses LOOCV to select hyperparameters.
    """
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    
    # Flatten if 2D with 1 column
    if X.ndim == 2 and X.shape[1] == 1:
        x = X.flatten()
    elif X.ndim == 1:
        x = X
    else:
        x = X[:, 0] if X.ndim == 2 else X
        
    # Standardize
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-10:
        x_std = 1.0
    y_mean = np.mean(y)
    y_std = np.std(y)
    if y_std < 1e-10:
        y_std = 1.0
        
    x_norm = (x - x_mean) / x_std
    y_norm = (y - y_mean) / y_std
    
    N = len(x_norm)
    
    # Define Matern 5/2 kernel
    def matern52_kernel(X1, X2, l):
        # X1, X2 are 1D arrays of normalized x
        # Compute Euclidean distance
        # Since 1D, distance is |x1 - x2|
        d = np.abs(X1[:, None] - X2[None, :])
        sqrt5d = np.sqrt(5.0) * d / l
        # K = (1 + sqrt5d + 5/3 sqrt5d^2) * exp(-sqrt5d)
        # To avoid overflow/underflow issues, compute carefully
        # exp term
        exp_term = np.exp(-sqrt5d)
        poly_term = 1.0 + sqrt5d + (5.0/3.0) * (sqrt5d**2)
        return poly_term * exp_term
    
    # Hyperparameter grid
    length_scales = [0.1, 0.5, 1.0, 2.0, 5.0]
    lambdas = [1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0]
    
    best_loocv_err = np.inf
    best_l = 1.0
    best_lambda = 1e-2
    best_w = None
    best_K = None
    
    # Pre-compute kernel matrices for each length scale? 
    # K depends on l. We need K for each l.
    # We can compute K for each l in the loop.
    
    for l in length_scales:
        K = matern52_kernel(x_norm, x_norm, l)
        # Add small diagonal for numerical stability
        K += 1e-12 * np.eye(N)
        
        for lam in lambdas:
            A = K + lam * np.eye(N)
            try:
                # Solve A w = y_norm
                w = np.linalg.solve(A, y_norm)
                
                # Compute LOOCV error
                # H = A^{-1} K
                # diag_H = diag(A^{-1} K)
                # loocv_res_i = (y_i - y_hat_i) / (1 - H_ii)
                # y_hat = K w
                y_hat = K @ w
                # We need diag of A^{-1} K
                # Instead of computing full inverse, we can use the formula:
                # h_ii = (A^{-1} K)_{ii}
                # Note that A = K + lam I
                # A^{-1} K = A^{-1} (A - lam I) = I - lam A^{-1}
                # So h_ii = 1 - lam * (A^{-1})_{ii}
                # We need (A^{-1})_{ii}.
                # We can compute A_inv = np.linalg.inv(A)
                # But that's O(N^3) again.
                # We already solved A w = y.
                # To get diagonal of A^{-1}, we can solve A X = I.
                # That's N solves, O(N^4). Too slow for grid search.
                
                # Alternative LOOCV formula for KRR:
                # e_i = (y_i - y_hat_i) / (1 - h_ii)
                # where h_ii is the i-th diagonal element of the hat matrix H = K (K + lam I)^{-1}
                # As derived, h_ii = 1 - lam * (A^{-1})_{ii}
                # We can estimate (A^{-1})_{ii} by solving A v = e_i for each i? No, too slow.
                
                # Let's just use the training error as a proxy if LOOCV is too complex?
                # No, LOOCV is important for noise.
                
                # Efficient LOOCV for KRR:
                # We can compute the diagonal of the inverse using the Cholesky factorization.
                L = np.linalg.cholesky(A)
                # A^{-1} = L^{-T} L^{-1}
                # diag(A^{-1}) = sum_j (L^{-1})_{ij}^2
                # We can solve L Z = I to get L^{-1}
                Z = np.linalg.solve(L, np.eye(N))
                L_inv = Z.T # This is L^{-T}? No.
                # L L^T = A.
                # We want A^{-1}.
                # Let's just compute the full inverse for N=200. It's fast enough.
                A_inv = np.linalg.inv(A)
                diag_A_inv = np.diag(A_inv)
                
                h_diag = 1.0 - lam * diag_A_inv
                # Avoid division by zero
                h_diag = np.where(np.abs(h_diag) < 1e-10, 1e-10, h_diag)
                
                loocv_res = (y_norm - y_hat) / h_diag
                loocv_err = np.mean(loocv_res**2)
                
                if loocv_err < best_loocv_err:
                    best_loocv_err = loocv_err
                    best_l = l
                    best_lambda = lam
                    best_w = w
                    best_K = K
                    # If error is very small, we might prefer larger lambda for stability
                    if loocv_err < 1e-10:
                        # Store this candidate, but we might want to break early or prefer larger lambda
                        # We'll just keep updating if strictly smaller, or if equal and lambda is larger?
                        # Let's just keep the first best found, or the one with largest lambda if tied.
                        pass
                        
            except np.linalg.LinAlgError:
                continue

    # If no valid fit found, fallback to mean
    if best_w is None:
        def predict(X_test):
            X_test = np.asarray(X_test, dtype=np.float64)
            if X_test.ndim == 2 and X_test.shape[1] == 1:
                x_test = X_test.flatten()
            elif X_test.ndim == 1:
                x_test = X_test
            else:
                x_test = X_test[:, 0]
            return np.full(x_test.shape[0], y_mean)
        return predict

    # Refit with best parameters (already done in loop, but ensure we have the correct K and w)
    # We stored best_w and best_K corresponding to best_l and best_lambda
    
    def predict(X_test):
        X_test = np.asarray(X_test, dtype=np.float64)
        if X_test.ndim == 2 and X_test.shape[1] == 1:
            x_test = X_test.flatten()
        elif X_test.ndim == 1:
            x_test = X_test
        else:
            x_test = X_test[:, 0]
            
        # Normalize test x
        x_test_norm = (x_test - x_mean) / x_std
        
        # Compute kernel between test and train
        K_test = matern52_kernel(x_test_norm, x_norm, best_l)
        
        # Predict
        y_pred_norm = K_test @ best_w
        
        # Unstandardize
        y_pred = y_pred_norm * y_std + y_mean
        
        # Ensure finite
        y_pred = np.nan_to_num(y_pred, nan=y_mean, posinf=1e10, neginf=-1e10)
        
        return y_pred

    return predict
