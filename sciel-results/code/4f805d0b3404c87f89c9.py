"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def fit(X_train, y_train):
    """
    Fit a regression model using Chebyshev polynomial basis with L2 regularization.
    The optimal degree is selected via cross-validation-like criterion on the training data.
    """
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    
    # Handle potential edge cases
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    if y_train.ndim == 1:
        y_train = y_train.reshape(-1)
    
    n, d = X_train.shape
    x_min = X_train.min(axis=0)
    x_max = X_train.max(axis=0)
    x_range = x_max - x_min
    x_range[x_range == 0] = 1.0
    
    # Normalize to [-1, 1]
    X_norm = (X_train - x_min) / x_range * 2 - 1
    
    def chebyshev_basis(x, degree):
        """Compute Chebyshev polynomials of the first kind up to given degree."""
        if degree == 0:
            return np.ones((len(x), 1))
        
        T = np.ones((len(x), degree + 1))
        T[:, 1] = x
        for k in range(2, degree + 1):
            T[:, k] = 2 * x * T[:, k-1] - T[:, k-2]
        return T
    
    # Try different degrees and regularization strengths
    # For 1D data, we can use higher degrees
    max_degree = 50
    best_score = np.inf
    best_degree = 2
    best_lambda = 1e-10
    best_coeffs = None
    
    # L2 regularization parameter
    # We'll try a small set of lambda values
    lambdas = [1e-15, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 0.1]
    
    # For efficiency, compute basis up to max_degree once
    if d == 1:
        T_all = chebyshev_basis(X_norm[:, 0], max_degree)
        
        for degree in range(2, max_degree + 1):
            T = T_all[:, :degree + 1]
            for lam in lambdas:
                # Solve (T^T T + lambda * I) w = T^T y
                # Use ridge regression
                try:
                    A = T.T @ T + lam * np.eye(degree + 1)
                    b = T.T @ y_train
                    coeffs = np.linalg.solve(A, b)
                    
                    # Compute training MSE
                    pred = T @ coeffs
                    mse = np.mean((pred - y_train) ** 2)
                    
                    # Effective degrees of freedom for penalty
                    # For ridge regression, trace of leverage matrix
                    # But for simplicity, use a heuristic penalty
                    # Use BIC-like criterion: n*log(MSE) + k*log(n)
                    k = degree + 1
                    bic = n * np.log(mse + 1e-30) + k * np.log(n)
                    
                    if bic < best_score:
                        best_score = bic
                        best_degree = degree
                        best_lambda = lam
                        best_coeffs = coeffs.copy()
                except np.linalg.LinAlgError:
                    continue
    else:
        # For multi-dimensional, use tensor product or just univariate on each dim
        # Since the task says 1 input for all equations, this branch may not be hit
        # but handle it gracefully
        for degree in range(2, min(max_degree, 20) + 1):
            # Build feature matrix: all monomials up to total degree
            # This gets expensive, so limit degree
            if degree > 6 and d > 1:
                continue
                
            cols = [np.ones(n)]
            for total_deg in range(1, degree + 1):
                for i in range(d):
                    for j in range(i, d):
                        if i + j + 1 <= degree and (i == j or True):
                            # This is getting complex; for d=1 it's just powers
                            pass
            
            # Simpler: just use univariate Chebyshev on each dimension and combine
            # For d=1 this is already handled above
            # For d>1, use a simpler approach
            T_list = []
            for dim in range(d):
                T_dim = chebyshev_basis(X_norm[:, dim], min(degree, 15))
                T_list.append(T_dim)
            
            # Combine: use all basis functions from each dimension up to degree
            # This is a union, not tensor product, to keep size manageable
            T = np.hstack(T_list)
            # Remove duplicates by using unique columns (approximate)
            # Actually, let's just use the first dimension's basis for simplicity
            # since the task guarantees 1D inputs
            T = T_list[0]
            
            for lam in lambdas:
                try:
                    A = T.T @ T + lam * np.eye(T.shape[1])
                    b = T.T @ y_train
                    coeffs = np.linalg.solve(A, b)
                    
                    pred = T @ coeffs
                    mse = np.mean((pred - y_train) ** 2)
                    k = T.shape[1]
                    bic = n * np.log(mse + 1e-30) + k * np.log(n)
                    
                    if bic < best_score:
                        best_score = bic
                        best_degree = degree
                        best_lambda = lam
                        best_coeffs = coeffs.copy()
                except np.linalg.LinAlgError:
                    continue
    
    # Build the prediction function
    def predict(X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        X_pred_norm = (X - x_min) / x_range * 2 - 1
        
        if d == 1:
            T_pred = chebyshev_basis(X_pred_norm[:, 0], best_degree)
            return T_pred @ best_coeffs
        else:
            # Fallback for multi-dim
            T_pred = chebyshev_basis(X_pred_norm[:, 0], best_degree)
            return T_pred @ best_coeffs
    
    return predict
# EVOLVE-BLOCK-END
