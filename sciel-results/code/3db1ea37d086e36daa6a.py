import numpy as np
import time

def fit(X_train, y_train):
    """
    Fits a Fourier Series regression model to the training data.
    Returns a predict callable.
    """
    start_time = time.time()
    DEADLINE = 290.0  # Safety margin before 300s limit
    
    # Ensure X is 1D
    x_train = X_train.flatten()
    y_train = y_train.flatten()
    
    # Determine domain
    x_min = np.min(x_train)
    x_max = np.max(x_train)
    
    # Add a small buffer to the domain to avoid edge issues if test points are slightly outside
    # But the problem says uniform input ranges are given, so test points should be within range.
    # We will map [x_min, x_max] to [0, 2*pi]
    x_range = x_max - x_min
    if x_range < 1e-8:
        x_range = 1.0
    
    # Number of Fourier terms
    K = 50
    
    def build_fourier_basis(x):
        # x is 1D array
        # Map x to [0, 2*pi]
        x_mapped = (x - x_min) / x_range * 2 * np.pi
        
        # Basis vectors
        # Constant term
        basis = [np.ones_like(x_mapped)]
        
        for k in range(1, K + 1):
            basis.append(np.cos(k * x_mapped))
            basis.append(np.sin(k * x_mapped))
            
        return np.column_stack(basis)
    
    # Build training basis
    Phi_train = build_fourier_basis(x_train)
    
    # Ridge Regression
    # (Phi^T Phi + lambda I) theta = Phi^T y
    lambda_reg = 1e-4
    I = np.eye(Phi_train.shape[1])
    
    try:
        A = Phi_train.T @ Phi_train + lambda_reg * I
        b = Phi_train.T @ y_train
        theta = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        # Fallback to least squares if singular (unlikely with ridge)
        try:
            theta = np.linalg.lstsq(Phi_train, y_train, rcond=None)[0]
        except Exception:
            theta = np.zeros(Phi_train.shape[1])
            theta[0] = np.mean(y_train)
            
    def predict(X):
        x_test = X.flatten()
        Phi_test = build_fourier_basis(x_test)
        y_pred = Phi_test @ theta
        return y_pred
        
    return predict
