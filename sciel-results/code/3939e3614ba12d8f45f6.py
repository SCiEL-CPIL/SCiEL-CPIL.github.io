"""Symbolic regression with 2 inputs using a rich basis and internal optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _build_basis(x, y):
    """Build a basis of polynomial features up to degree 4."""
    n = len(x)
    features = [np.ones(n)]
    
    # Linear
    features.append(x)
    features.append(y)
    
    # Quadratic
    features.append(x**2)
    features.append(x * y)
    features.append(y**2)
    
    # Cubic
    features.append(x**3)
    features.append(x**2 * y)
    features.append(x * y**2)
    features.append(y**3)
    
    # Quartic
    features.append(x**4)
    features.append(x**3 * y)
    features.append(x**2 * y**2)
    features.append(x * y**3)
    features.append(y**4)
    
    return np.column_stack(features)


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Fit a polynomial model to the data and return predictions.
    
    The params array is used to seed the random number generator for
    potential multiple restarts, but we primarily use a direct least-squares
    approach with ridge regularization for stability.
    """
    start_time = time.time()
    deadline = start_time + 120.0  # 120 second limit
    
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    # Build the feature matrix
    X = _build_basis(x, y)
    
    # Use ridge regression for stability
    # The params can be used to set the regularization strength
    # params[0] could control lambda, but let's use a fixed reasonable value
    # Actually, let's use the first few params as initial weights if we do optimization
    # But for simplicity and speed, let's use closed-form ridge regression
    
    # Ridge regression: w = (X^T X + lambda I)^{-1} X^T y
    # lambda = 1e-4 for mild regularization
    lambda_reg = 1e-4
    n_features = X.shape[1]
    
    # Compute in a numerically stable way
    XtX = X.T @ X
    Xty = X.T @ y
    A = XtX + lambda_reg * np.eye(n_features)
    
    try:
        w = np.linalg.solve(A, Xty)
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse if solve fails
        w = np.linalg.pinv(X) @ y
    
    # Check time
    if time.time() > deadline - 10.0:
        return X @ w
    
    # Try to improve with a few BFGS steps if time permits
    # Actually, ridge regression is already optimal for linear models, so BFGS won't help
    # unless we have a non-linear model. Let's stick with the polynomial basis.
    
    return X @ w
# EVOLVE-BLOCK-END
