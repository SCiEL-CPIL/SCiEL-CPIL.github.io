"""
Strategy: Gaussian Process Regression with Matérn-5/2 kernel.
This approach abandons the fixed feature basis (polynomials, trig, exp) for a flexible non-parametric model.
GP regression is ideal for noiseless data as it can interpolate perfectly if the kernel is chosen well.
We use a simple kernel with fixed length scales (1.0) and unit variance to avoid complex hyperparameter optimization,
relying on the flexibility of the Gaussian kernel to capture smooth variations.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import cdist

def fit(X_train, y_train):
    start_time = time.time()
    deadline = 325.0  # Leave 5s safety margin for 330s limit
    
    d = X_train.shape[1]
    n = X_train.shape[0]
    
    # Normalize inputs to [0, 1] range for better kernel conditioning
    # This helps the GP generalize better if the scale of inputs varies
    min_vals = np.min(X_train, axis=0)
    max_vals = np.max(X_train, axis=0)
    ranges = max_vals - min_vals
    # Avoid division by zero
    ranges[ranges == 0] = 1.0
    
    X_norm = (X_train - min_vals) / ranges
    y_mean = np.mean(y_train)
    y_centered = y_train - y_mean
    y_var = np.var(y_train)
    
    # Kernel: Matérn 5/2 is often better for smooth but non-constant curvature functions
    # K(x, x') = (1 + sqrt(5)*r + 5/3*r^2) * exp(-sqrt(5)*r) where r = ||x-x'|| / l
    # We use l=1.0 (in normalized space) and sigma_f=1.0
    # Since data is noiseless, we can use a very small noise term for numerical stability
    noise = 1e-8
    
    def matern52_kernel(X1, X2, length_scale=1.0):
        # X1: (n, d), X2: (m, d)
        dists = cdist(X1, X2, 'euclidean') / length_scale
        sqrt5 = np.sqrt(5.0)
        K = (1.0 + sqrt5 * dists + (5.0 / 3.0) * dists**2) * np.exp(-sqrt5 * dists)
        return K
    
    try:
        # Compute training kernel matrix
        K = matern52_kernel(X_norm, X_norm)
        # Add noise for stability
        K += noise * np.eye(n)
        
        # Solve for coefficients: alpha = (K + sigma^2 I)^-1 y
        # Use Cholesky for stability
        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_centered))
        except np.linalg.LinAlgError:
            # Fallback to lstsq if Cholesky fails (e.g., due to near-singularity)
            # This is slow but robust
            alpha, *_ = np.linalg.lstsq(K, y_centered, rcond=None)
            
        def predict(X):
            # Normalize new inputs
            X_new_norm = (X - min_vals) / ranges
            # Compute kernel between new points and training points
            K_s = matern52_kernel(X_new_norm, X_norm)
            # Prediction mean
            y_pred = y_mean + K_s @ alpha
            return y_pred

        # Verify prediction on training set to ensure it's finite and reasonable
        pred_train = predict(X_train)
        if not np.all(np.isfinite(pred_train)):
            raise ValueError("Non-finite predictions")
            
        # Check if the fit is actually good (normalized MSE < 1.0)
        # If not, we might fall back to a simpler model, but GP should be good.
        mse = np.mean((pred_train - y_train)**2)
        nrmse = mse / max(y_var, 1e-12)
        
        # If the GP fails (e.g., numerical issues leading to bad fit), fall back to linear regression
        if nrmse > 1.0:
             # Fallback: Linear Regression
             from numpy.linalg import lstsq
             X_with_ones = np.column_stack([np.ones(n), X_train])
             w, *_ = lstsq(X_with_ones, y_train, rcond=None)
             def predict_fallback(X):
                 X_new_with_ones = np.column_stack([np.ones(X.shape[0]), X])
                 return X_new_with_ones @ w
             return predict_fallback
             
        return predict

    except Exception as e:
        # Ultimate fallback: Return mean
        # Or a simple linear regression if GP fails entirely
        try:
            X_with_ones = np.column_stack([np.ones(n), X_train])
            w, *_ = np.linalg.lstsq(X_with_ones, y_train, rcond=None)
            def predict_linear(X):
                X_new_with_ones = np.column_stack([np.ones(X.shape[0]), X])
                return X_new_with_ones @ w
            return predict_linear
        except Exception:
            mean_y = np.mean(y_train)
            return lambda X: np.full(X.shape[0], mean_y)

# EVOLVE-BLOCK-END
