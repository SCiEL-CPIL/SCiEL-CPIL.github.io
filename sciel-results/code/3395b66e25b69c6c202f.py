"""Symbolic regression via GP Bagging with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import operator
from scipy.stats import norm

def fit(X_train, y_train):
    start_time = time.time()
    # 300s limit. We reserve 50s for safety/overhead, leaving 250s for computation.
    deadline = start_time + 250.0
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64).flatten()
    n_samples, n_features = X.shape
    
    # Normalize inputs to [0, 1] for GP stability
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range == 0] = 1.0
    X_norm = (X - X_min) / X_range
    
    # Standardize y
    y_mean = y.mean()
    y_std = y.std()
    if y_std == 0:
        y_std = 1.0
    y_norm = (y - y_mean) / y_std

    # We will fit multiple GPs on different random subsets (bagging)
    # and average their predictions.
    # Given time constraints, we limit the number of GPs and iterations.
    max_gps = 10
    gp_models = []
    
    # Try to fit as many GPs as possible before deadline
    # Each GP fit involves optimizing hyperparameters.
    # For speed, we use a simplified RBF kernel with fixed or quickly optimized length scales.
    
    # A simpler, faster approach for "different strategy":
    # Use a Random Forest or Gradient Boosting? 
    # But GP is better for smooth functions if hyperparams are right.
    # However, GP is O(N^3). N=200 is small. 
    # Let's use a "Kernel Ridge Regression" with multiple kernels (RBF, Polynomial) 
    # and ensemble them. This is faster than full GP MCMC/Sampling and often better for regression.
    
    # Strategy: Kernel Ridge Regression (KRR) with multiple kernels, averaged.
    # This avoids the heavy optimization of GP hyperparameters while retaining non-linearity.
    
    kernels = []
    
    # 1. RBF Kernel with varying lengths
    for length_scale in [0.1, 0.2, 0.5, 1.0, 2.0]:
        def rbf_kernel(X1, X2, ls=length_scale):
            X1_norm = (X1 - X_min) / X_range
            X2_norm = (X2 - X_min) / X_range
            sq_dists = np.sum(X1_norm**2, axis=1)[:, None] + np.sum(X2_norm**2, axis=1)[None, :] - 2 * X1_norm @ X2_norm.T
            sq_dists = np.maximum(sq_dists, 0)
            return np.exp(-0.5 * sq_dists / (ls * ls))
        kernels.append(rbf_kernel)
        
    # 2. Polynomial Kernel (degree 3 and 4)
    for degree in [3, 4]:
        def poly_kernel(X1, X2, d=degree):
            X1_norm = (X1 - X_min) / X_range
            X2_norm = (X2 - X_min) / X_range
            return (1 + X1_norm @ X2_norm.T) ** d
        kernels.append(poly_kernel)
        
    # 3. Linear Kernel
    def lin_kernel(X1, X2):
        X1_norm = (X1 - X_min) / X_range
        X2_norm = (X2 - X_min) / X_range
        return X1_norm @ X2_norm.T
    kernels.append(lin_kernel)

    # Fit KRR for each kernel
    # KRR: alpha = (K + lambda*I)^-1 y
    # We use a small regularization lambda to ensure invertibility
    lambda_reg = 1e-6
    
    alphas = []
    kernel_funcs = []
    
    for k_func in kernels:
        if time.time() > deadline - 10: # Safety margin
            break
            
        K_train = k_func(X_norm, X_norm)
        # Add regularization
        K_train += lambda_reg * np.eye(n_samples)
        
        try:
            # Solve for alpha
            alpha = np.linalg.solve(K_train, y_norm)
            alphas.append(alpha)
            kernel_funcs.append(k_func)
        except np.linalg.LinAlgError:
            # Fallback to lstsq if singular
            try:
                alpha, *_ = np.linalg.lstsq(K_train, y_norm, rcond=None)
                alphas.append(alpha)
                kernel_funcs.append(k_func)
            except:
                pass

    if not alphas:
        # Fallback to simple mean if all fail
        mean_y = y.mean()
        return lambda X: np.full(X.shape[0], mean_y)

    def predict(X_test):
        if time.time() > deadline:
             # This shouldn't happen during predict, but safety check
             pass
             
        X_test = np.asarray(X_test, dtype=np.float64)
        n_test = X_test.shape[0]
        
        # Normalize test data using train stats
        X_test_norm = (X_test - X_min) / X_range
        
        preds = []
        for k_func, alpha in zip(kernel_funcs, alphas):
            K_test = k_func(X_test_norm, X_norm)
            pred_norm = K_test @ alpha
            preds.append(pred_norm)
            
        # Average predictions
        if preds:
            avg_pred_norm = np.mean(preds, axis=0)
        else:
            avg_pred_norm = np.zeros(n_test)
            
        # Denormalize
        avg_pred = avg_pred_norm * y_std + y_mean
        
        return avg_pred

    return predict
# EVOLVE-BLOCK-END
