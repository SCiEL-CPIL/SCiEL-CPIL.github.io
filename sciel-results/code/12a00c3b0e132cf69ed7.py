"""Radial Basis Function (RBF) regressor with adaptive Gaussian basis functions."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import time

def fit(X_train, y_train):
    """
    Fit an RBF model to the training data.
    
    Args:
        X_train: (n, d) training inputs
        y_train: (n,) training targets
    
    Returns:
        predict: callable that takes (m, d) array and returns (m,) predictions
    """
    start_time = time.time()
    deadline = start_time + 280  # Leave 50s safety margin before 330s limit
    
    n, d = X_train.shape
    
    # Normalize inputs to [0, 1] for better kernel behavior
    X_min = X_train.min(axis=0)
    X_max = X_train.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range < 1e-12] = 1.0
    
    def normalize(X):
        return (X - X_min) / X_range
    
    def denormalize(X_norm):
        return X_norm * X_range + X_min
    
    X_train_norm = normalize(X_train)
    
    # Estimate optimal bandwidth using median heuristic
    # Sample a subset for efficiency
    if n > 100:
        idx = np.random.choice(n, 100, replace=False)
        X_sample = X_train_norm[idx]
    else:
        X_sample = X_train_norm
    
    # Compute pairwise distances for bandwidth estimation
    dists = cdist(X_sample, X_sample)
    np.fill_diagonal(dists, np.inf)
    # Use median of nearest neighbor distances
    nn_dists = np.min(dists, axis=1)
    sigma = np.median(nn_dists)
    
    # Ensure sigma is reasonable
    if sigma < 1e-8:
        sigma = 1.0
    elif sigma > 10:
        sigma = 1.0
    
    # Number of RBF centers: use all points for small n, subset for large n
    # With 200 points, using all is fine (200x200 matrix is manageable)
    if n <= 200:
        centers = X_train_norm
        n_centers = n
    else:
        # Use k-means-like selection: pick evenly spaced points
        n_centers = min(100, n)
        step = max(1, n // n_centers)
        centers = X_train_norm[::step][:n_centers]
        n_centers = len(centers)
    
    # Build RBF kernel matrix
    # K_ij = exp(-||x_i - x_j||^2 / (2 * sigma^2))
    K = cdist(X_train_norm, centers, 'sqeuclidean')
    K = np.exp(-K / (2.0 * sigma * sigma))
    
    # Add regularization to ensure invertibility
    reg = 1e-8
    K_reg = K + reg * np.eye(n_centers)
    
    # Solve for weights using least squares
    try:
        w, residuals, rank, sv = np.linalg.lstsq(K_reg, y_train, rcond=None)
    except np.linalg.LinAlgError:
        # Fallback to polynomial if RBF fails
        def features_poly(X):
            Xn = normalize(X)
            cols = [np.ones(len(Xn))]
            for j in range(d):
                cols.append(Xn[:, j])
            for i in range(d):
                for j in range(i, d):
                    cols.append(Xn[:, i] * Xn[:, j])
            return np.column_stack(cols)
        
        w_poly, *_ = np.linalg.lstsq(features_poly(X_train), y_train, rcond=None)
        def predict_poly(X):
            Xn = normalize(X)
            return features_poly(X) @ w_poly
        return predict_poly
    
    # Check if we're still within time limit
    if time.time() > deadline:
        # Return a simple mean predictor as fallback
        y_mean = np.mean(y_train)
        return lambda X: np.full(len(X), y_mean)
    
    def predict(X):
        """Predict function for RBF model."""
        X_norm = normalize(X)
        # Compute kernel values between test points and centers
        K_test = cdist(X_norm, centers, 'sqeuclidean')
        K_test = np.exp(-K_test / (2.0 * sigma * sigma))
        return K_test @ w
    
    return predict
# EVOLVE-BLOCK-END
