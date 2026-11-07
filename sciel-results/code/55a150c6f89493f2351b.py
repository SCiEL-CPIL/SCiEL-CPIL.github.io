"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.interpolate import PchipInterpolator


def fit(X_train, y_train):
    """
    Fit a regression model using high-resolution monotone cubic interpolation (PCHIP)
    on the sorted training data. This respects local monotonicity and shape without
    oscillatory artifacts.
    
    For multi-dimensional inputs, we fall back to a robust polynomial least-squares
    fit with higher degree and regularization, but given the task specifies 1 input
    for all equations, we primarily use PCHIP.
    """
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    
    # Ensure 2D
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    
    d = X_train.shape[1]
    
    if d == 1:
        # Sort by input value
        sort_idx = np.argsort(X_train[:, 0])
        x_sorted = X_train[sort_idx, 0]
        y_sorted = y_train[sort_idx]
        
        # Check for duplicate x values; if present, average them
        unique_x, inverse_indices = np.unique(x_sorted, return_inverse=True)
        if len(unique_x) < len(x_sorted):
            # Average y values for duplicate x
            y_avg = np.zeros(len(unique_x))
            counts = np.zeros(len(unique_x))
            np.add.at(y_avg, inverse_indices, y_sorted)
            np.add.at(counts, inverse_indices, 1)
            y_avg /= counts
            x_sorted = unique_x
            y_sorted = y_avg
        
        # Fit PCHIP interpolator
        try:
            pchip_interp = PchipInterpolator(x_sorted, y_sorted)
            
            def predict(X):
                X = np.asarray(X, dtype=np.float64)
                if X.ndim == 1:
                    X = X.reshape(-1, 1)
                x_pred = X[:, 0]
                return pchip_interp(x_pred)
            
            return predict
        except Exception:
            # Fallback to linear interpolation if PCHIP fails
            try:
                from scipy.interpolate import interp1d
                interp_func = interp1d(x_sorted, y_sorted, kind='linear', fill_value='extrapolate')
                
                def predict(X):
                    X = np.asarray(X, dtype=np.float64)
                    if X.ndim == 1:
                        X = X.reshape(-1, 1)
                    x_pred = X[:, 0]
                    return interp_func(x_pred)
                
                return predict
            except Exception:
                # Final fallback: nearest neighbor
                def predict(X):
                    X = np.asarray(X, dtype=np.float64)
                    if X.ndim == 1:
                        X = X.reshape(-1, 1)
                    x_pred = X[:, 0]
                    # Find nearest training point
                    idx = np.argmin(np.abs(x_sorted[:, None] - x_pred[None, :]), axis=0)
                    return y_sorted[idx]
                
                return predict
    else:
        # Multi-dimensional fallback: use polynomial features with ridge regression
        # Build polynomial features up to degree 3
        def make_features(X, degree=3):
            cols = [np.ones(len(X))]
            for d in range(1, degree + 1):
                for i in range(X.shape[1]):
                    cols.append(X[:, i] ** d)
            # Add interaction terms for degree 2 and 3
            if degree >= 2:
                for i in range(X.shape[1]):
                    for j in range(i + 1, X.shape[1]):
                        cols.append(X[:, i] * X[:, j])
            if degree >= 3:
                for i in range(X.shape[1]):
                    for j in range(i + 1, X.shape[1]):
                        for k in range(j + 1, X.shape[1]):
                            cols.append(X[:, i] * X[:, j] * X[:, k])
            return np.column_stack(cols)
        
        try:
            feat_train = make_features(X_train, degree=3)
            # Ridge regression for stability
            alpha = 1e-6
            w = np.linalg.solve(feat_train.T @ feat_train + alpha * np.eye(feat_train.shape[1]), feat_train.T @ y_train)
            
            def predict(X):
                X = np.asarray(X, dtype=np.float64)
                if X.ndim == 1:
                    X = X.reshape(-1, 1)
                feat_pred = make_features(X, degree=3)
                return feat_pred @ w
            
            return predict
        except Exception:
            # Final fallback: simple linear
            def features_lin(X):
                return np.column_stack([np.ones(len(X)), X])
            
            w, *_ = np.linalg.lstsq(features_lin(X_train), y_train, rcond=None)
            
            def predict(X):
                X = np.asarray(X, dtype=np.float64)
                if X.ndim == 1:
                    X = X.reshape(-1, 1)
                return features_lin(X) @ w
            
            return predict
# EVOLVE-BLOCK-END
