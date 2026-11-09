"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def fit(X_train, y_train):
    """
    Fit a regression model for 1D noiseless data using direct interpolation.
    Since the data is noiseless and 1D, we sort by x and use cubic spline
    interpolation to exactly reconstruct the underlying function.
    """
    n, d = X_train.shape
    start_time = time.time()
    deadline = start_time + 280.0  # safety margin

    if d != 1:
        # Fallback for multi-dimensional: use polynomial regression
        def make_poly(degree):
            def build_features(X):
                cols = [np.ones(len(X))]
                for j in range(d):
                    for p in range(1, degree + 1):
                        cols.append(X[:, j] ** p)
                if d > 1:
                    for i in range(d):
                        for j in range(i + 1, d):
                            cols.append(X[:, i] * X[:, j])
                return np.column_stack(cols)
            def model_fn(X_tr, y_tr):
                w, *_ = np.linalg.lstsq(build_features(X_tr), y_tr, rcond=None)
                return lambda X: build_features(X) @ w
            return model_fn
        # Try increasing degrees until time runs out
        best_pred = None
        for deg in [2, 3, 4, 5, 6]:
            if time.time() > deadline - 30:
                break
            try:
                model = make_poly(deg)
                pred_fn = model(X_train, y_train)
                if best_pred is None:
                    best_pred = pred_fn
            except Exception:
                continue
        if best_pred is None:
            model = make_poly(2)
            best_pred = model(X_train, y_train)
        return best_pred

    # 1D case: use direct interpolation
    x_tr = X_train[:, 0]
    y_tr = y_train
    
    # Sort by x
    sort_idx = np.argsort(x_tr)
    x_sorted = x_tr[sort_idx]
    y_sorted = y_tr[sort_idx]
    
    # Handle potential duplicate x values by averaging
    unique_x, inverse = np.unique(x_sorted, return_inverse=True)
    if len(unique_x) < len(x_sorted):
        y_unique = np.zeros(len(unique_x))
        counts = np.zeros(len(unique_x))
        np.add.at(y_unique, inverse, y_sorted)
        np.add.at(counts, inverse, 1)
        y_unique /= counts
    else:
        unique_x = x_sorted
        y_unique = y_sorted
    
    # Use cubic spline interpolation
    from scipy.interpolate import CubicSpline
    
    try:
        cs = CubicSpline(unique_x, y_unique)
        
        def predict(X):
            x = X[:, 0]
            return cs(x)
        
        # Verify it produces finite outputs on training data
        test_pred = predict(X_train[:10])
        if not np.all(np.isfinite(test_pred)):
            raise ValueError("Non-finite predictions")
        
        return predict
    except Exception:
        # Fallback: use PCHIP (monotone cubic interpolation)
        from scipy.interpolate import PchipInterpolator
        
        try:
            pchip = PchipInterpolator(unique_x, y_unique)
            
            def predict(X):
                x = X[:, 0]
                return pchip(x)
            
            test_pred = predict(X_train[:10])
            if not np.all(np.isfinite(test_pred)):
                raise ValueError("Non-finite predictions")
            
            return predict
        except Exception:
            # Last resort: linear interpolation with extrapolation
            from scipy.interpolate import interp1d
            
            try:
                interp = interp1d(unique_x, y_unique, kind='linear',
                                   bounds_error=False, fill_value='extrapolate')
                
                def predict(X):
                    x = X[:, 0]
                    return interp(x)
                
                return predict
            except Exception:
                # Absolute last resort: constant prediction (mean)
                mean_y = np.mean(y_tr)
                def predict(X):
                    return np.full(len(X), mean_y)
                return predict
# EVOLVE-BLOCK-END
