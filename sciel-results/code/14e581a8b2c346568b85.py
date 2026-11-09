# EVOLVE-BLOCK-START
import numpy as np
from scipy.interpolate import UnivariateSpline
import time

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 290  # 300s limit, keep 10s margin for safety

    X = X_train
    y = y_train
    
    if X.ndim == 1:
        X = X.reshape(-1, 1)
        
    x_all = X[:, 0]
    y_all = y
    n = len(x_all)
    
    # Sort by X for spline fitting
    sort_idx = np.argsort(x_all)
    x_sorted = x_all[sort_idx]
    y_sorted = y_all[sort_idx]
    
    # Handle duplicates
    unique_x, unique_indices = np.unique(x_sorted, return_index=True)
    if len(unique_x) < len(x_sorted):
        X_clean = []
        y_clean = []
        current_x = None
        sum_y = 0
        count = 0
        for xi, yi in zip(x_sorted, y_sorted):
            if xi != current_x:
                if current_x is not None:
                    X_clean.append(current_x)
                    y_clean.append(sum_y / count)
                current_x = xi
                sum_y = yi
                count = 1
            else:
                sum_y += yi
                count += 1
        if current_x is not None:
            X_clean.append(current_x)
            y_clean.append(sum_y / count)
        
        x_fit = np.array(X_clean)
        y_fit = np.array(y_clean)
    else:
        x_fit = x_sorted
        y_fit = y_sorted
        
    n_fit = len(x_fit)
    
    if n_fit < 5:
        # Fallback to mean
        mean_y = np.mean(y_fit)
        return lambda X_test: np.full(len(X_test), mean_y)

    # Define candidate spline configurations
    # (degree, num_knots_fraction)
    # num_knots = int(n_fit * fraction)
    candidates = [
        (3, 0.2),  # Cubic, moderate knots
        (3, 0.5),  # Cubic, dense knots
        (3, 0.8),  # Cubic, very dense
        (4, 0.3),  # Quartic, moderate
        (5, 0.4),  # Quintic, moderate
    ]
    
    best_val_error = np.inf
    best_model_params = None
    best_spline = None
    
    # Simple K-Fold like validation: split into train/val
    # Since n is small (200), we can do a few random splits or just one holdout
    # To be robust, let's do 3 random splits
    
    rng = np.random.default_rng(42)
    
    for deg, knot_frac in candidates:
        num_knots = max(1, int(n_fit * knot_frac))
        # Ensure num_knots is not too high relative to n_fit
        if num_knots >= n_fit:
            num_knots = n_fit - 1
            
        total_error = 0
        n_splits = 3
        for _ in range(n_splits):
            if time.time() > deadline:
                break
                
            # Shuffle for split
            indices = rng.permutation(n_fit)
            split_idx = int(n_fit * 0.8)
            train_idx = indices[:split_idx]
            val_idx = indices[split_idx:]
            
            X_tr = x_fit[train_idx]
            y_tr = y_fit[train_idx]
            X_val = x_fit[val_idx]
            y_val = y_fit[val_idx]
            
            # Sort training data again for this subset
            sort_idx_tr = np.argsort(X_tr)
            X_tr_s = X_tr[sort_idx_tr]
            y_tr_s = y_tr[sort_idx_tr]
            
            try:
                # s=0 for exact interpolation
                # k=degree
                # w=weights (uniform)
                # Check if num_knots is valid. UnivariateSpline takes k and s.
                # The number of knots is determined by the data if s=0?
                # Actually, UnivariateSpline(x, y, k=3, s=0) creates a spline with knots at every point if s=0?
                # No, it finds a smooth spline. If s=0, it interpolates.
                # The number of knots is roughly n - k.
                # To control knot density, we can use `splrep` or just rely on `UnivariateSpline`.
                # UnivariateSpline with s=0 will interpolate all points. This is equivalent to a spline with knots at every point (or close to it).
                # This might be too flexible (overfitting to numerical noise if any).
                # But the task says noiseless. So interpolation is ideal.
                
                # Let's just use UnivariateSpline with s=0 and vary the degree.
                # The knot density is effectively maxed out for interpolation.
                
                spl = UnivariateSpline(X_tr_s, y_tr_s, k=min(deg, n_fit-1), s=0)
                pred_val = spl(X_val)
                error = np.mean((pred_val - y_val) ** 2)
                total_error += error
            except Exception:
                total_error = np.inf
                break
                
        if total_error < best_val_error:
            best_val_error = total_error / n_splits
            best_model_params = (deg, knot_frac)
            
    # Refit on all data with the best degree
    if best_model_params is None:
        # Fallback
        deg = 3
    else:
        deg = best_model_params[0]
        
    try:
        final_spline = UnivariateSpline(x_fit, y_fit, k=min(deg, n_fit-1), s=0)
    except Exception:
        # Fallback to a simpler model if spline fails
        # Linear regression
        w, *_ = np.linalg.lstsq(np.column_stack([x_fit, np.ones(n_fit)]), y_fit, rcond=None)
        def predict(X_test):
            return X_test[:, 0] * w[0] + w[1]
        return predict

    def predict(X_test):
        x_test = X_test[:, 0]
        return final_spline(x_test)

    return predict
# EVOLVE-BLOCK-END
