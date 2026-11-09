# EVOLVE-BLOCK-START
import numpy as np
from scipy.interpolate import PchipInterpolator, CubicSpline, BaryRational, make_interp_spline
import time

def fit(X_train, y_train):
    """
    Fits a 1D interpolating spline to noiseless data using adaptive approach.
    Since data is noiseless and dense (200 points), we can achieve near-zero error.
    """
    start_time = time.time()
    deadline = start_time + 290  # Safety margin before 330s limit
    
    X = np.asarray(X_train, dtype=np.float64).flatten()
    y = np.asarray(y_train, dtype=np.float64).flatten()
    
    # Ensure 1D
    X = X.reshape(-1)
    y = y.reshape(-1)
    
    # Sort by X
    sort_idx = np.argsort(X)
    X_sorted = X[sort_idx]
    y_sorted = y[sort_idx]
    
    # Check for duplicate X values and average y if needed
    _, unique_indices = np.unique(X_sorted, return_index=True)
    if len(unique_indices) < len(X_sorted):
        X_unique = X_sorted[unique_indices]
        y_unique = np.array([y_sorted[X_sorted == xv].mean() for xv in X_unique])
    else:
        X_unique = X_sorted
        y_unique = y_sorted
    
    n_points = len(X_unique)
    
    # Try different interpolation strategies and pick the best one
    best_predict = None
    best_score = np.inf
    
    # Strategy 1: PCHIP (monotonicity-preserving)
    try:
        interp = PchipInterpolator(X_unique, y_unique)
        y_pred_train = interp(X_unique)
        train_mse = np.mean((y_pred_train - y_unique)**2)
        if train_mse < best_score:
            best_score = train_mse
            best_predict = lambda X_test: interp(np.asarray(X_test, dtype=np.float64).flatten())
    except Exception:
        pass
    
    # Strategy 2: Cubic Spline
    try:
        cs = CubicSpline(X_unique, y_unique)
        y_pred_train = cs(X_unique)
        train_mse = np.mean((y_pred_train - y_unique)**2)
        if train_mse < best_score:
            best_score = train_mse
            best_predict = lambda X_test: cs(np.asarray(X_test, dtype=np.float64).flatten())
    except Exception:
        pass
    
    # Strategy 3: Barycentric rational interpolation (high accuracy)
    try:
        if n_points >= 2:
            br = BaryRational(X_unique, y_unique)
            y_pred_train = br(X_unique)
            train_mse = np.mean((y_pred_train - y_unique)**2)
            if train_mse < best_score:
                best_score = train_mse
                best_predict = lambda X_test: br(np.asarray(X_test, dtype=np.float64).flatten())
    except Exception:
        pass
    
    # Strategy 4: Higher-order spline with adaptive knots
    try:
        # Use make_interp_spline with higher degree for better accuracy
        for k in [4, 5, 6]:
            if n_points > k:
                spline = make_interp_spline(X_unique, y_unique, k=k)
                y_pred_train = spline(X_unique)
                train_mse = np.mean((y_pred_train - y_unique)**2)
                if train_mse < best_score:
                    best_score = train_mse
                    best_predict = lambda X_test, s=spline: s(np.asarray(X_test, dtype=np.float64).flatten())
    except Exception:
        pass
    
    # Strategy 5: Piecewise polynomial with local fitting
    try:
        # Divide into segments and fit locally
        n_segments = min(10, n_points // 5)
        if n_segments >= 1:
            segment_bounds = np.linspace(0, n_points, n_segments + 1).astype(int)
            segment_models = []
            
            for i in range(n_segments):
                start = segment_bounds[i]
                end = segment_bounds[i + 1]
                if end - start < 2:
                    continue
                x_seg = X_unique[start:end]
                y_seg = y_unique[start:end]
                try:
                    # Fit a polynomial locally
                    coeffs = np.polyfit(x_seg, y_seg, min(5, len(x_seg) - 1))
                    segment_models.append((x_seg[0], x_seg[-1], coeffs))
                except Exception:
                    continue
            
            if segment_models:
                def predict_piecewise(X_test):
                    X_test = np.asarray(X_test, dtype=np.float64).flatten()
                    y_pred = np.zeros_like(X_test)
                    for x_val, y_val in zip(X_test, np.zeros_like(X_test)):
                        for x_min, x_max, coeffs in segment_models:
                            if x_min <= x_val <= x_max:
                                y_val = np.polyval(coeffs, x_val)
                                break
                        else:
                            # Extrapolate using nearest segment
                            nearest = min(segment_models, key=lambda s: min(abs(x_val - s[0]), abs(x_val - s[1])))
                            y_val = np.polyval(nearest[2], x_val)
                        y_pred = np.append(y_pred, y_val)
                    return y_pred
                best_predict = predict_piecewise
    except Exception:
        pass
    
    # Fallback to linear interpolation if nothing worked
    if best_predict is None:
        def predict_linear(X_test):
            X_test = np.asarray(X_test, dtype=np.float64).flatten()
            return np.interp(X_test, X_unique, y_unique)
        best_predict = predict_linear
    
    return best_predict
# EVOLVE-BLOCK-END
