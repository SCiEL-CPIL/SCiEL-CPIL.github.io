import numpy as np
import time
from scipy.interpolate import PchipInterpolator, CubicSpline, interp1d, UnivariateSpline

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape
    
    if d != 1:
        # Fallback for d > 1 (unlikely)
        X_aug = np.column_stack([np.ones(n), X])
        w, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
        def predict(X_new):
            X_new = np.asarray(X_new, dtype=np.float64)
            X_new_aug = np.column_stack([np.ones(X_new.shape[0]), X_new])
            return X_new_aug @ w
        return predict

    # 1D Specific Logic
    x = X.flatten()
    
    # Handle duplicates: If x values are identical, y should be identical (noiseless).
    # We aggregate to ensure strict monotonicity for splines.
    unique_x, inverse_indices = np.unique(x, return_inverse=True)
    if len(unique_x) < n:
        # Average y for duplicate x values
        y_sum = np.zeros(len(unique_x))
        counts = np.zeros(len(unique_x))
        np.add.at(y_sum, inverse_indices, y)
        np.add.at(counts, inverse_indices, 1)
        y = y_sum / counts
        x = unique_x
    else:
        # Sort if not already
        sort_idx = np.argsort(x)
        x = x[sort_idx]
        y = y[sort_idx]
    
    n_unique = len(x)
    
    # Validation Split
    # Use a deterministic split based on indices to avoid randomness issues
    # Stratified-ish split: take every 10th point for validation? 
    # Or just random with fixed seed.
    np.random.seed(42)
    indices = np.arange(n_unique)
    np.random.shuffle(indices)
    val_size = max(10, n_unique // 10)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    
    x_val = x[val_idx]
    y_val = y[val_idx]
    x_tr = x[train_idx]
    y_tr = y[train_idx]
    
    best_model = None
    best_val_mse = np.inf
    
    # Helper to evaluate a model
    def evaluate_model(model_func, x_new, y_true):
        try:
            pred = model_func(x_new)
            if not np.all(np.isfinite(pred)):
                return np.inf
            mse = np.mean((pred - y_true)**2)
            return mse
        except Exception:
            return np.inf

    # Candidate 1: PchipInterpolator (Monotonicity preserving, no overshoot)
    def make_pchip(x_tr, y_tr):
        return PchipInterpolator(x_tr, y_tr)
    
    # Candidate 2: CubicSpline
    def make_cubic(x_tr, y_tr):
        return CubicSpline(x_tr, y_tr)
    
    # Candidate 3: UnivariateSpline with s=0 (Interpolation)
    # UnivariateSpline can be unstable if not careful, but s=0 is interpolation.
    def make_unispline(x_tr, y_tr):
        return UnivariateSpline(x_tr, y_tr, s=0)
        
    # Candidate 4: Linear Interpolation
    def make_linear(x_tr, y_tr):
        return interp1d(x_tr, y_tr, kind='linear', fill_value="extrapolate")
        
    # Candidate 5: Low degree Polynomial (2, 3, 4)
    def make_poly(deg):
        def _make(x_tr, y_tr):
            coeffs = np.polyfit(x_tr, y_tr, deg)
            def predict(x_new):
                return np.polyval(coeffs, x_new)
            return predict
        return _make

    candidates = [
        ("pchip", make_pchip),
        ("cubic", make_cubic),
        ("unispline", make_unispline),
        ("linear", make_linear),
        ("poly2", make_poly(2)),
        ("poly3", make_poly(3)),
        ("poly4", make_poly(4)),
    ]
    
    for name, factory in candidates:
        if time.time() > deadline:
            break
        try:
            model = factory(x_tr, y_tr)
            mse = evaluate_model(model, x_val, y_val)
            if mse < best_val_mse:
                best_val_mse = mse
                best_model = model
                best_name = name
        except Exception:
            pass
            
    # If no model worked, fallback to mean
    if best_model is None:
        y_mean = np.mean(y)
        def predict(X_new):
            return np.full(len(X_new), y_mean)
        return predict

    # Refit best model on ALL data
    # We need to identify which factory to use for refitting
    # Since we stored the instance, we can just use it if it was fitted on all data? 
    # No, we fitted on x_tr, y_tr. We must refit on x, y.
    
    # Re-run the factory on full data
    final_model = None
    for name, factory in candidates:
        if name == best_name:
            try:
                final_model = factory(x, y)
                break
            except Exception:
                pass
                
    if final_model is None:
        # Fallback to Pchip on all data
        try:
            final_model = PchipInterpolator(x, y)
        except Exception:
            y_mean = np.mean(y)
            def predict(X_new):
                return np.full(len(X_new), y_mean)
            return predict

    def predict(X_new):
        X_new = np.asarray(X_new, dtype=np.float64).flatten()
        # Clip predictions to reasonable range? No, let it extrapolate.
        # But handle exceptions.
        try:
            pred = final_model(X_new)
            # Ensure finite
            pred = np.where(np.isfinite(pred), pred, np.nan)
            # If any nan, replace with local mean or 0? 
            # The task says "one finite prediction per row".
            # If extrapolation fails (e.g. polynomial overflow), we might get inf/nan.
            # Let's replace non-finite with the mean of y_train or the last valid value.
            if not np.all(np.isfinite(pred)):
                # Simple fallback: replace non-finite with mean of y
                mean_y = np.mean(y)
                pred = np.where(np.isfinite(pred), pred, mean_y)
            return pred
        except Exception:
            # Total failure, return mean
            return np.full(len(X_new), np.mean(y))

    return predict
