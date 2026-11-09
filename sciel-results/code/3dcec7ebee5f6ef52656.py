import numpy as np
import time
from scipy.interpolate import CubicSpline, interp1d, PchipInterpolator
from scipy.optimize import curve_fit

def fit(X_train, y_train):
    """
    Multi-strategy regression for 1D noiseless data.
    Tries multiple interpolation/fitting approaches and selects the best
    based on validation error on a held-out subset.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # 20s safety margin before 300s limit
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    
    # Flatten if 2D with 1 column
    if X.ndim == 2 and X.shape[1] == 1:
        x = X.flatten()
    elif X.ndim == 1:
        x = X
    else:
        x = X[:, 0] if X.ndim == 2 else X
    
    # Sort by x
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    
    # Remove duplicate x values
    unique_x, unique_indices = np.unique(x_sorted, return_index=True)
    if len(unique_x) < len(x_sorted):
        y_unique = np.zeros_like(unique_x)
        for i, idx in enumerate(unique_indices):
            mask = (x_sorted == unique_x[i])
            y_unique[i] = np.mean(y_sorted[mask])
        x_sorted = unique_x
        y_sorted = y_unique
    
    n = len(x_sorted)
    if n < 2:
        y_mean = np.mean(y_sorted) if n > 0 else 0.0
        def predict_const(X_test):
            X_test = np.asarray(X_test, dtype=np.float64)
            return np.full(X_test.shape[0], y_mean)
        return predict_const
    
    # Split into train and validation sets for model selection
    # Use a deterministic split based on position
    val_frac = 0.15
    n_val = max(5, int(n * val_frac))
    n_train = n - n_val
    
    # Take evenly spaced points for validation to cover the domain
    val_indices = np.linspace(0, n - 1, n_val, dtype=int)
    train_indices = np.array([i for i in range(n) if i not in set(val_indices.tolist())])
    
    x_train_sel = x_sorted[train_indices]
    y_train_sel = y_sorted[train_indices]
    x_val = x_sorted[val_indices]
    y_val = y_sorted[val_indices]
    
    candidates = []
    
    # Strategy 1: Cubic spline with not-a-knot
    if time.time() < deadline:
        try:
            spline1 = CubicSpline(x_train_sel, y_train_sel, bc_type='not-a-knot')
            pred_val1 = spline1(x_val)
            pred_val1 = np.nan_to_num(pred_val1, nan=0.0, posinf=1e10, neginf=-1e10)
            mse1 = np.mean((pred_val1 - y_val) ** 2)
            candidates.append(('spline_nak', spline1, mse1))
        except Exception:
            pass
    
    # Strategy 2: Cubic spline with natural boundary conditions
    if time.time() < deadline:
        try:
            spline2 = CubicSpline(x_train_sel, y_train_sel, bc_type='natural')
            pred_val2 = spline2(x_val)
            pred_val2 = np.nan_to_num(pred_val2, nan=0.0, posinf=1e10, neginf=-1e10)
            mse2 = np.mean((pred_val2 - y_val) ** 2)
            candidates.append(('spline_nat', spline2, mse2))
        except Exception:
            pass
    
    # Strategy 3: PCHIP (Piecewise Cubic Hermite Interpolating Polynomial)
    if time.time() < deadline:
        try:
            pchip = PchipInterpolator(x_train_sel, y_train_sel)
            pred_val3 = pchip(x_val)
            pred_val3 = np.nan_to_num(pred_val3, nan=0.0, posinf=1e10, neginf=-1e10)
            mse3 = np.mean((pred_val3 - y_val) ** 2)
            candidates.append(('pchip', pchip, mse3))
        except Exception:
            pass
    
    # Strategy 4: Linear interpolation
    if time.time() < deadline:
        try:
            lin = interp1d(x_train_sel, y_train_sel, kind='linear', fill_value='extrapolate')
            pred_val4 = lin(x_val)
            pred_val4 = np.nan_to_num(pred_val4, nan=0.0, posinf=1e10, neginf=-1e10)
            mse4 = np.mean((pred_val4 - y_val) ** 2)
            candidates.append(('linear', lin, mse4))
        except Exception:
            pass
    
    # Strategy 5: Quadratic polynomial fit (global)
    if time.time() < deadline:
        try:
            coeffs = np.polyfit(x_train_sel, y_train_sel, 2)
            def quad_interp(xq):
                return np.polyval(coeffs, xq)
            pred_val5 = quad_interp(x_val)
            pred_val5 = np.nan_to_num(pred_val5, nan=0.0, posinf=1e10, neginf=-1e10)
            mse5 = np.mean((pred_val5 - y_val) ** 2)
            candidates.append(('quad', quad_interp, mse5))
        except Exception:
            pass
    
    # Strategy 6: Cubic polynomial fit (global)
    if time.time() < deadline:
        try:
            coeffs = np.polyfit(x_train_sel, y_train_sel, 3)
            def cubic_interp(xq):
                return np.polyval(coeffs, xq)
            pred_val6 = cubic_interp(x_val)
            pred_val6 = np.nan_to_num(pred_val6, nan=0.0, posinf=1e10, neginf=-1e10)
            mse6 = np.mean((pred_val6 - y_val) ** 2)
            candidates.append(('cubic_poly', cubic_interp, mse6))
        except Exception:
            pass
    
    # Strategy 7: RANSAC-like approach with local quadratic fitting
    if time.time() < deadline:
        try:
            # Use a subset of points and fit locally
            n_local = min(50, n_train)
            local_indices = np.linspace(0, n_train - 1, n_local, dtype=int)
            x_local = x_train_sel[local_indices]
            y_local = y_train_sel[local_indices]
            
            # Fit a quadratic spline using local points
            if len(x_local) >= 4:
                spline7 = CubicSpline(x_local, y_local, bc_type='not-a-knot')
                pred_val7 = spline7(x_val)
                pred_val7 = np.nan_to_num(pred_val7, nan=0.0, posinf=1e10, neginf=-1e10)
                mse7 = np.mean((pred_val7 - y_val) ** 2)
                candidates.append(('local_spline', spline7, mse7))
        except Exception:
            pass
    
    # Strategy 8: Use all points with cubic spline (no validation split)
    if time.time() < deadline:
        try:
            spline8 = CubicSpline(x_sorted, y_sorted, bc_type='not-a-knot')
            # For this candidate, evaluate on full training data as proxy
            pred_val8 = spline8(x_val)
            pred_val8 = np.nan_to_num(pred_val8, nan=0.0, posinf=1e10, neginf=-1e10)
            mse8 = np.mean((pred_val8 - y_val) ** 2)
            candidates.append(('full_spline', spline8, mse8))
        except Exception:
            pass
    
    # Strategy 9: Use all points with PCHIP
    if time.time() < deadline:
        try:
            pchip9 = PchipInterpolator(x_sorted, y_sorted)
            pred_val9 = pchip9(x_val)
            pred_val9 = np.nan_to_num(pred_val9, nan=0.0, posinf=1e10, neginf=-1e10)
            mse9 = np.mean((pred_val9 - y_val) ** 2)
            candidates.append(('full_pchip', pchip9, mse9))
        except Exception:
            pass
    
    # Strategy 10: Weighted average of top 2 candidates
    if len(candidates) >= 2:
        candidates_sorted = sorted(candidates, key=lambda c: c[2])
        top2 = candidates_sorted[:2]
        # Create an ensemble that averages the top 2
        def ensemble_predict(xq):
            preds = []
            for name, model, mse in top2:
                try:
                    p = model(xq)
                    p = np.nan_to_num(p, nan=0.0, posinf=1e10, neginf=-1e10)
                    preds.append(p)
                except Exception:
                    pass
            if preds:
                return np.mean(preds, axis=0)
            else:
                return np.zeros_like(xq)
        pred_val_ens = ensemble_predict(x_val)
        mse_ens = np.mean((pred_val_ens - y_val) ** 2)
        candidates.append(('ensemble', ensemble_predict, mse_ens))
    
    # Select the best candidate
    if candidates:
        candidates_sorted = sorted(candidates, key=lambda c: c[2])
        best_name, best_model, best_mse = candidates_sorted[0]
        
        def predict(X_test):
            X_test = np.asarray(X_test, dtype=np.float64)
            if X_test.ndim == 2 and X_test.shape[1] == 1:
                x_test = X_test.flatten()
            elif X_test.ndim == 1:
                x_test = X_test
            else:
                x_test = X_test[:, 0]
                
            try:
                pred = best_model(x_test)
            except Exception:
                # Fallback to linear interpolation on all data
                try:
                    lin_fb = interp1d(x_sorted, y_sorted, kind='linear', fill_value='extrapolate')
                    pred = lin_fb(x_test)
                except Exception:
                    pred = np.full_like(x_test, np.mean(y_sorted))
            
            pred = np.nan_to_num(pred, nan=0.0, posinf=1e10, neginf=-1e10)
            return pred
    else:
        # All strategies failed, fallback to linear interpolation
        try:
            lin = interp1d(x_sorted, y_sorted, kind='linear', fill_value='extrapolate')
        except Exception:
            y_mean = np.mean(y_sorted)
            def predict_const(X_test):
                X_test = np.asarray(X_test, dtype=np.float64)
                return np.full(X_test.shape[0], y_mean)
            return predict_const
        
        def predict(X_test):
            X_test = np.asarray(X_test, dtype=np.float64)
            if X_test.ndim == 2 and X_test.shape[1] == 1:
                x_test = X_test.flatten()
            elif X_test.ndim == 1:
                x_test = X_test
            else:
                x_test = X_test[:, 0]
            pred = lin(x_test)
            pred = np.nan_to_num(pred, nan=0.0, posinf=1e10, neginf=-1e10)
            return pred
    
    return predict
