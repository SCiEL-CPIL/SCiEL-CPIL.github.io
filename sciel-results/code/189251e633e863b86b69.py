"""Rewritten program using non-parametric interpolation with model selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.interpolate import PchipInterpolator, CubicSpline, interp1d

def fit(X_train, y_train):
    start_time = time.time()
    DEADLINE = 320.0  # Safety margin before 330s limit

    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape

    # Handle trivial cases
    if n == 0:
        return lambda X: np.zeros(len(X))
    
    y_mean = np.mean(y)
    y_var = np.var(y)
    if y_var < 1e-12:
        c = y_mean
        return lambda X: np.full(len(X), c)

    # All inputs are 1D based on the task description
    if d != 1:
        # Fallback for multi-dim (should not happen per contract, but safe to have)
        # Use a simple linear regression as a last resort
        X_aug = np.column_stack([np.ones(n), X])
        try:
            w, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
            def predict_linear(X_pred):
                X_pred = np.asarray(X_pred, dtype=np.float64)
                if X_pred.ndim == 1:
                    X_pred = X_pred.reshape(-1, 1)
                X_aug_pred = np.column_stack([np.ones(len(X_pred)), X_pred])
                return X_aug_pred @ w
            return predict_linear
        except:
            return lambda X: np.full(len(X), y_mean)

    x = X[:, 0]
    
    # Sort by x
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]

    # Handle duplicates by averaging
    unique_x, inverse, counts = np.unique(x_sorted, return_inverse=True, return_counts=True)
    if len(unique_x) < n:
        y_sum = np.zeros(len(unique_x))
        np.add.at(y_sum, inverse, y_sorted)
        y_unique = y_sum / counts
        x_unique = unique_x
    else:
        x_unique = x_sorted
        y_unique = y_sorted

    n_unique = len(x_unique)
    
    if n_unique < 2:
        return lambda X: np.full(len(X), y_mean)

    # Define candidate interpolators
    # We want to avoid oscillation, so PCHIP is preferred.
    # Cubic Spline is C2 but can oscillate.
    # Linear is C0, no oscillation but less accurate for smooth curves.
    
    models = []
    
    # 1. PCHIP (Monotone cubic, no overshoot)
    if n_unique >= 2:
        try:
            pchip = PchipInterpolator(x_unique, y_unique)
            models.append(('pchip', pchip))
        except Exception:
            pass

    # 2. Cubic Spline (Not-a-knot)
    if n_unique >= 4:
        try:
            cs = CubicSpline(x_unique, y_unique, bc_type='not-a-knot')
            models.append(('cs', cs))
        except Exception:
            pass

    # 3. Linear Interpolation
    if n_unique >= 2:
        try:
            lin = interp1d(x_unique, y_unique, kind='linear', fill_value='extrapolate')
            models.append(('lin', lin))
        except Exception:
            pass

    if not models:
        # Fallback if all failed
        return lambda X: np.full(len(X), y_mean)

    # Model Selection:
    # Since data is noiseless, we can use a simple hold-out validation.
    # Split sorted data into train/val for selection.
    # However, with 200 points, we can also just check the max residual on the training set.
    # But for smooth functions, all interpolators will have near-zero residual on training points.
    # The difference is in the behavior between points.
    # A robust metric for "smoothness" without noise is the total variation of the derivative or simply picking PCHIP by default as it's the most robust for unknown smooth functions.
    
    # Let's do a quick cross-validation on a subset to be safe.
    # Use the last 20% of points as validation set for model selection.
    split_idx = int(0.8 * n_unique)
    x_train_sel = x_unique[:split_idx]
    y_train_sel = y_unique[:split_idx]
    x_val_sel = x_unique[split_idx:]
    y_val_sel = y_unique[split_idx:]
    
    best_model = None
    best_score = np.inf
    
    if len(x_val_sel) > 0:
        for name, model in models:
            try:
                # Re-fit model on the subset? 
                # No, the model is already fit on all data. 
                # For fair selection, we should fit on train_sel and predict on val_sel.
                # But re-fitting PCHIP/CS is fast.
                
                if name == 'pchip':
                    m_train = PchipInterpolator(x_train_sel, y_train_sel)
                    y_pred_val = m_train(x_val_sel)
                elif name == 'cs':
                    if len(x_train_sel) < 4:
                        continue
                    m_train = CubicSpline(x_train_sel, y_train_sel, bc_type='not-a-knot')
                    y_pred_val = m_train(x_val_sel)
                elif name == 'lin':
                    m_train = interp1d(x_train_sel, y_train_sel, kind='linear', fill_value='extrapolate')
                    y_pred_val = m_train(x_val_sel)
                else:
                    continue

                # Calculate MSE on validation set
                mse = np.mean((y_pred_val - y_val_sel)**2)
                
                # Also check if the model is stable (no NaNs)
                if np.all(np.isfinite(y_pred_val)) and mse < best_score:
                    best_score = mse
                    best_model = name
                    
            except Exception:
                continue
        
        # If validation fails or is empty, default to PCHIP as it's the most robust
        if best_model is None:
            best_model = 'pchip' if any(m[0]=='pchip' for m in models) else models[0][0]
    else:
        # If no validation set (very few points), just pick PCHIP
        best_model = 'pchip' if any(m[0]=='pchip' for m in models) else models[0][0]

    # Retrieve the full model fit on all data
    final_model = None
    for name, model in models:
        if name == best_model:
            final_model = model
            break
    
    if final_model is None:
        # Fallback to first model
        final_model = models[0][1]

    # Define the prediction function
    # We need to handle extrapolation safely.
    # The test points are in the same range as training, so extrapolation shouldn't happen often.
    # However, if it does, linear extrapolation is safer than cubic oscillation.
    # PCHIP and CubicSpline extrapolate by default in a potentially unstable way.
    # We will clip the input x to the range of x_unique to ensure we stay in the interpolation region,
    # or use the boundary values. Given the uniform distribution, clipping is a safe assumption.
    
    x_min = x_unique[0]
    x_max = x_unique[-1]
    y_min = np.min(y_unique)
    y_max = np.max(y_unique)

    def predict(X_pred):
        X_pred = np.asarray(X_pred, dtype=np.float64)
        if X_pred.ndim == 1:
            X_pred = X_pred.reshape(-1, 1)
        x_pred = X_pred[:, 0]
        
        # Clip to training range to prevent extrapolation artifacts
        x_pred_clipped = np.clip(x_pred, x_min, x_max)
        
        try:
            y_pred = final_model(x_pred_clipped)
        except Exception:
            y_pred = np.full(len(x_pred), y_mean)
            
        # Ensure finite values
        y_pred = np.nan_to_num(y_pred, nan=y_mean, posinf=y_max, neginf=y_min)
        
        # Optional: Clip output to reasonable range to prevent wild swings if extrapolation logic was flawed
        # But since we clipped input, output should be within [y_min, y_max] for PCHIP/Linear.
        # CubicSpline might overshoot slightly.
        # Let's not clip output unless it's extremely far, to avoid bias.
        
        return y_pred

    return predict

# EVOLVE-BLOCK-END
