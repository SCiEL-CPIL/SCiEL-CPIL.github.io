import numpy as np
import time
from math import comb

def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial approximation to the unresolved fast-scale forcing.
    Returns 6 coefficients in ascending powers of x.
    """
    start_time = time.time()
    deadline = 140.0

    # Ensure inputs are float64 numpy arrays
    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)

    # Handle edge cases
    if len(x) == 0 or len(y) == 0:
        return np.zeros(6)
    if len(x) != len(y):
        n = min(len(x), len(y))
        x = x[:n]
        y = y[:n]
    
    n = len(x)
    if n < 6:
        # Fallback for very small data
        deg = max(0, n - 1)
        coeff = np.zeros(6)
        try:
            c = np.polynomial.polynomial.polyfit(x, y, deg)
            coeff[:deg+1] = c
        except Exception:
            pass
        return coeff

    # Normalize x to [-1, 1] for better conditioning
    x_min, x_max = x.min(), x.max()
    x_range = x_max - x_min
    if x_range < 1e-12:
        return np.array([y.mean(), 0, 0, 0, 0, 0], dtype=np.float64)
    
    mid = (x_min + x_max) / 2.0
    half_range = x_range / 2.0
    x_norm = (x - mid) / half_range

    # Prepare data for fitting
    # We want to fit in normalized space, then convert back.
    
    # Model 1: Odd Polynomial (x, x^3, x^5)
    # Model 2: Full Polynomial (1, x, x^2, x^3, x^4, x^5)
    
    # Split for validation if possible
    use_holdout = n > 50
    if use_holdout:
        rng = np.random.default_rng(42)
        indices = rng.permutation(n)
        n_val = max(10, n // 10)
        val_idx = indices[:n_val]
        tr_idx = indices[n_val:]
        x_tr, y_tr = x_norm[tr_idx], y[tr_idx]
        x_val, y_val = x_norm[val_idx], y[val_idx]
    else:
        x_tr, y_tr = x_norm, y
        x_val, y_val = x_norm, y

    best_coeff_norm = None
    best_score = np.inf
    best_model_type = 'full' # Track which model was best

    # --- Fit Odd Model ---
    # Basis: [x, x^3, x^5]
    V_odd = np.column_stack([x_tr**1, x_tr**3, x_tr**5])
    
    # Try a few regularization strengths for the odd model
    # Scale lambda by 1/n for consistent penalty
    for lam in [0.0, 1e-6/n, 1e-4/n, 1e-2/n, 1e-1/n]:
        if time.time() > deadline: break
        try:
            if lam == 0.0:
                if V_odd.shape[0] > V_odd.shape[1]:
                    c_odd, _, _, _ = np.linalg.lstsq(V_odd, y_tr, rcond=None)
                else:
                    c_odd = np.zeros(3)
                    c_odd[0] = np.dot(y_tr, x_tr) / np.dot(x_tr, x_tr) # Fallback
            else:
                # Ridge: no penalty on any term for odd model (all are "slope-like")
                A = V_odd.T @ V_odd + lam * np.eye(3)
                b = V_odd.T @ y_tr
                c_odd = np.linalg.solve(A, b)
            
            if not np.all(np.isfinite(c_odd)): continue
            
            # Evaluate on validation
            pred_val = c_odd[0]*x_val**1 + c_odd[1]*x_val**3 + c_odd[2]*x_val**5
            mse = np.mean((pred_val - y_val)**2)
            
            if mse < best_score:
                best_score = mse
                # Store coefficients in full 6-vector format (zeros for even powers)
                temp_coeff = np.zeros(6)
                temp_coeff[1] = c_odd[0]
                temp_coeff[3] = c_odd[1]
                temp_coeff[5] = c_odd[2]
                best_coeff_norm = temp_coeff
                best_model_type = 'odd'
        except Exception:
            continue

    # --- Fit Full Model with proper ridge (exclude intercept from penalty) ---
    V_full = np.polynomial.polynomial.polyvander(x_tr, 5)
    
    for lam in [0.0, 1e-6/n, 1e-4/n, 1e-2/n, 1e-1/n]:
        if time.time() > deadline: break
        try:
            if lam == 0.0:
                if V_full.shape[0] > V_full.shape[1]:
                    c_full, _, _, _ = np.linalg.lstsq(V_full, y_tr, rcond=None)
                else:
                    c_full = np.zeros(6)
                    c_full[0] = y_tr.mean()
            else:
                # Ridge with intercept excluded from penalty
                # Penalty matrix: zeros for intercept, lam for others
                P = lam * np.eye(6)
                P[0, 0] = 0.0  # No penalty on intercept
                A = V_full.T @ V_full + P
                b = V_full.T @ y_tr
                c_full = np.linalg.solve(A, b)
            
            if not np.all(np.isfinite(c_full)): continue
            
            # Evaluate on validation
            pred_val = np.polynomial.polynomial.polyval(x_val, c_full)
            mse = np.mean((pred_val - y_val)**2)
            
            # Heuristic: Prefer Odd model if it's close (within 10%) or better.
            # The "Knowledge" suggests Odd generalizes better to unseen regimes.
            # So we only switch to Full if it's significantly better on validation.
            
            if best_model_type == 'odd':
                # Only accept Full if it's dramatically better to avoid overfitting
                if mse < best_score * 0.85: 
                    best_score = mse
                    best_coeff_norm = c_full
                    best_model_type = 'full'
            else:
                if mse < best_score:
                    best_score = mse
                    best_coeff_norm = c_full
                    best_model_type = 'full'
        except Exception:
            continue

    # --- Weighted Odd Model: emphasize high-|x| regions (F=10 regime proxy) ---
    # Weight samples by |x|^2 to focus on larger displacements where F-dependence matters
    weights = x_tr**2 + 1e-12
    sqrt_w = np.sqrt(weights)
    V_odd_w = V_odd * sqrt_w[:, None]
    y_tr_w = y_tr * sqrt_w
    
    for lam in [0.0, 1e-6/n, 1e-4/n]:
        if time.time() > deadline: break
        try:
            if lam == 0.0:
                if V_odd_w.shape[0] > V_odd_w.shape[1]:
                    c_odd_w, _, _, _ = np.linalg.lstsq(V_odd_w, y_tr_w, rcond=None)
                else:
                    c_odd_w = np.zeros(3)
            else:
                A = V_odd_w.T @ V_odd_w + lam * np.eye(3)
                b = V_odd_w.T @ y_tr_w
                c_odd_w = np.linalg.solve(A, b)
            
            if not np.all(np.isfinite(c_odd_w)): continue
            
            # Evaluate on validation with same weighting
            pred_val = c_odd_w[0]*x_val**1 + c_odd_w[1]*x_val**3 + c_odd_w[2]*x_val**5
            w_val = x_val**2 + 1e-12
            mse_w = np.mean(w_val * (pred_val - y_val)**2)
            
            # Also compute unweighted MSE for comparison
            mse_unw = np.mean((pred_val - y_val)**2)
            
            # Use unweighted MSE for comparison since validation score is unweighted
            if mse_unw < best_score:
                best_score = mse_unw
                temp_coeff = np.zeros(6)
                temp_coeff[1] = c_odd_w[0]
                temp_coeff[3] = c_odd_w[1]
                temp_coeff[5] = c_odd_w[2]
                best_coeff_norm = temp_coeff
                best_model_type = 'odd_weighted'
        except Exception:
            continue

    # If no valid coefficient was found, fall back to linear fit
    if best_coeff_norm is None:
        try:
            c = np.polynomial.polynomial.polyfit(x_norm, y, 1)
            best_coeff_norm = np.zeros(6)
            best_coeff_norm[:2] = c
        except Exception:
            best_coeff_norm = np.zeros(6)
            best_coeff_norm[0] = y.mean()

    # --- Final polish: refit on ALL data with best model structure ---
    if time.time() < deadline - 10.0:
        try:
            if best_model_type == 'odd' or best_model_type == 'odd_weighted':
                # Refit odd model on all data
                V_all_odd = np.column_stack([x_norm**1, x_norm**3, x_norm**5])
                c_all, _, _, _ = np.linalg.lstsq(V_all_odd, y, rcond=None)
                if np.all(np.isfinite(c_all)):
                    best_coeff_norm = np.zeros(6)
                    best_coeff_norm[1] = c_all[0]
                    best_coeff_norm[3] = c_all[1]
                    best_coeff_norm[5] = c_all[2]
            elif best_model_type == 'full':
                # Refit full model on all data with light ridge
                V_all_full = np.polynomial.polynomial.polyvander(x_norm, 5)
                lam_polish = 1e-6 / n
                P = lam_polish * np.eye(6)
                P[0, 0] = 0.0
                A = V_all_full.T @ V_all_full + P
                b = V_all_full.T @ y
                c_all = np.linalg.solve(A, b)
                if np.all(np.isfinite(c_all)):
                    best_coeff_norm = c_all
        except Exception:
            pass  # Keep previous best

    # Convert from normalized x to original x
    # p(x_norm) = sum c_i x_norm^i
    # x_norm = (x - mid) / half_range
    # We need d_k such that sum d_k x^k = p((x-mid)/half_range)
    
    d = np.zeros(6)
    a = 1.0 / half_range
    b_const = -mid / half_range
    
    for k in range(0, 6):
        s = 0.0
        for i in range(k, 6):
            ci = best_coeff_norm[i]
            if ci == 0.0: continue
            # Coefficient of x^k in (a*x + b)^i is a^k * b^(i-k) * comb(i, k)
            s += ci * (a ** k) * (b_const ** (i - k)) * comb(i, k)
        d[k] = s

    # Ensure finite
    if not np.all(np.isfinite(d)):
        d = np.zeros(6)
        d[0] = y.mean()

    return d
