"""Symbolic regression: per-point ensemble of interpolants with aggressive time management."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings

warnings.filterwarnings('ignore')

# Budget: evaluator kills at 330s. We use 285s as our internal budget to be safe.
_TOTAL_BUDGET = 285.0
_GLOBAL_START = time.time()


def _remaining():
    return _TOTAL_BUDGET - (time.time() - _GLOBAL_START)


def _fit_interpolant(X, y, kind):
    """
    Fit a single-point interpolant (RBF or spline) to all training data.
    Returns a predict function that, given X_new, predicts each y_new[i]
    by fitting a model to the training data using only the i-th point of X_new
    as the 'target' location? No, that's not right.
    
    Actually, the standard approach for these 1D symbolic regression tasks with
    noiseless data is that the test points are distinct from training points,
    but the function is smooth.
    
    A very strong baseline for 1D noiseless regression is to use a high-capacity
    model that interpolates the training data well and generalizes.
    
    However, the 'per-point ensemble' idea is:
    For each test point x_t, we can fit a local model using the training data.
    But we don't know x_t during fit().
    
    So we must return a predict(X) function.
    
    Let's stick to a robust global model selection but with better models.
    The previous approach used K-Fold CV to select the best model from a list.
    The score was 76M, which is terrible. The parent was 27M.
    This suggests the model selection was picking bad models or the models were bad.
    
    Key insight: For 1D noiseless data, an interpolating spline or a high-degree
    polynomial often works well IF the function is smooth.
    However, if the function has sharp features or is non-smooth, it fails.
    
    Let's try a different strategy:
    1. Fit a very flexible model (e.g., RBF with small sigma or high-degree spline)
       that interpolates the training data.
    2. Use a simple heuristic to select the model:
          - If the data looks smooth (low residual variance of a smooth fit), use a smooth interpolant.
          - Otherwise, use a local method.
    
    Actually, the most robust method for 1D noiseless symbolic regression in this
    context (likely simple functions like sin, exp, poly, rational) is:
    
    Option A: High-degree polynomial (e.g., deg 10-20).
    Option B: RBF with sigma tuned to the data scale.
    Option C: Spline with 0 smoothing (interpolating).
    
    The previous code used K-Fold CV. K-Fold CV on 200 points with 5 folds means
    fitting on 160 points. This should be robust. Why did it fail?
    Maybe the RBF sigma was too small or too large, leading to overfitting or
    underfitting.
    
    Let's try a simpler, more robust approach:
    1. Fit an interpolating cubic spline (scipy.interpolate.make_interp_spline).
    2. Fit a high-degree polynomial (e.g., deg 12).
    3. Fit an RBF with sigma = x_range / 10.
    4. Fit an RBF with sigma = x_range / 5.
    5. Fit a linear regression.
    
    Then, select the best one using a small hold-out set or simply by checking
    the training error? No, training error will be 0 for interpolants.
    
    We need a validation strategy. Since we have 200 points, we can do:
    - Split into 180 train / 20 val.
    - Fit models on 180, evaluate on 20.
    - Pick the best.
    - Re-fit on all 200.
    
    This is safer than K-Fold if K-Fold is too slow or noisy.
    
    Let's implement this.
    """
    pass  # Placeholder, actual logic in fit()


def _fit_rbf(X, y, sigma):
    """Fit Gaussian RBF model. Returns a predict function."""
    x = X[:, 0].astype(np.float64)
    y = y.astype(np.float64)
    n = len(x)

    diff = x[:, None] - x[None, :]
    K = np.exp(-diff**2 / (2.0 * sigma**2))

    reg = 1e-6
    K_reg = K + reg * np.eye(n)

    try:
        w = np.linalg.solve(K_reg, y)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(K_reg, y, rcond=None)[0]

    def predict(X_new):
        xn = X_new[:, 0].astype(np.float64)
        diff_new = xn[:, None] - x[None, :]
        K_new = np.exp(-diff_new**2 / (2.0 * sigma**2))
        return K_new @ w

    return predict


def _fit_poly(X, y, deg):
    """Fit polynomial model with scaled input."""
    x = X[:, 0].astype(np.float64)
    y = y.astype(np.float64)

    x_mean = x.mean()
    x_std = x.std()
    if x_std < 1e-10:
        x_std = 1.0

    x_scaled = (x - x_mean) / x_std
    V = np.vander(x_scaled, N=deg + 1, increasing=True)

    try:
        w = np.linalg.lstsq(V, y, rcond=None)[0]
    except Exception:
        w = np.zeros(deg + 1)

    def predict(X_new):
        xn = X_new[:, 0].astype(np.float64)
        xn_scaled = (xn - x_mean) / x_std
        Vn = np.vander(xn_scaled, N=deg + 1, increasing=True)
        return Vn @ w

    return predict


def _fit_spline_interp(X, y):
    """Fit interpolating cubic spline."""
    from scipy.interpolate import make_interp_spline

    x = X[:, 0].astype(np.float64)
    y = y.astype(np.float64)

    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]

    # Remove duplicate x values if any (shouldn't happen with random uniform)
    # But just in case, keep the first occurrence
    unique_x, unique_idx = np.unique(x_sorted, return_index=True)
    if len(unique_x) < 4:
        # Fallback to linear
        A = np.column_stack([np.ones(len(x)), x])
        try:
            w = np.linalg.lstsq(A, y, rcond=None)[0]
            def predict(X_new):
                An = np.column_stack([np.ones(len(X_new)), X_new[:, 0].astype(np.float64)])
                return An @ w
            return predict
        except Exception:
            c = np.mean(y)
            def predict(X_new):
                return np.full(len(X_new), c)
            return predict

    y_unique = y_sorted[unique_idx]
    
    try:
        sp = make_interp_spline(unique_x, y_unique, k=3)
        def predict(X_new):
            return sp(X_new[:, 0].astype(np.float64))
        return predict
    except Exception:
        # Fallback to linear
        A = np.column_stack([np.ones(len(x)), x])
        try:
            w = np.linalg.lstsq(A, y, rcond=None)[0]
            def predict(X_new):
                An = np.column_stack([np.ones(len(X_new)), X_new[:, 0].astype(np.float64)])
                return An @ w
            return predict
        except Exception:
            c = np.mean(y)
            def predict(X_new):
                return np.full(len(X_new), c)
            return predict


def _fit_linear(X, y):
    """Fit linear model."""
    x = X[:, 0].astype(np.float64)
    y = y.astype(np.float64)
    A = np.column_stack([np.ones(len(x)), x])
    try:
        w = np.linalg.lstsq(A, y, rcond=None)[0]
        def predict(X_new):
            An = np.column_stack([np.ones(len(X_new)), X_new[:, 0].astype(np.float64)])
            return An @ w
        return predict
    except Exception:
        c = np.mean(y)
        def predict(X_new):
            return np.full(len(X_new), c)
        return predict


def fit(X_train, y_train):
    global _GLOBAL_START
    _GLOBAL_START = time.time()

    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n = len(X)

    if n < 4:
        c = np.mean(y) if n > 0 else 0.0
        def predict(X_new):
            return np.full(len(X_new), c)
        return predict

    x = X[:, 0]
    x_range = x.max() - x.min()
    if x_range < 1e-10:
        x_range = 1.0

    # Candidate models
    # We will use a simple hold-out validation to select the best model.
    # This is faster and often more robust than K-Fold for small n.
    
    # Split data into train_val and test_val
    rng = np.random.default_rng(42)
    idx = rng.permutation(n)
    n_val = max(10, n // 10)  # 10% for validation
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]
    
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    candidates = []
    
    # RBF with various sigmas
    sigma_candidates = [0.05 * x_range, 0.1 * x_range, 0.2 * x_range, 0.3 * x_range, 0.5 * x_range]
    for sigma in sigma_candidates:
        candidates.append((f"RBF(s={sigma:.3f})", lambda X_t, y_t, s=sigma: _fit_rbf(X_t, y_t, s)))

    # Polynomials
    for deg in [2, 4, 6, 8, 12]:
        candidates.append((f"Poly(d={deg})", lambda X_t, y_t, d=deg: _fit_poly(X_t, y_t, d)))

    # Interpolating Spline
    candidates.append(("Spline", lambda X_t, y_t: _fit_spline_interp(X_t, y_t)))

    # Linear
    candidates.append(("Linear", lambda X_t, y_t: _fit_linear(X_t, y_t)))

    best_model = None
    best_val_err = np.inf
    best_name = "none"

    for name, fit_func in candidates:
        if _remaining() < 5:
            break
        try:
            predict_func = fit_func(X_tr, y_tr)
            preds = predict_func(X_val)
            preds = np.asarray(preds, dtype=np.float64).flatten()
            if not np.all(np.isfinite(preds)):
                continue
            val_err = np.mean((preds - y_val) ** 2)
            if val_err < best_val_err:
                best_val_err = val_err
                best_name = name
                # Store the fit function, we will re-fit on all data later
                best_fit_func = fit_func
        except Exception:
            pass

    # If no model was selected, use linear
    if best_model is None:
        best_fit_func = _fit_linear
        best_name = "Linear"

    # Re-fit the best model on ALL data
    try:
        final_predict = best_fit_func(X, y)
    except Exception:
        # Fallback
        c = np.mean(y)
        def final_predict(X_new):
            return np.full(len(X_new), c)

    def safe_predict(X_new):
        try:
            r = final_predict(np.asarray(X_new, dtype=np.float64))
            r = np.asarray(r, dtype=np.float64).flatten()
            r = np.nan_to_num(r, nan=0.0, posinf=1e6, neginf=-1e6)
            return r
        except Exception:
            return np.zeros(len(X_new))

    return safe_predict
# EVOLVE-BLOCK-END
