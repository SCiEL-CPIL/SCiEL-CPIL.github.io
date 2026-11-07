"""Symbolic regression with time-bounded feature search."""
import numpy as np
import time
import sys

_TOTAL_DEADLINE = 310.0  # safety margin under 330s wall-clock

def _time_left():
    return _TOTAL_DEADLINE - time.time()

def _make_predictor(X, w):
    """Return a callable that applies feature matrix @ w."""
    def predict(X_test):
        X_test = np.asarray(X_test, dtype=np.float64)
        if X_test.ndim == 1:
            X_test = X_test.reshape(1, -1)
        try:
            out = _build_features(X_test) @ w
            out = np.atleast_1d(out)
            # Ensure finite
            out = np.where(np.isfinite(out), out, 0.0)
            return out
        except Exception:
            return np.zeros(X_test.shape[0])
    return predict

def _build_features(X, mode=0):
    """
    Build feature matrix based on mode.
    mode 0: constant + linear
    mode 1: + quadratic
    mode 2: + cubic (squares of squares, x_i*x_j*x_k)
    mode 3: + trig (sin, cos of each input)
    mode 4: + exp, log, tanh, sqrt, abs
    mode 5: + rational (1/x_i), x_i/x_j
    mode 6: + interaction trig (sin(x_i)*x_j), exp(x_i)*x_j
    mode 7: + higher trig (sin(2x), cos(2x), sin(x_i+x_j))
    """
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    cols = [np.ones(n)]
    
    # Linear
    for j in range(d):
        cols.append(X[:, j])
    
    if mode >= 1:
        # Quadratic
        for i in range(d):
            for j in range(i, d):
                cols.append(X[:, i] * X[:, j])
    
    if mode >= 2:
        # Cubic
        for i in range(d):
            for j in range(i, d):
                for k in range(j, d):
                    cols.append(X[:, i] * X[:, j] * X[:, k])
    
    if mode >= 3:
        # Trig
        for j in range(d):
            cols.append(np.sin(X[:, j]))
            cols.append(np.cos(X[:, j]))
    
    if mode >= 4:
        # Exp, log, tanh, sqrt, abs
        for j in range(d):
            cols.append(np.exp(-X[:, j] / max(1.0, X[:, j].max() + 1e-9)))  # scaled exp to avoid overflow
            cols.append(np.log1p(X[:, j]))
            cols.append(np.tanh(X[:, j]))
            cols.append(np.sqrt(X[:, j]))
            cols.append(np.abs(X[:, j]))
    
    if mode >= 5:
        # Rational
        for j in range(d):
            safe_x = np.where(np.abs(X[:, j]) > 1e-8, X[:, j], 1e-8)
            cols.append(1.0 / safe_x)
        for i in range(d):
            for j in range(i+1, d):
                safe_xj = np.where(np.abs(X[:, j]) > 1e-8, X[:, j], 1e-8)
                cols.append(X[:, i] / safe_xj)
    
    if mode >= 6:
        # Interaction trig/exp
        for j in range(d):
            cols.append(np.sin(X[:, j]) * X[:, j])
            cols.append(np.cos(X[:, j]) * X[:, j])
            scaled = X[:, j] / max(1.0, X[:, j].max() + 1e-9)
            cols.append(np.exp(scaled) * X[:, j])
        for i in range(d):
            for j in range(i+1, d):
                cols.append(np.sin(X[:, i]) * X[:, j])
                cols.append(np.cos(X[:, i]) * X[:, j])
    
    if mode >= 7:
        # Higher trig
        for j in range(d):
            cols.append(np.sin(2 * X[:, j]))
            cols.append(np.cos(2 * X[:, j]))
        for i in range(d):
            for j in range(i+1, d):
                cols.append(np.sin(X[:, i] + X[:, j]))
                cols.append(np.cos(X[:, i] + X[:, j]))
    
    return np.column_stack(cols)

def _ridge_fit(Xf, y, alpha):
    """Ridge regression: (Xf^T Xf + alpha*I) w = Xf^T y"""
    n, p = Xf.shape
    A = Xf.T @ Xf + alpha * np.eye(p)
    b = Xf.T @ y
    try:
        w = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(A, b, rcond=None)[0]
    return w

def _normalized_mse(pred, y):
    """Compute normalized MSE: mean((pred-y)^2) / var(y)"""
    if len(y) < 2:
        return 1.0
    var_y = np.var(y)
    if var_y < 1e-12:
        # Constant target
        mse = np.mean((pred - y)**2)
        return mse / max(1e-12, np.mean(y**2) + 1e-12)
    return np.mean((pred - y)**2) / var_y

def fit(X_train, y_train):
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64).ravel()
    n, d = X_train.shape
    
    # Split for validation: 80% train, 20% val
    np.random.seed(42)
    idx = np.random.permutation(n)
    n_val = max(1, int(0.2 * n))
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]
    
    X_tr = X_train[tr_idx]
    y_tr = y_train[tr_idx]
    X_val = X_train[val_idx]
    y_val = y_train[val_idx]
    
    # Try progressively more complex models
    best_w = None
    best_score = float('inf')
    best_mode = 0
    
    # Start with simple models, increase complexity
    max_mode = 7
    # Allocate time per mode based on remaining budget
    # Try modes 0 through max_mode
    
    for mode in range(max_mode + 1):
        if _time_left() < 15.0:
            break  # Not enough time for more
        
        try:
            Xf_tr = _build_features(X_tr, mode)
            Xf_val = _build_features(X_val, mode)
        except Exception:
            continue
        
        # Use ridge with alpha scaled to feature count
        p = Xf_tr.shape[1]
        # Try a few alpha values
        alphas = [1e-6, 1e-4, 1e-2, 1.0, 10.0]
        # If many features, use larger alphas
        if p > 50:
            alphas = [1e-4, 1e-2, 1.0, 10.0, 100.0]
        
        for alpha in alphas:
            if _time_left() < 10.0:
                break
            try:
                w = _ridge_fit(Xf_tr, y_tr, alpha)
                pred_val = Xf_val @ w
                score = _normalized_mse(pred_val, y_val)
                if score < best_score and np.isfinite(score):
                    best_score = score
                    best_w = w.copy()
                    best_mode = mode
            except Exception:
                continue
        
        # Early stop if we've achieved very good fit
        if best_score < 1e-8:
            break
    
    # If no model worked, fall back to constant prediction
    if best_w is None:
        # Just predict the mean
        mean_y = np.mean(y_train)
        def predict_const(X_test):
            X_test = np.asarray(X_test, dtype=np.float64)
            if X_test.ndim == 1:
                X_test = X_test.reshape(1, -1)
            return np.full(X_test.shape[0], mean_y)
        return predict_const
    
    # Final: refit on ALL training data with best mode and alpha
    # Re-find the best alpha for the best mode
    try:
        Xf_all = _build_features(X_train, best_mode)
        p = Xf_all.shape[1]
        alphas = [1e-6, 1e-4, 1e-2, 1.0, 10.0]
        if p > 50:
            alphas = [1e-4, 1e-2, 1.0, 10.0, 100.0]
        
        best_final_w = None
        best_final_val_score = float('inf')
        
        for alpha in alphas:
            if _time_left() < 5.0:
                break
            try:
                w = _ridge_fit(Xf_all, y_train, alpha)
                # Evaluate on validation set to pick alpha
                Xf_val = _build_features(X_val, best_mode)
                pred_val = Xf_val @ w
                score = _normalized_mse(pred_val, y_val)
                if score < best_final_val_score and np.isfinite(score):
                    best_final_val_score = score
                    best_final_w = w.copy()
            except Exception:
                continue
        
        if best_final_w is not None:
            best_w = best_final_w
    except Exception:
        pass
    
    # Create the predictor
    mode_used = best_mode
    w_final = best_w.copy()
    
    def predict(X_test):
        X_test = np.asarray(X_test, dtype=np.float64)
        if X_test.ndim == 1:
            X_test = X_test.reshape(1, -1)
        try:
            Xf = _build_features(X_test, mode_used)
            out = Xf @ w_final
            out = np.atleast_1d(out)
            out = np.where(np.isfinite(out), out, 0.0)
            return out
        except Exception:
            return np.zeros(X_test.shape[0])
    
    return predict
