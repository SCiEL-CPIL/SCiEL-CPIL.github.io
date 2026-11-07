"""Symbolic regression over 10 hidden equations: Greedy forward selection on rich basis."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_START = time.time()
_DEADLINE = 295.0  # Safety margin vs 330s kill

def _remaining():
    return _DEADLINE - (time.time() - _START)

def _safe_div(a, b, eps=1e-8):
    """Element-wise division with safe denominator."""
    return a / np.where(np.abs(b) < eps, np.sign(b) * eps + (b == 0) * eps, b)

def _generate_candidates(X):
    """Generate a rich pool of candidate features from input X."""
    n, d = X.shape
    candidates = []
    names = []
    
    # Intercept
    candidates.append(np.ones(n))
    names.append('const')
    
    # Linear terms
    for j in range(d):
        candidates.append(X[:, j].copy())
        names.append(f'x{j}')
    
    # Polynomial terms up to degree 4
    for j in range(d):
        for k in range(2, 5):
            candidates.append(X[:, j] ** k)
            names.append(f'x{j}^{k}')
    
    # Pairwise products (all pairs, including same)
    for i in range(d):
        for j in range(i, d):
            candidates.append(X[:, i] * X[:, j])
            names.append(f'x{i}*x{j}')
    
    # Pairwise sums and differences
    for i in range(d):
        for j in range(i + 1, d):
            candidates.append(X[:, i] + X[:, j])
            names.append(f'x{i}+x{j}')
            candidates.append(X[:, i] - X[:, j])
            names.append(f'x{i}-x{j}')
            candidates.append(X[:, j] - X[:, i])
            names.append(f'x{j}-x{i}')
    
    # Trigonometric functions of individual inputs
    for j in range(d):
        for freq in [1, 2, 3]:
            candidates.append(np.sin(freq * X[:, j]))
            names.append(f'sin({freq}*x{j})')
            candidates.append(np.cos(freq * X[:, j]))
            names.append(f'cos({freq}*x{j})')
    
    # Trigonometric functions of pairwise sums/differences
    for i in range(d):
        for j in range(i + 1, d):
            s = X[:, i] + X[:, j]
            diff = X[:, i] - X[:, j]
            for freq in [1, 2]:
                candidates.append(np.sin(freq * s))
                names.append(f'sin({freq}*(x{i}+x{j}))')
                candidates.append(np.cos(freq * s))
                names.append(f'cos({freq}*(x{i}+x{j}))')
                candidates.append(np.sin(freq * diff))
                names.append(f'sin({freq}*(x{i}-x{j}))')
                candidates.append(np.cos(freq * diff))
                names.append(f'cos({freq}*(x{i}-x{j}))')
    
    # Exponential functions (clipped)
    for j in range(d):
        x_clip = np.clip(X[:, j], -10, 10)
        candidates.append(np.exp(x_clip))
        names.append(f'exp(x{j})')
        candidates.append(np.exp(-x_clip))
        names.append(f'exp(-x{j})')
    
    # Sigmoid / tanh
    for j in range(d):
        candidates.append(np.tanh(X[:, j]))
        names.append(f'tanh(x{j})')
        x_clip = np.clip(X[:, j], -50, 50)
        candidates.append(1.0 / (1.0 + np.exp(-x_clip)))
        names.append(f'sigmoid(x{j})')
    
    # Rational functions (safe)
    for i in range(d):
        for j in range(i + 1, d):
            candidates.append(_safe_div(X[:, i], X[:, j]))
            names.append(f'x{i}/x{j}')
            candidates.append(_safe_div(X[:, j], X[:, i]))
            names.append(f'x{j}/x{i}')
        candidates.append(_safe_div(1.0, X[:, j]))
        names.append(f'1/x{j}')
    
    # Product of all inputs
    if d >= 1:
        p = np.ones(n)
        for j in range(d):
            p = p * X[:, j]
        candidates.append(p)
        names.append('prod_all')
    
    # Sum of all inputs
    s = np.sum(X, axis=1)
    candidates.append(s)
    names.append('sum_all')
    candidates.append(s ** 2)
    names.append('sum_all^2')
    candidates.append(np.sin(s))
    names.append('sin(sum_all)')
    candidates.append(np.cos(s))
    names.append('cos(sum_all)')
    
    # Norm
    norm = np.sqrt(np.sum(X ** 2, axis=1))
    candidates.append(norm)
    names.append('norm')
    candidates.append(np.sin(norm))
    names.append('sin(norm)')
    candidates.append(np.cos(norm))
    names.append('cos(norm)')
    
    # Stack and filter non-finite columns
    A = np.column_stack(candidates)
    mask = np.isfinite(A)
    A_clean = np.where(mask, A, 0.0)
    col_valid = mask.all(axis=0)
    A_clean = A_clean[:, col_valid]
    names_valid = [names[i] for i in range(len(names)) if col_valid[i]]
    
    return A_clean, names_valid

def _ols_fit(A, y):
    """Ordinary least squares fit."""
    try:
        w, residuals, rank, sv = np.linalg.lstsq(A, y, rcond=None)
        pred = A @ w
        if not np.all(np.isfinite(pred)):
            return None, float('inf')
        mse = np.mean((pred - y) ** 2)
        return w, mse
    except Exception:
        return None, float('inf')

def _greedy_forward_select(A, y, max_features=15, min_improvement=1e-6):
    """
    Greedy forward selection: start with intercept, add features one by one.
    Returns list of selected column indices and final MSE.
    """
    n, p = A.shape
    if p == 0:
        return [], float('inf')
    
    # Start with just the intercept (column 0)
    selected = [0]
    current_mse = np.var(y)  # MSE of predicting the mean
    
    for step in range(1, min(max_features, p)):
        if _remaining() < 5.0:
            break
        
        best_idx = None
        best_mse = current_mse
        best_w = None
        
        # Try adding each unselected feature
        for j in range(p):
            if j in selected:
                continue
            
            new_selected = selected + [j]
            A_sub = A[:, new_selected]
            w, mse = _ols_fit(A_sub, y)
            
            if w is not None and mse < best_mse:
                best_mse = mse
                best_idx = j
                best_w = w
        
        if best_idx is None or (current_mse - best_mse) < min_improvement * current_mse:
            break
        
        selected.append(best_idx)
        current_mse = best_mse
    
    return selected, current_mse

def _make_predict(A_train, selected_indices, w):
    """Create a predict function that uses only the selected features."""
    def predict(X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        A, _ = _generate_candidates(X)
        # Extract only the selected features
        A_sub = A[:, selected_indices]
        return A_sub @ w
    return predict

def fit(X_train, y_train):
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y = y.reshape(-1)
    
    if _remaining() < 10.0:
        # Emergency fallback: simple linear
        A_lin = np.column_stack([np.ones(len(X)), X])
        w, *_ = np.linalg.lstsq(A_lin, y, rcond=None)
        def predict(X):
            X = np.asarray(X, dtype=float)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            return np.column_stack([np.ones(len(X)), X]) @ w
        return predict
    
    # Generate candidate features
    A, names = _generate_candidates(X)
    
    if A.shape[1] == 0:
        # Fallback: constant
        w = np.array([np.mean(y)])
        def predict(X):
            return np.ones(len(X)) * w[0]
        return predict
    
    # Greedy forward selection
    selected, final_mse = _greedy_forward_select(A, y, max_features=15, min_improvement=1e-6)
    
    if not selected:
        # Fallback: use intercept only
        w = np.array([np.mean(y)])
        def predict(X):
            return np.ones(len(X)) * w[0]
        return predict
    
    # Final fit on selected features
    A_sub = A[:, selected]
    w, mse = _ols_fit(A_sub, y)
    
    if w is None:
        # Fallback: simple linear
        A_lin = np.column_stack([np.ones(len(X)), X])
        w, *_ = np.linalg.lstsq(A_lin, y, rcond=None)
        def predict(X):
            X = np.asarray(X, dtype=float)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            return np.column_stack([np.ones(len(X)), X]) @ w
        return predict
    
    return _make_predict(A, selected, w)
# EVOLVE-BLOCK-END
