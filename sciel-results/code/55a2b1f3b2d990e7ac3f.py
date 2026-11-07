# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 315  # safety margin before 330s kill

def _time_left():
    return _DEADLINE - time.time()

def _safe_exp(x):
    return np.exp(np.clip(x, -20, 20))

def _safe_log(x):
    return np.log(np.clip(x, 1e-8, None))

def _safe_div(a, b):
    return a / np.clip(b, 1e-8, None)

def _safe_sqrt(x):
    return np.sqrt(np.clip(x, 1e-8, None))

def _build_feature_lambdas(d):
    """
    Generate a list of (name, lambda) pairs for basis functions.
    d is the dimension of the input X.
    """
    fns = []
    
    # Constant
    fns.append(("1", lambda X: np.ones(X.shape[0])))
    
    # Linear, Quadratic, Cubic, Quartic
    powers = [1, 2, 3, 4]
    for p in powers:
        for j in range(d):
            fns.append((f"x{j}^{p}", lambda X, j=j, p=p: X[:, j] ** p))
            
    # Square Root
    for j in range(d):
        fns.append((f"sqrt(x{j})", lambda X, j=j: _safe_sqrt(X[:, j])))
        
    # Logarithm
    for j in range(d):
        fns.append((f"log(x{j})", lambda X, j=j: _safe_log(X[:, j])))
        
    # Exponential
    for j in range(d):
        fns.append((f"exp(x{j})", lambda X, j=j: _safe_exp(X[:, j])))
        
    # Reciprocal
    for j in range(d):
        fns.append((f"1/x{j}", lambda X, j=j: 1.0 / np.clip(X[:, j], 1e-8, None)))
        
    # Trigonometric
    for j in range(d):
        fns.append((f"sin(x{j})", lambda X, j=j: np.sin(X[:, j])))
        fns.append((f"cos(x{j})", lambda X, j=j: np.cos(X[:, j])))
        fns.append((f"tanh(x{j})", lambda X, j=j: np.tanh(X[:, j])))
        
    # Absolute value
    for j in range(d):
        fns.append((f"abs(x{j})", lambda X, j=j: np.abs(X[:, j])))

    # Pairwise products (polynomial)
    for i in range(d):
        for j in range(i, d):
            fns.append((f"x{i}*x{j}", lambda X, i=i, j=j: X[:, i] * X[:, j]))
            
    # Pairwise sums and differences
    for i in range(d):
        for j in range(i+1, d):
            fns.append((f"x{i}+x{j}", lambda X, i=i, j=j: X[:, i] + X[:, j]))
            fns.append((f"x{i}-x{j}", lambda X, i=i, j=j: X[:, i] - X[:, j]))
            fns.append((f"x{i}/x{j}", lambda X, i=i, j=j: _safe_div(X[:, i], X[:, j])))
            
    # Cross terms with trig/exp/log
    for i in range(d):
        for j in range(i+1, d):
            fns.append((f"x{i}*sin(x{j})", lambda X, i=i, j=j: X[:, i] * np.sin(X[:, j])))
            fns.append((f"x{i}*cos(x{j})", lambda X, i=i, j=j: X[:, i] * np.cos(X[:, j])))
            fns.append((f"x{i}*exp(x{j})", lambda X, i=i, j=j: X[:, i] * _safe_exp(X[:, j])))
            fns.append((f"x{i}*log(x{j})", lambda X, i=i, j=j: X[:, i] * _safe_log(X[:, j])))
            fns.append((f"sin(x{i})*sin(x{j})", lambda X, i=i, j=j: np.sin(X[:, i]) * np.sin(X[:, j])))
            fns.append((f"cos(x{i})*cos(x{j})", lambda X, i=i, j=j: np.cos(X[:, i]) * np.cos(X[:, j])))
            
    # Triple products (only for d>=3, limit combinations to avoid explosion)
    if d >= 3:
        for i in range(d):
            for j in range(i+1, d):
                for k in range(j+1, d):
                    fns.append((f"x{i}*x{j}*x{k}", lambda X, i=i, j=j, k=k: X[:, i] * X[:, j] * X[:, k]))
    
    # Norms
    fns.append(("norm2", lambda X: np.linalg.norm(X, axis=1)))
    fns.append(("sum_sq", lambda X: np.sum(X**2, axis=1)))

    return fns

def _fit_lsq(B, y):
    """
    Robust least squares fit.
    Returns coefficients or None if failed.
    """
    if B.size == 0:
        return None
    try:
        # Add small ridge for stability if needed
        # We try standard lstsq first
        w, residuals, rank, s = np.linalg.lstsq(B, y, rcond=None)
        if not np.all(np.isfinite(w)):
            return None
        return w
    except np.linalg.LinAlgError:
        # Fallback to ridge
        try:
            lam = 1e-6
            A = B.T @ B + lam * np.eye(B.shape[1])
            b = B.T @ y
            w = np.linalg.solve(A, b)
            if not np.all(np.isfinite(w)):
                return None
            return w
        except Exception:
            return None

def fit(X_train, y_train):
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    N, d = X_train.shape
    
    if N == 0:
        return lambda X: np.zeros(len(X))
        
    # Generate feature lambdas
    feature_lambdas = _build_feature_lambdas(d)
    
    # Evaluate all features on training data
    B_cols = []
    valid_indices = []
    for idx, (name, fn) in enumerate(feature_lambdas):
        try:
            v = fn(X_train)
            v = np.asarray(v, dtype=np.float64)
            # Check for non-finite values
            if not np.all(np.isfinite(v)):
                # Try to repair by clipping or replacing
                v = np.where(np.isfinite(v), v, 0.0)
            # Check magnitude to avoid instability
            if np.max(np.abs(v)) > 1e6:
                continue
            B_cols.append(v)
            valid_indices.append(idx)
        except Exception:
            continue
            
    if not B_cols:
        # Fallback to mean
        mean_y = np.mean(y_train)
        return lambda X: np.full(len(X), mean_y)
        
    B = np.column_stack(B_cols)
    p = B.shape[1]
    
    # Greedy Forward Selection
    # Start with constant (index 0 in B_cols corresponds to "1")
    # We need to map valid_indices back to B columns
    # B column 0 is the constant because it was the first valid feature (usually)
    # Let's ensure constant is first.
    # feature_lambdas[0] is "1". If it's valid, it's B[:,0].
    
    selected_cols = [0]
    current_B = B[:, [0]]
    current_w = _fit_lsq(current_B, y_train)
    if current_w is None:
        current_w = np.array([np.mean(y_train)])
        current_B = np.ones((N, 1))
        selected_cols = [0]
        
    current_pred = current_B @ current_w
    current_mse = np.mean((current_pred - y_train) ** 2)
    y_var = np.var(y_train)
    if y_var == 0:
        y_var = 1e-9
    
    best_mse = current_mse
    best_selected = selected_cols.copy()
    best_w = current_w.copy()
    
    # Time budget for selection
    # We have ~300s total. Let's allow up to 280s for this, but stop early if perfect.
    start_time = time.time()
    max_steps = 50 # Safety limit on number of features
    
    for step in range(max_steps):
        if time.time() - start_time > 280:
            break
        if len(selected_cols) >= p:
            break
            
        best_add_idx = -1
        best_add_mse = current_mse
        
        # Try adding each unselected column
        unselected = [c for c in range(p) if c not in selected_cols]
        
        for c in unselected:
            new_sel = selected_cols + [c]
            sub_B = B[:, new_sel]
            w = _fit_lsq(sub_B, y_train)
            if w is None:
                continue
            pred = sub_B @ w
            mse = np.mean((pred - y_train) ** 2)
            
            if mse < best_add_mse - 1e-12:
                best_add_mse = mse
                best_add_idx = c
                
        if best_add_idx == -1:
            break
            
        selected_cols.append(best_add_idx)
        current_B = B[:, selected_cols]
        current_w = _fit_lsq(current_B, y_train)
        if current_w is None:
            # Revert if fit fails
            selected_cols.pop()
            current_B = B[:, selected_cols]
            current_w = _fit_lsq(current_B, y_train)
            break
        current_pred = current_B @ current_w
        current_mse = np.mean((current_pred - y_train) ** 2)
        
        if current_mse < best_mse:
            best_mse = current_mse
            best_selected = selected_cols.copy()
            best_w = current_w.copy()
            
        # Stop if fit is excellent
        if current_mse < 1e-10 * y_var:
            break
            
    # Use the best model found
    final_selected = best_selected
    final_w = best_w
    
    # Get the corresponding lambdas for prediction
    # final_selected contains indices into B (which are indices into valid_indices)
    # We need to map back to feature_lambdas
    # B column i corresponds to valid_indices[i]
    # So final_selected[i] is the column index in B.
    # The lambda index is valid_indices[final_selected[i]]
    
    final_lambdas = []
    for col_idx in final_selected:
        lam_idx = valid_indices[col_idx]
        final_lambdas.append(feature_lambdas[lam_idx])
        
    def predict(X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        cols = []
        for name, fn in final_lambdas:
            try:
                v = fn(X)
                v = np.asarray(v, dtype=np.float64)
                if not np.all(np.isfinite(v)):
                    v = np.where(np.isfinite(v), v, 0.0)
                cols.append(v)
            except Exception:
                cols.append(np.zeros(X.shape[0]))
                
        if not cols:
            return np.full(X.shape[0], np.mean(y_train))
            
        F_pred = np.column_stack(cols)
        # Ensure shape matches
        if F_pred.shape[1] != len(final_w):
            # This shouldn't happen if we are careful, but just in case
            return np.full(X.shape[0], np.mean(y_train))
            
        preds = F_pred @ final_w
        return preds

    return predict
# EVOLVE-BLOCK-END
