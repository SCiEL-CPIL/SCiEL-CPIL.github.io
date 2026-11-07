"""Symbolic regression over 10 hidden equations: Enhanced basis + adaptive greedy selection."""
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

def _generate_basis(X):
    """
    Generate a comprehensive, diverse feature basis for symbolic regression.
    X: (N, D)
    Returns: (N, M) feature matrix
    """
    N, D = X.shape
    cols = [np.ones(N)]
    
    # 1. Individual powers
    for j in range(D):
        x = X[:, j]
        cols.append(x)
        cols.append(x**2)
        cols.append(x**3)
        cols.append(x**4)  # Add 4th power for more flexibility

    # 2. Trigonometric functions of individuals
    for j in range(D):
        x = X[:, j]
        cols.append(np.sin(x))
        cols.append(np.cos(x))
        cols.append(np.sin(2*x))
        cols.append(np.cos(2*x))
        cols.append(np.sin(3*x))
        cols.append(np.cos(3*x))
        cols.append(np.tanh(x))
        # Exponentials (clipped for stability)
        x_clip = np.clip(x, -10, 10)
        cols.append(np.exp(x_clip))
        cols.append(np.exp(-x_clip))

    # 3. Pairwise interactions
    for i in range(D):
        for j in range(i + 1, D):
            xi = X[:, i]
            xj = X[:, j]
            # Products
            cols.append(xi * xj)
            cols.append(xi**2 * xj)
            cols.append(xi * xj**2)
            # Sums and differences
            s = xi + xj
            d = xi - xj
            cols.append(s)
            cols.append(d)
            cols.append(s**2)
            cols.append(d**2)
            # Trig of sums/diffs
            cols.append(np.sin(s))
            cols.append(np.cos(s))
            cols.append(np.sin(d))
            cols.append(np.cos(d))
            cols.append(np.sin(2*s))
            cols.append(np.cos(2*s))
            # Rational
            cols.append(_safe_div(xi, xj))
            cols.append(_safe_div(xj, xi))
            cols.append(_safe_div(1.0, xi))
            cols.append(_safe_div(1.0, xj))
            # Product of trig
            cols.append(np.sin(xi) * np.cos(xj))
            cols.append(np.cos(xi) * np.sin(xj))
            cols.append(np.sin(xi) * np.sin(xj))
            cols.append(np.cos(xi) * np.cos(xj))

    # 4. Global features
    if D > 1:
        s_all = np.sum(X, axis=1)
        cols.append(s_all)
        cols.append(s_all**2)
        cols.append(s_all**3)
        cols.append(np.sin(s_all))
        cols.append(np.cos(s_all))
        cols.append(np.sin(2*s_all))
        cols.append(np.cos(2*s_all))
        cols.append(np.tanh(s_all))
        
        norm = np.sqrt(np.sum(X**2, axis=1))
        cols.append(norm)
        cols.append(norm**2)
        cols.append(np.sin(norm))
        cols.append(np.cos(norm))
        cols.append(np.tanh(norm))
        
        # Product of all
        p_all = np.ones(N)
        for j in range(D):
            p_all = p_all * X[:, j]
        cols.append(p_all)
        cols.append(p_all**2)
        cols.append(np.sin(p_all))
        cols.append(np.cos(p_all))

    # 5. Additional combinations for higher-dimensional inputs
    if D >= 3:
        for i in range(D):
            for j in range(i+1, D):
                for k in range(j+1, D):
                    xi = X[:, i]
                    xj = X[:, j]
                    xk = X[:, k]
                    cols.append(xi * xj * xk)
                    cols.append(xi + xj + xk)
                    cols.append(np.sin(xi + xj + xk))
                    cols.append(np.cos(xi + xj + xk))

    # Stack and clean
    A = np.column_stack(cols)
    # Remove non-finite columns
    mask = np.isfinite(A).all(axis=0)
    A = A[:, mask]
    
    # Clip extreme values to prevent numerical issues in dot products
    A = np.clip(A, -1e6, 1e6)
    
    return A

def fit(X_train, y_train):
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y = y.reshape(-1)
    N, D = X.shape
    
    # Emergency fallback
    if _remaining() < 5.0:
        # Just use linear + bias
        A_lin = np.column_stack([np.ones(N), X])
        try:
            w, *_ = np.linalg.lstsq(A_lin, y, rcond=None)
            def predict(X):
                X = np.asarray(X, dtype=float)
                if X.ndim == 1:
                    X = X.reshape(-1, 1)
                return np.column_stack([np.ones(len(X)), X]) @ w
            return predict
        except Exception:
            mean_y = np.mean(y)
            def predict(X):
                return np.ones(len(X)) * mean_y
            return predict

    # Generate Basis
    A = _generate_basis(X)
    M = A.shape[1]
    
    if M == 0:
        mean_y = np.mean(y)
        def predict(X):
            return np.ones(len(X)) * mean_y
        return predict

    # Greedy Forward Selection with adaptive stopping
    r = y.copy()
    selected = []
    
    best_w = None
    best_err = np.inf
    
    max_steps = min(30, M)  # Allow more steps for better fit
    
    A_sel = None
    y_sel = y
    
    for step in range(max_steps):
        if _remaining() < 2.0:
            break
            
        best_rss = np.inf
        best_j = -1
        best_w_candidate = None
        
        # If no features selected yet, find best single feature
        if A_sel is None:
            for j in range(M):
                if _remaining() < 1.0:
                    break
                a_j = A[:, j].reshape(-1, 1)
                try:
                    w, *_ = np.linalg.lstsq(a_j, y, rcond=None)
                    pred = a_j @ w
                    rss = np.sum((y - pred)**2)
                    if rss < best_rss:
                        best_rss = rss
                        best_j = j
                        best_w_candidate = w
                except:
                    pass
            
            if best_j == -1:
                break
                
            # Add best feature
            A_sel = A[:, best_j].reshape(-1, 1)
            selected.append(best_j)
            best_w = best_w_candidate
            best_err = best_rss
            
            if best_err < 1e-8:  # Stricter tolerance for noiseless data
                break
            continue
        
        # If we have selected features, try adding each remaining one
        for j in range(M):
            if j in selected:
                continue
            
            if _remaining() < 1.0:
                break
                
            A_new = np.hstack([A_sel, A[:, j].reshape(-1, 1)])
            try:
                w_new, *_ = np.linalg.lstsq(A_new, y, rcond=None)
                pred = A_new @ w_new
                rss = np.sum((y - pred)**2)
                if rss < best_rss:
                    best_rss = rss
                    best_j = j
                    best_w_candidate = w_new
            except:
                pass
        
        if best_j == -1:
            break
        
        # Check if improvement is significant
        if best_err - best_rss < 1e-10 * max(1.0, best_err):
            break
            
        # Update state
        A_sel = np.hstack([A_sel, A[:, best_j].reshape(-1, 1)])
        selected.append(best_j)
        best_w = best_w_candidate
        best_err = best_rss
        
        if best_err < 1e-8:
            break

    # If we didn't find a perfect fit, best_w is from the last step.
    if best_w is None:
        # Fallback: Just use the first feature or mean
        mean_y = np.mean(y)
        def predict(X):
            return np.ones(len(X)) * mean_y
        return predict

    # Create predict function
    def predict(X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Generate basis for new X
        A_new = _generate_basis(X)
        
        if len(selected) == 0:
            return np.ones(len(X)) * np.mean(y)
            
        A_sel_new = A_new[:, selected]
        
        try:
            result = A_sel_new @ best_w
            # Ensure finite values
            if not np.all(np.isfinite(result)):
                result = np.where(np.isfinite(result), result, np.mean(y))
            return result
        except Exception:
            return np.ones(len(X)) * np.mean(y)

    return predict
# EVOLVE-BLOCK-END
