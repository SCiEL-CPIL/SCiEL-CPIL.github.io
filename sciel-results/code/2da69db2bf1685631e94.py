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
        # cols.append(x**4) # Might be too many if D is large, but D<=3 usually

    # 2. Trigonometric functions of individuals
    for j in range(D):
        x = X[:, j]
        cols.append(np.sin(x))
        cols.append(np.cos(x))
        cols.append(np.sin(2*x))
        cols.append(np.cos(2*x))
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
            # Sums and differences
            s = xi + xj
            d = xi - xj
            cols.append(s)
            cols.append(d)
            # Trig of sums/diffs
            cols.append(np.sin(s))
            cols.append(np.cos(s))
            cols.append(np.sin(d))
            cols.append(np.cos(d))
            # Rational
            cols.append(_safe_div(xi, xj))
            cols.append(_safe_div(xj, xi))
            # Product of trig
            cols.append(np.sin(xi) * np.cos(xj))
            cols.append(np.cos(xi) * np.sin(xj))

    # 4. Global features
    if D > 1:
        s_all = np.sum(X, axis=1)
        cols.append(s_all)
        cols.append(s_all**2)
        cols.append(np.sin(s_all))
        cols.append(np.cos(s_all))
        
        norm = np.sqrt(np.sum(X**2, axis=1))
        cols.append(norm)
        cols.append(np.sin(norm))
        cols.append(np.cos(norm))
        
        # Product of all
        p_all = np.ones(N)
        for j in range(D):
            p_all = p_all * X[:, j]
        cols.append(p_all)

    # Stack and clean
    A = np.column_stack(cols)
    # Remove non-finite columns
    mask = np.isfinite(A).all(axis=0)
    A = A[:, mask]
    
    # Standardize? No, for symbolic regression, raw scale is often better for exact fit, 
    # but for numerical stability in least squares, scaling helps. 
    # However, since we are doing greedy selection, we want features that correlate well with y.
    # Let's keep them raw but ensure no huge values.
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

    # Greedy Forward Selection
    # We want to find a subset of columns in A that best predicts y.
    # Since data is noiseless, we hope to find a subset where RSS ~ 0.
    
    # Pre-compute A.T @ A and A.T @ y? 
    # M is likely < 100. N=200. 
    # A.T @ A is MxM. A.T @ y is Mx1.
    # We can update the residual efficiently.
    
    # Initial residual
    r = y.copy()
    # We track selected features indices
    selected = []
    # We need to solve the linear system for the selected features.
    # Instead of solving from scratch each time, we can use the normal equations.
    # Let A_sel be the matrix of selected features.
    # We maintain A_sel.T @ A_sel and A_sel.T @ y.
    
    # To save time, let's just iterate and use np.linalg.lstsq on the small A_sel.
    # With N=200 and small k (selected features), this is fast.
    
    best_w = None
    best_err = np.inf
    
    # Pre-compute A.T @ y to speed up correlation checks?
    # Actually, for greedy selection, at each step k, we want to find j that minimizes RSS after adding j.
    # RSS_new = RSS_old - (A_j - P_A A_j)^T (A_j - P_A A_j)
    # This is equivalent to finding the feature with the highest correlation with the current residual, 
    # adjusted for the fact that we are adding it to the existing model.
    # A simpler, robust approach: 
    # 1. Start with empty set.
    # 2. For each step:
    #    a. For each candidate feature j not in selected:
    #       - Form A_temp = [A_sel, A_j]
    #       - Solve lstsq(A_temp, y)
    #       - Calculate RSS
    #    b. Pick j with min RSS.
    #    c. If RSS < threshold, stop.
    
    # Optimization: Use the fact that we only add 1 feature at a time.
    # We can pre-compute the projection of each feature onto the current subspace.
    
    # Let's try a simpler loop first, as M is small.
    # Max steps: 20 or until time runs out.
    
    max_steps = min(20, M)
    
    A_sel = None
    y_sel = y
    
    for step in range(max_steps):
        if _remaining() < 2.0:
            break
            
        best_rss = np.inf
        best_j = -1
        best_w_candidate = None
        
        # If no features selected yet, we can just look at correlation with y
        if A_sel is None:
            # Find feature with max correlation with y
            # corr(j) = |A_j^T y| / (||A_j|| ||y||)
            # Since we will normalize in lstsq, we can just look at A_j^T y
            # Actually, let's just test all features individually.
            # This is M lstsq calls of size (N, 1) vs y. Very fast.
            for j in range(M):
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
            
            if best_err < 1e-6:
                break
            continue
            
        # If we have selected features, try adding each remaining one
        # We can speed this up by using the residual from the current model.
        # But the "best next feature" in OLS is not simply the one with highest correlation with residual 
        # unless we account for the change in coefficients of existing features.
        # However, for a greedy approach, checking all candidates is safest.
        
        # Optimization: 
        # We can compute the projection of each candidate A_j onto the column space of A_sel.
        # Let P = A_sel (A_sel^T A_sel)^-1 A_sel^T
        # The reduction in RSS by adding A_j is || (I-P) A_j ||^2 * (correlation with residual projected)
        
        # Given N=200 and M<100, and steps<20, brute force M * Steps lstsq calls is:
        # 100 * 20 = 2000 lstsq calls. Each is (200, k) vs y.
        # This might be slow in Python if not vectorized.
        
        # Let's try a vectorized approach for the selection step.
        # We want to minimize || y - A_new w_new ||^2
        # This is equivalent to maximizing the squared norm of the projection of y onto the new subspace.
        
        # Alternative: Use the residual r from the current model.
        # The feature that best reduces RSS is the one that is most correlated with r, 
        # AFTER orthogonalizing r with respect to the new feature? No.
        
        # Let's stick to the efficient update formula for Greedy OLS (LARS-like but for exact fit).
        # 1. Compute residual r = y - A_sel @ w_current
        # 2. For each candidate j, compute the correlation c_j = A_j^T r
        # 3. The reduction in RSS is not just c_j^2 / ||A_j||^2 because w_current changes.
        
        # Given the small size, let's just do the brute force but with a time check inside the loop.
        # We can parallelize the "try all j" part? No, standard lib.
        
        # Let's optimize the brute force by pre-computing A_sel^T A_sel inverse etc?
        # Actually, let's just use np.linalg.lstsq on the augmented matrix.
        
        # To speed up, we can sample a subset of candidates if M is large? 
        # But M is likely < 50.
        
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
            
        # Update state
        A_sel = np.hstack([A_sel, A[:, best_j].reshape(-1, 1)])
        selected.append(best_j)
        best_w = best_w_candidate
        best_err = best_rss
        
        if best_err < 1e-6:
            break

    # If we didn't find a perfect fit, best_w is from the last step.
    # If we broke early due to time, best_w is from the last completed step.
    
    if best_w is None:
        # Fallback: Just use the first feature or mean
        mean_y = np.mean(y)
        def predict(X):
            return np.ones(len(X)) * mean_y
        return predict

    # Create predict function
    # We need to map the selected features back to the new X.
    # We stored the indices in 'selected'.
    
    def predict(X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Generate basis for new X
        A_new = _generate_basis(X)
        
        # Extract selected columns
        # Note: _generate_basis must produce columns in the same order.
        # Since it's deterministic, it should.
        
        if len(selected) == 0:
            return np.ones(len(X)) * np.mean(y)
            
        A_sel_new = A_new[:, selected]
        
        # Solve for weights? No, we have best_w from training.
        # But wait, best_w corresponds to the order of columns in A_sel during training.
        # A_sel was built by hstacking columns in the order they were selected.
        # So A_sel_new should be built the same way.
        
        try:
            return A_sel_new @ best_w
        except Exception:
            # Fallback if dimensions mismatch (should not happen)
            return np.ones(len(X)) * np.mean(y)

    return predict
# EVOLVE-BLOCK-END
