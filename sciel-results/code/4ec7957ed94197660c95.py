"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _fit_l1(x, y, deg, lam, max_iter=2000, tol=1e-10):
    """Fit polynomial using L1 (lasso) regularization via proximal gradient.
    Returns 6-coeff array padded to length 6, or None on failure."""
    n = len(x)
    if n < deg + 1:
        return None
    
    V = np.polynomial.polynomial.polyvander(x, deg)
    # Normalize columns to help with convergence
    col_norms = np.sqrt(np.sum(V ** 2, axis=0))
    col_norms[col_norms < 1e-12] = 1.0
    V_norm = V / col_norms
    
    # Initial guess: least squares
    try:
        c0 = np.linalg.lstsq(V_norm, y, rcond=None)[0]
    except Exception:
        return None
    
    c = c0.copy()
    
    # Step size based on Lipschitz constant
    # L = max eigenvalue of V_norm.T @ V_norm
    try:
        # Approximate Lipschitz constant
        L = np.max(np.abs(np.linalg.eigvalsh(V_norm.T @ V_norm / n)))
        if L < 1e-12:
            L = 1.0
        step = 1.0 / (L * 1.5)
    except Exception:
        step = 1e-3
    
    for _ in range(max_iter):
        # Gradient step
        grad = V_norm.T @ (V_norm @ c - y) / n
        c_new = c - step * grad
        
        # Proximal operator for L1: soft thresholding
        for i in range(deg + 1):
            if i == 0:
                # Don't regularize the constant term too heavily, or use same lam
                c_new[i] = np.sign(c_new[i]) * max(abs(c_new[i]) - step * lam, 0.0)
            else:
                c_new[i] = np.sign(c_new[i]) * max(abs(c_new[i]) - step * lam, 0.0)
        
        # Check convergence
        if np.max(np.abs(c_new - c)) < tol:
            c = c_new
            break
        c = c_new
    
    # Transform back to original scale
    c_orig = c * col_norms
    
    if not np.all(np.isfinite(c_orig)):
        return None
    
    out = np.zeros(6)
    out[:deg + 1] = c_orig
    return out


def _ridge_fit(x, y, deg, lam):
    """Fit polynomial with ridge regularization, return 6-coeff array."""
    if lam <= 0.0:
        try:
            c = np.polynomial.polynomial.polyfit(x, y, deg)
            if not np.all(np.isfinite(c)):
                return None
            out = np.zeros(6)
            out[:deg + 1] = c
            return out
        except Exception:
            return None

    V = np.polynomial.polynomial.polyvander(x, deg)
    # Regularization: penalize higher-order terms more strongly
    reg_weights = np.ones(deg + 1)
    for i in range(1, deg + 1):
        reg_weights[i] = lam * (i + 1) ** 2
    R = np.diag(reg_weights)
    try:
        A = V.T @ V + R
        b = V.T @ y
        c = np.linalg.solve(A, b)
        if not np.all(np.isfinite(c)):
            return None
        out = np.zeros(6)
        out[:deg + 1] = c
        return out
    except Exception:
        return None


def _cv_mse(x, y, deg, lam, n_folds=5, seed=42, use_l1=False, l1_lam=0.0):
    """K-fold CV MSE for a given degree and regularization."""
    n = len(x)
    if n < 2 * n_folds:
        n_folds = max(2, n // 2)
    if n_folds < 2:
        if use_l1:
            c = _fit_l1(x, y, deg, l1_lam)
        else:
            c = _ridge_fit(x, y, deg, lam)
        if c is None:
            return np.inf
        pred = np.polynomial.polynomial.polyval(x, c)
        return np.mean((pred - y) ** 2)

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    fold_size = n // n_folds
    total_mse = 0.0
    valid_folds = 0

    for i in range(n_folds):
        start = i * fold_size
        end = start + fold_size if i < n_folds - 1 else n
        val_idx = indices[start:end]
        train_idx = np.concatenate([indices[:start], indices[end:]])

        if len(train_idx) < deg + 1:
            continue

        if use_l1:
            c = _fit_l1(x[train_idx], y[train_idx], deg, l1_lam)
        else:
            c = _ridge_fit(x[train_idx], y[train_idx], deg, lam)
        if c is None:
            continue

        pred = np.polynomial.polynomial.polyval(x[val_idx], c)
        total_mse += np.mean((pred - y[val_idx]) ** 2)
        valid_folds += 1

    if valid_folds == 0:
        return np.inf
    return total_mse / valid_folds


def construct(training_x, training_subgrid_tendency):
    t0 = time.time()
    deadline = t0 + 115.0  # safety margin before 120s limit

    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)

    mask = np.isfinite(x) & np.isfinite(y)
    if not np.all(mask):
        x = x[mask]
        y = y[mask]

    if len(x) < 6:
        return np.zeros(6)

    best_coeff = None
    best_score = np.inf

    # Candidate set: ridge with various regularization, and L1 for sparsity
    candidates = [
        # (type, deg, lam, l1_lam)
        ('ridge', 5, 0.0, 0.0),
        ('ridge', 5, 1e-4, 0.0),
        ('ridge', 5, 1e-3, 0.0),
        ('ridge', 5, 1e-2, 0.0),
        ('ridge', 4, 1e-3, 0.0),
        ('ridge', 4, 1e-2, 0.0),
        ('ridge', 3, 1e-2, 0.0),
        # L1 candidates for sparse feature selection
        ('l1', 5, 0.0, 1e-4),
        ('l1', 5, 0.0, 1e-3),
        ('l1', 5, 0.0, 1e-2),
        ('l1', 4, 0.0, 1e-3),
        ('l1', 4, 0.0, 1e-2),
    ]

    for ctype, deg, lam, l1_lam in candidates:
        if time.time() > deadline - 5.0:
            break

        if ctype == 'l1':
            score = _cv_mse(x, y, deg, lam, n_folds=5, seed=42, use_l1=True, l1_lam=l1_lam)
        else:
            score = _cv_mse(x, y, deg, lam, n_folds=5, seed=42, use_l1=False, l1_lam=0.0)

        if score < best_score:
            best_score = score
            if ctype == 'l1':
                c = _fit_l1(x, y, deg, l1_lam)
            else:
                c = _ridge_fit(x, y, deg, lam)
            if c is not None and np.all(np.isfinite(c)):
                best_coeff = c

    if best_coeff is not None and np.all(np.isfinite(best_coeff)):
        return best_coeff

    # Fallback: unregularized degree 5
    try:
        c = np.polynomial.polynomial.polyfit(x, y, 5)
        if np.all(np.isfinite(c)):
            out = np.zeros(6)
            out[:6] = c
            return out
    except Exception:
        pass

    # Fallback: degree 3
    try:
        c = np.polynomial.polynomial.polyfit(x, y, 3)
        if np.all(np.isfinite(c)):
            out = np.zeros(6)
            out[:4] = c
            return out
    except Exception:
        pass

    # Fallback: linear
    try:
        c = np.polynomial.polynomial.polyfit(x, y, 1)
        if np.all(np.isfinite(c)):
            out = np.zeros(6)
            out[:2] = c
            return out
    except Exception:
        pass

    return np.zeros(6)
# EVOLVE-BLOCK-END
