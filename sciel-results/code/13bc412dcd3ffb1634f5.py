"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


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


def _cv_mse(x, y, deg, lam, n_folds=5, seed=42):
    """K-fold CV MSE for a given degree and regularization."""
    n = len(x)
    if n < 2 * n_folds:
        n_folds = max(2, n // 2)
    if n_folds < 2:
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

    # Focused candidate set: degree 5 with varying regularization, plus lower degrees
    # The physics suggests the unresolved forcing has nonlinear structure but is
    # smooth, so degree 5 with moderate regularization should capture it well.
    candidates = [
        (5, 0.0),
        (5, 1e-5),
        (5, 1e-4),
        (5, 1e-3),
        (5, 1e-2),
        (5, 0.1),
        (4, 0.0),
        (4, 1e-3),
        (4, 1e-2),
        (3, 0.0),
        (3, 1e-2),
        (2, 0.0),
    ]

    for deg, lam in candidates:
        if time.time() > deadline - 5.0:
            break

        score = _cv_mse(x, y, deg, lam, n_folds=5, seed=42)

        if score < best_score:
            best_score = score
            c = _ridge_fit(x, y, deg, lam)
            if c is not None:
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
