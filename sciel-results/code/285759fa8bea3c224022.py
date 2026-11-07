"""Symbolic regression via forward-selection over a rich primitive basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 320.0  # safety margin before 330s kill


def _remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)


def _build_basis(X, d):
    """Build a rich basis of candidate primitives for 1-D (or general d) input.
    Returns (basis_matrix, names). Each column is a candidate feature.
    """
    n = X.shape[0]
    cols = [np.ones(n, dtype=float)]
    names = ['1']
    for j in range(d):
        x = X[:, j]
        # Polynomials
        for p in range(1, 8):
            cols.append(x ** p)
            names.append(f'x{j}^{p}')
        # Trig
        cols.append(np.sin(x))
        names.append(f'sin(x{j})')
        cols.append(np.cos(x))
        names.append(f'cos(x{j})')
        cols.append(np.tan(x))
        names.append(f'tan(x{j})')
        # Exp / log
        cols.append(np.exp(x))
        names.append(f'exp(x{j})')
        cols.append(np.exp(-x))
        names.append(f'exp(-x{j})')
        cols.append(np.exp(x ** 2))
        names.append(f'exp(x{j}^2)')
        cols.append(np.exp(-(x ** 2)))
        names.append(f'exp(-x{j}^2)')
        safe_x = np.where(np.abs(x) < 1e-8, np.sign(x) * 1e-8 + (x >= 0) * 1e-8, x)
        cols.append(np.log(np.abs(safe_x)))
        names.append(f'log(|x{j}|)')
        # Rational / reciprocal
        cols.append(1.0 / safe_x)
        names.append(f'1/x{j}')
        cols.append(1.0 / (1.0 + x))
        names.append(f'1/(1+x{j})')
        cols.append(1.0 / (1.0 + x ** 2))
        names.append(f'1/(1+x{j}^2)')
        cols.append(x / (1.0 + x))
        names.append(f'x{j}/(1+x{j})')
        cols.append(x / (1.0 + x ** 2))
        names.append(f'x{j}/(1+x{j}^2)')
        cols.append(1.0 / (x - 1.0) + 1e-8 * (x >= 1.0))
        names.append(f'1/(x{j}-1)')
        cols.append(1.0 / (x + 1.0) + 1e-8 * (x <= -1.0))
        names.append(f'1/(x{j}+1)')
        # tanh, sqrt
        cols.append(np.tanh(x))
        names.append(f'tanh(x{j})')
        cols.append(np.sqrt(np.abs(x)))
        names.append(f'sqrt(|x{j}|)')
    # Cross terms for d>1 (keep modest)
    if d >= 2:
        for i in range(d):
            for j in range(i + 1, d):
                xi = X[:, i]
                xj = X[:, j]
                cols.append(xi * xj)
                names.append(f'x{i}*x{j}')
                cols.append(np.sin(xi) * np.cos(xj))
                names.append(f'sin(x{i})*cos(x{j})')
                cols.append(np.exp(xi) * xj)
                names.append(f'exp(x{i})*x{j}')
    B = np.column_stack(cols)
    # Remove non-finite columns
    mask = np.all(np.isfinite(B), axis=0)
    B = B[:, mask]
    names = [nm for nm, m in zip(names, mask) if m]
    return B, names


def _ridge_solve(B, y, lam):
    """Solve (B^T B + lam I) w = B^T y. Returns w or None if singular."""
    try:
        G = B.T @ B + lam * np.eye(B.shape[1])
        rhs = B.T @ y
        w = np.linalg.solve(G, rhs)
        if not np.all(np.isfinite(w)):
            return None
        return w
    except np.linalg.LinAlgError:
        return None


def _forward_select(B, y, max_terms, max_time, lam_start=1e-6, lam_growth=10.0):
    """Forward selection: greedily add the basis column that most reduces
    regularized residual sum of squares, with early stopping on validation-like
    improvement. Returns (selected_indices, weights)."""
    n, m = B.shape
    y_mean = y.mean()
    best_ssr = np.sum((y - y_mean) ** 2)
    selected = []
    remaining = list(range(m))
    lam = lam_start
    deadline = time.time() + max_time

    # Precompute B^T y and B^T B lazily per step for speed via incremental
    # We'll just do a full re-solve each step (m is modest, ~60-100)
    while selected and remaining and len(selected) < max_terms and time.time() < deadline:
        # Current SSR with selected features
        B_sel = B[:, selected]
        w = _ridge_solve(B_sel, y, lam)
        if w is None:
            break
        pred = B_sel @ w
        ssr = np.sum((y - pred) ** 2)
        # Try adding each remaining feature
        best_gain = 1e-12
        best_feat = None
        for f in remaining:
            cand = selected + [f]
            B_c = B[:, cand]
            wc = _ridge_solve(B_c, y, lam)
            if wc is None:
                continue
            predc = B_c @ wc
            ssrc = np.sum((y - predc) ** 2)
            gain = ssr - ssrc
            if gain > best_gain:
                best_gain = gain
                best_feat = f
        if best_feat is None:
            break
        selected.append(best_feat)
        remaining.remove(best_feat)
        # Increase lambda slightly to encourage sparsity as we add terms
        lam *= 1.2
        if len(selected) >= max_terms:
            break

    # Final solve
    if not selected:
        # Fallback: constant
        return [], np.array([y.mean()])
    B_sel = B[:, selected]
    w = _ridge_solve(B_sel, y, lam)
    if w is None:
        return [], np.array([y.mean()])
    return selected, w


def fit(X_train, y_train):
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    d = X_train.shape[1]
    n = X_train.shape[0]

    # Build basis
    B, names = _build_basis(X_train, d)
    m = B.shape[1]

    # Normalize y for numerical stability
    y_mean = y_train.mean()
    y_std = y_train.std()
    if y_std < 1e-12:
        # Constant target
        c = y_mean
        return lambda X: np.full(X.shape[0], c, dtype=float)
    y_norm = (y_train - y_mean) / y_std

    # Time budget for forward selection
    time_budget = min(120.0, max(10.0, _remaining() * 0.5))
    max_terms = min(m, 12)

    selected, w = _forward_select(B, y_norm, max_terms, time_budget)

    # Build prediction function
    def predict(X):
        X = np.asarray(X, dtype=float)
        Bx, _ = _build_basis(X, d)
        if not selected:
            return np.full(X.shape[0], y_mean, dtype=float)
        Bx_sel = Bx[:, selected]
        pred_norm = Bx_sel @ w
        pred = pred_norm * y_std + y_mean
        # Ensure finite
        pred = np.where(np.isfinite(pred), pred, y_mean)
        return pred

    # Quick sanity check on training data
    pred_check = predict(X_train)
    if not np.all(np.isfinite(pred_check)):
        # Fallback to linear
        try:
            A = np.column_stack([np.ones(n), X_train])
            wl, *_ = np.linalg.lstsq(A, y_train, rcond=None)
            def predict_lin(X):
                X = np.asarray(X, dtype=float)
                Al = np.column_stack([np.ones(X.shape[0]), X])
                p = Al @ wl
                return np.where(np.isfinite(p), p, y_mean)
            return predict_lin
        except Exception:
            return lambda X: np.full(np.asarray(X).shape[0], y_mean, dtype=float)

    return predict
# EVOLVE-BLOCK-END
