# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr

_START_TIME = time.time()
_DEADLINE = _START_TIME + 320  # safety margin before 330s kill


def _time_left():
    return _DEADLINE - time.time()


def _build_basis(X, d):
    """Build a rich set of candidate basis functions evaluated on X."""
    N = X.shape[0]
    cols = [np.ones(N)]
    names = ["1"]

    # Linear
    for j in range(d):
        cols.append(X[:, j])
        names.append(f"x{j}")

    # Squares
    for j in range(d):
        cols.append(X[:, j] ** 2)
        names.append(f"x{j}^2")

    # Cubics
    for j in range(d):
        cols.append(X[:, j] ** 3)
        names.append(f"x{j}^3")

    # Quartics
    for j in range(d):
        cols.append(X[:, j] ** 4)
        names.append(f"x{j}^4")

    # Quintics
    for j in range(d):
        cols.append(X[:, j] ** 5)
        names.append(f"x{j}^5")

    # Pairwise products
    for i in range(d):
        for j in range(i, d):
            cols.append(X[:, i] * X[:, j])
            names.append(f"x{i}*x{j}")

    # Triple products
    for i in range(d):
        for j in range(i, d):
            for k in range(j, d):
                cols.append(X[:, i] * X[:, j] * X[:, k])
                names.append(f"x{i}*x{j}*x{k}")

    # Quartic cross terms
    if d >= 2:
        for i in range(d):
            for j in range(i, d):
                for k in range(j, d):
                    for l in range(k, d):
                        if len({i, j, k, l}) == 4:
                            cols.append(X[:, i] * X[:, j] * X[:, k] * X[:, l])
                            names.append(f"x{i}*x{j}*x{k}*x{l}")

    # Trig functions
    for j in range(d):
        xj = X[:, j]
        cols.append(np.sin(xj))
        names.append(f"sin(x{j})")
        cols.append(np.cos(xj))
        names.append(f"cos(x{j})")
        cols.append(np.sin(2 * xj))
        names.append(f"sin(2x{j})")
        cols.append(np.cos(2 * xj))
        names.append(f"cos(2x{j})")
        cols.append(np.tanh(xj))
        names.append(f"tanh(x{j})")
        cols.append(np.tanh(2 * xj))
        names.append(f"tanh(2x{j})")

    # Exp and log
    for j in range(d):
        xj = X[:, j]
        cols.append(np.exp(xj))
        names.append(f"exp(x{j})")
        cols.append(np.exp(-xj))
        names.append(f"exp(-x{j})")
        cols.append(np.log1p(xj))
        names.append(f"log1p(x{j})")
        cols.append(np.sqrt(xj))
        names.append(f"sqrt(x{j})")
        cols.append(1.0 / (1.0 + np.exp(-xj)))  # sigmoid
        names.append(f"sigmoid(x{j})")

    # Cross terms with trig
    if d >= 2:
        for i in range(d):
            for j in range(i + 1, d):
                cols.append(np.sin(X[:, i]) * X[:, j])
                names.append(f"sin(x{i})*x{j}")
                cols.append(np.cos(X[:, i]) * X[:, j])
                names.append(f"cos(x{i})*x{j}")
                cols.append(np.sin(X[:, i]) * np.sin(X[:, j]))
                names.append(f"sin(x{i})*sin(x{j})")
                cols.append(np.cos(X[:, i]) * np.cos(X[:, j]))
                names.append(f"cos(x{i})*cos(x{j})")
                cols.append(X[:, i] * np.exp(X[:, j]))
                names.append(f"x{i}*exp(x{j})")
                cols.append(np.exp(X[:, i]) * X[:, j])
                names.append(f"exp(x{i})*x{j}")
                cols.append(X[:, i] * np.sqrt(X[:, j]))
                names.append(f"x{i}*sqrt(x{j})")
                cols.append(np.sqrt(X[:, i]) * X[:, j])
                names.append(f"sqrt(x{i})*x{j}")

    # Sum and product of all inputs
    cols.append(np.sum(X, axis=1))
    names.append("sum(x)")
    cols.append(np.prod(X, axis=1))
    names.append("prod(x)")
    cols.append(np.sum(X ** 2, axis=1))
    names.append("sum(x^2)")
    cols.append(np.sum(X ** 3, axis=1))
    names.append("sum(x^3)")

    # Distance from origin
    cols.append(np.linalg.norm(X, axis=1))
    names.append("norm(x)")
    cols.append(np.linalg.norm(X, axis=1) ** 2)
    names.append("norm(x)^2")

    # Min and max
    cols.append(np.min(X, axis=1))
    names.append("min(x)")
    cols.append(np.max(X, axis=1))
    names.append("max(x)")

    # Differences
    if d >= 2:
        for i in range(d):
            for j in range(i + 1, d):
                cols.append(X[:, i] - X[:, j])
                names.append(f"x{i}-x{j}")
                cols.append(X[:, i] * (X[:, i] - X[:, j]))
                names.append(f"x{i}*(x{i}-x{j})")
                cols.append((X[:, i] - X[:, j]) ** 2)
                names.append(f"(x{i}-x{j})^2")

    # Additional trig combinations
    if d >= 2:
        for i in range(d):
            for j in range(i + 1, d):
                cols.append(np.sin(X[:, i] + X[:, j]))
                names.append(f"sin(x{i}+x{j})")
                cols.append(np.cos(X[:, i] + X[:, j]))
                names.append(f"cos(x{i}+x{j})")
                cols.append(np.sin(X[:, i] - X[:, j]))
                names.append(f"sin(x{i}-x{j})")
                cols.append(np.cos(X[:, i] - X[:, j]))
                names.append(f"cos(x{i}-x{j})")

    return np.column_stack(cols), names


def _fit_lsq(F, y, lam=1e-8):
    """Fit y = F @ w using ridge regression."""
    n, p = F.shape
    if p == 0:
        return np.zeros(0)
    A = F.T @ F + lam * np.eye(p)
    b = F.T @ y
    try:
        w = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(F, y, rcond=None)[0]
    return w


def _omp_select(F, y, max_features=15):
    """Orthogonal Matching Pursuit for feature selection."""
    n, p = F.shape
    if p <= max_features:
        return list(range(p))

    residual = y.copy()
    selected = [0]  # start with constant
    F_sel = F[:, selected]
    w = _fit_lsq(F_sel, y)
    pred = F_sel @ w
    residual = y - pred

    for _ in range(max_features - 1):
        if _time_left() < 3.0:
            break

        # Find feature most correlated with residual
        corr = np.abs(F.T @ residual)
        corr[selected] = 0
        best_idx = np.argmax(corr)

        if corr[best_idx] < 1e-10:
            break

        selected.append(best_idx)
        F_sel = F[:, selected]
        w = _fit_lsq(F_sel, y)
        pred = F_sel @ w
        residual = y - pred

        if np.linalg.norm(residual) < 1e-12:
            break

    return selected


def _forward_select(F, y, initial_selected, max_features=20, time_limit=15.0):
    """Forward selection with greedy additions."""
    n, p = F.shape
    selected = list(initial_selected)
    F_sel = F[:, selected]
    w = _fit_lsq(F_sel, y)
    pred = F_sel @ w
    best_mse = np.mean((pred - y) ** 2)

    start = time.time()
    improved = True
    while improved and len(selected) < max_features and (time.time() - start) < time_limit:
        improved = False
        best_idx = -1
        best_score = best_mse

        residual = y - pred
        candidates = [j for j in range(p) if j not in selected]
        if not candidates:
            break

        corr = np.abs(F[:, candidates].T @ residual)
        top_candidates = [candidates[i] for i in np.argsort(corr)[::-1][:min(30, len(candidates))]]

        for j in top_candidates:
            trial = selected + [j]
            F_trial = F[:, trial]
            w_trial = _fit_lsq(F_trial, y)
            pred_trial = F_trial @ w_trial
            mse = np.mean((pred_trial - y) ** 2)
            if mse < best_score * 0.995:
                best_score = mse
                best_idx = j
                improved = True

        if best_idx == -1:
            break

        selected.append(best_idx)
        F_sel = F[:, selected]
        w = _fit_lsq(F_sel, y)
        pred = F_sel @ w
        best_mse = np.mean((pred - y) ** 2)

    return selected, best_mse


def _swap_search(F, y, selected, max_swaps=5, time_limit=10.0):
    """Try swapping features to improve fit."""
    n, p = F.shape
    F_sel = F[:, selected]
    w = _fit_lsq(F_sel, y)
    pred = F_sel @ w
    best_mse = np.mean((pred - y) ** 2)

    start = time.time()
    for _ in range(max_swaps):
        if _time_left() < 3.0 or (time.time() - start) > time_limit:
            break

        improved = False
        residual = y - pred
        for i in range(1, len(selected)):  # keep constant
            candidates = [j for j in range(p) if j not in selected]
            if not candidates:
                break
            corr = np.abs(F[:, candidates].T @ residual)
            top_cands = [candidates[k] for k in np.argsort(corr)[::-1][:min(15, len(candidates))]]

            for j in top_cands:
                trial = selected.copy()
                trial[i] = j
                F_trial = F[:, trial]
                w_trial = _fit_lsq(F_trial, y)
                pred_trial = F_trial @ w_trial
                mse = np.mean((pred_trial - y) ** 2)
                if mse < best_mse * 0.99:
                    selected = trial
                    F_sel = F_trial
                    w = w_trial
                    pred = pred_trial
                    best_mse = mse
                    improved = True
                    break
            if improved:
                break

        if not improved:
            break

    return selected, best_mse


def fit(X_train, y_train):
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    N, d = X_train.shape

    if N == 0:
        return lambda X: np.zeros(len(X))

    # Build rich basis
    F, names = _build_basis(X_train, d)

    # Remove non-finite columns
    finite_cols = np.all(np.isfinite(F), axis=0)
    F = F[:, finite_cols]
    p = F.shape[1]

    if p == 0:
        return lambda X: np.zeros(len(X))

    # Time budget allocation
    time_budget = min(25.0, max(5.0, _time_left() / 10.0))
    max_feat = min(p, 25)

    # Step 1: OMP for initial selection
    selected_omp = _omp_select(F, y_train, max_features=min(max_feat, 15))
    F_sel = F[:, selected_omp]
    w = _fit_lsq(F_sel, y_train)
    pred = F_sel @ w
    best_mse = np.mean((pred - y_train) ** 2)
    best_selected = selected_omp

    # Step 2: Forward selection to expand
    selected_fwd, mse_fwd = _forward_select(F, y_train, selected_omp, max_features=max_feat, time_limit=time_budget * 0.5)
    if mse_fwd < best_mse:
        best_mse = mse_fwd
        best_selected = selected_fwd

    # Step 3: Swap search
    selected_swap, mse_swap = _swap_search(F, y_train, best_selected, max_swaps=3, time_limit=time_budget * 0.3)
    if mse_swap < best_mse:
        best_mse = mse_swap
        best_selected = selected_swap

    # Final fit
    F_sel = F[:, best_selected]
    w = _fit_lsq(F_sel, y_train)
    pred_train = F_sel @ w
    train_mse = np.mean((pred_train - y_train) ** 2)

    # Check if fit is good
    y_var = np.var(y_train)
    if y_var > 0:
        normalized_mse = train_mse / y_var
    else:
        normalized_mse = train_mse

    # If fit is poor, try simpler models as fallback
    if normalized_mse > 0.1 and d <= 3:
        # Try linear model
        F_lin = np.column_stack([np.ones(N)] + [X_train[:, j] for j in range(d)])
        w_lin = _fit_lsq(F_lin, y_train)
        mse_lin = np.mean((F_lin @ w_lin - y_train) ** 2)
        if mse_lin < train_mse:
            best_selected = list(range(F_lin.shape[1]))
            F_sel = F_lin
            w = w_lin
            train_mse = mse_lin

    # Capture for prediction
    sel_indices = best_selected

    def predict(X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        F_pred, _ = _build_basis(X, d)
        F_pred = F_pred[:, finite_cols]
        F_pred_sel = F_pred[:, sel_indices]
        return F_pred_sel @ w

    return predict
# EVOLVE-BLOCK-END
