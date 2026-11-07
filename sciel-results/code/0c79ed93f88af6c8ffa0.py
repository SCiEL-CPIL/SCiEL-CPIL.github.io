"""Structural change: monotone cubic (PCHIP) interpolation with a polynomial fallback,
plus a time-budgeted symbolic forward-selection to capture non-linear primitives.
The PCHIP path is exact for noiseless 1-D data and avoids polynomial oscillation.
The symbolic path adds sin, cos, exp, log, x^n terms via greedy forward selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize

# Global deadline guard: 300s total across all 4 equations.
# We track a soft deadline per fit call; the evaluator runs 4 fits sequentially.
# To be safe, we assume each fit gets ~75s, but we also guard globally.
_START_TIME = time.time()
# We'll use a per-call budget of 65s to leave margin.
PER_FIT_BUDGET = 65.0


def _pchip_fit(X_train, y_train):
    """Fit a PCHIP interpolant on the 1-D training data."""
    x = X_train[:, 0].copy()
    y = y_train.copy()
    # Sort by x
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    # Remove duplicate x values (keep first occurrence)
    unique_mask = np.concatenate([True, x_sorted[1:] != x_sorted[:-1]])
    x_unique = x_sorted[unique_mask]
    y_unique = y_sorted[unique_mask]
    if len(x_unique) < 2:
        return lambda X: np.full(X.shape[0], y_unique[0])
    pchip = PchipInterpolator(x_unique, y_unique)
    return lambda X: pchip(X[:, 0])


def _symbolic_forward_selection(X_train, y_train, time_budget=20.0):
    """
    Greedy forward selection over a rich basis to capture non-linear structures.
    Basis includes: 1, x, x^2, x^3, x^4, sin(x), cos(x), exp(x), exp(-x), log(|x|+1),
    and products like x*sin(x), x*exp(-x), etc.
    """
    start = time.time()
    x = X_train[:, 0].copy()
    y = y_train.copy()
    n = len(x)

    # Build candidate basis functions
    def basis_funcs():
        funcs = []
        names = []
        # Polynomials
        for p in range(0, 6):
            if p == 0:
                funcs.append(np.ones_like(x))
                names.append("1")
            else:
                funcs.append(x ** p)
                names.append(f"x^{p}")
        # Trig
        funcs.append(np.sin(x)); names.append("sin(x)")
        funcs.append(np.cos(x)); names.append("cos(x)")
        funcs.append(np.sin(2 * x)); names.append("sin(2x)")
        funcs.append(np.cos(2 * x)); names.append("cos(2x)")
        # Exponential
        funcs.append(np.exp(x)); names.append("exp(x)")
        funcs.append(np.exp(-x)); names.append("exp(-x)")
        funcs.append(np.exp(-x ** 2)); names.append("exp(-x^2)")
        # Log
        funcs.append(np.log(np.abs(x) + 1)); names.append("log(|x|+1)")
        # Products
        funcs.append(x * np.sin(x)); names.append("x*sin(x)")
        funcs.append(x * np.cos(x)); names.append("x*cos(x)")
        funcs.append(x * np.exp(-x)); names.append("x*exp(-x)")
        funcs.append(x * np.exp(-x ** 2)); names.append("x*exp(-x^2)")
        funcs.append(x ** 2 * np.exp(-x)); names.append("x^2*exp(-x)")
        funcs.append(np.sin(x) * np.exp(-x)); names.append("sin(x)*exp(-x)")
        return funcs, names

    funcs, names = basis_funcs()
    n_features = len(funcs)

    # Start with intercept
    selected = [0]  # index 0 is "1"
    X_design = np.column_stack([funcs[0]])
    # Fit initial
    w, *_ = np.linalg.lstsq(X_design, y, rcond=None)
    pred = X_design @ w
    sse = np.sum((y - pred) ** 2)
    var_y = np.var(y)
    if var_y == 0:
        var_y = 1.0
    nrmse = sse / var_y

    # Forward selection
    max_terms = 8
    while len(selected) < max_terms:
        if time.time() - start > time_budget:
            break
        best_idx = -1
        best_sse = sse
        for i in range(n_features):
            if i in selected:
                continue
            new_selected = selected + [i]
            X_new = np.column_stack([funcs[j] for j in new_selected])
            try:
                w_new, *_ = np.linalg.lstsq(X_new, y, rcond=None)
                pred_new = X_new @ w_new
                sse_new = np.sum((y - pred_new) ** 2)
                if sse_new < best_sse:
                    best_sse = sse_new
                    best_idx = i
            except np.linalg.LinAlgError:
                continue
        if best_idx == -1 or best_sse >= sse * 0.999:
            break
        selected.append(best_idx)
        X_design = np.column_stack([funcs[j] for j in selected])
        w, *_ = np.linalg.lstsq(X_design, y, rcond=None)
        pred = X_design @ w
        sse = best_sse
        nrmse = sse / var_y
        if nrmse < 1e-10:
            break

    # Build prediction function
    def make_predict(selected_idx, funcs, X_train_x, y_train, w):
        # Pre-compute the basis at training points for the selected terms
        # For prediction, we need to evaluate the same basis at new X
        def predict(X):
            x_new = X[:, 0]
            cols = []
            for idx in selected_idx:
                # Re-evaluate the basis function at x_new
                # We need to replicate the logic
                name = names[idx]
                if name == "1":
                    cols.append(np.ones_like(x_new))
                elif name.startswith("x^"):
                    p = int(name[2:])
                    cols.append(x_new ** p)
                elif name == "sin(x)":
                    cols.append(np.sin(x_new))
                elif name == "cos(x)":
                    cols.append(np.cos(x_new))
                elif name == "sin(2x)":
                    cols.append(np.sin(2 * x_new))
                elif name == "cos(2x)":
                    cols.append(np.cos(2 * x_new))
                elif name == "exp(x)":
                    cols.append(np.exp(x_new))
                elif name == "exp(-x)":
                    cols.append(np.exp(-x_new))
                elif name == "exp(-x^2)":
                    cols.append(np.exp(-x_new ** 2))
                elif name == "log(|x|+1)":
                    cols.append(np.log(np.abs(x_new) + 1))
                elif name == "x*sin(x)":
                    cols.append(x_new * np.sin(x_new))
                elif name == "x*cos(x)":
                    cols.append(x_new * np.cos(x_new))
                elif name == "x*exp(-x)":
                    cols.append(x_new * np.exp(-x_new))
                elif name == "x*exp(-x^2)":
                    cols.append(x_new * np.exp(-x_new ** 2))
                elif name == "x^2*exp(-x)":
                    cols.append(x_new ** 2 * np.exp(-x_new))
                elif name == "sin(x)*exp(-x)":
                    cols.append(np.sin(x_new) * np.exp(-x_new))
                else:
                    cols.append(np.ones_like(x_new))
            X_new_design = np.column_stack(cols)
            return X_new_design @ w
        return predict

    return make_predict(selected, funcs, x, y, w), nrmse


def fit(X_train, y_train):
    t_start = time.time()
    deadline = t_start + PER_FIT_BUDGET

    # 1. Try PCHIP first (fast, exact for noiseless 1-D)
    try:
        pchip_pred = _pchip_fit(X_train, y_train)
        # Evaluate PCHIP on training data to get its NRMSE
        pred_train = pchip_pred(X_train)
        var_y = np.var(y_train)
        if var_y == 0:
            var_y = 1.0
        pchip_nrmse = np.sum((y_train - pred_train) ** 2) / var_y
    except Exception:
        pchip_pred = None
        pchip_nrmse = np.inf

    # 2. Try symbolic forward selection
    sym_pred = None
    sym_nrmse = np.inf
    if time.time() < deadline - 5:
        try:
            sym_pred, sym_nrmse = _symbolic_forward_selection(
                X_train, y_train, time_budget=min(20.0, max(0, deadline - time.time() - 5))
            )
        except Exception:
            sym_pred = None
            sym_nrmse = np.inf

    # 3. Choose the best model
    if pchip_pred is not None and sym_pred is not None:
        if pchip_nrmse <= sym_nrmse:
            return pchip_pred
        else:
            return sym_pred
    elif pchip_pred is not None:
        return pchip_pred
    elif sym_pred is not None:
        return sym_pred
    else:
        # Fallback: constant predictor
        mean_y = np.mean(y_train)
        return lambda X: np.full(X.shape[0], mean_y)
# EVOLVE-BLOCK-END
