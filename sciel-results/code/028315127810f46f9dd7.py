# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings
from scipy.linalg import solve
from scipy.optimize import minimize

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 280.0  # safety margin for 330s kill

    n, d = X_train.shape
    y_mean = np.mean(y_train)
    y_var = np.var(y_train)
    if y_var < 1e-12:
        return lambda X: np.full(len(X), y_mean)

    def check_time():
        return time.time() < deadline

    # Feature normalization for numerical stability
    X_min = X_train.min(axis=0)
    X_max = X_train.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range < 1e-10] = 1.0
    Xn_train = (X_train - X_min) / X_range  # in [0,1]

    y_std = np.std(y_train)
    if y_std < 1e-10:
        y_std = 1.0
    yn = (y_train - y_mean) / y_std

    # Build a rich basis of candidate terms
    # Each term is a function of Xn (normalized to [0,1])
    # We want to cover: polynomials, trig, exp, log, sqrt, rational, interactions

    def build_basis(Xn, d, max_poly_deg=4, max_trig_freq=4):
        """Build basis matrix with many candidate terms."""
        terms = [np.ones(len(Xn))]  # constant
        term_descs = ["1"]

        # Univariate terms for each dimension
        for j in range(d):
            xj = Xn[:, j]
            # Polynomials
            for k in range(1, max_poly_deg + 1):
                terms.append(xj ** k)
                term_descs.append(f"x{j}^{k}")
            # Trig
            for k in range(1, max_trig_freq + 1):
                terms.append(np.sin(k * 2 * np.pi * xj))
                term_descs.append(f"sin({k}*2*pi*x{j})")
                terms.append(np.cos(k * 2 * np.pi * xj))
                term_descs.append(f"cos({k}*2*pi*x{j})")
            # Exponential
            terms.append(np.exp(xj))
            term_descs.append(f"exp(x{j})")
            terms.append(np.exp(-xj))
            term_descs.append(f"exp(-x{j})")
            # Log
            terms.append(np.log(xj + 1e-10))
            term_descs.append(f"log(x{j}+eps)")
            # Sqrt
            terms.append(np.sqrt(xj))
            term_descs.append(f"sqrt(x{j})")
            # Inverse
            terms.append(1.0 / (xj + 1e-10))
            term_descs.append(f"1/(x{j}+eps)")
            # Tanh
            terms.append(np.tanh(xj))
            term_descs.append(f"tanh(x{j})")
            # Arcsin/arccos (clamped)
            xc = np.clip(xj, 1e-8, 1 - 1e-8)
            terms.append(np.arcsin(xc))
            term_descs.append(f"arcsin(x{j})")
            terms.append(np.arccos(xc))
            term_descs.append(f"arccos(x{j})")
            # Higher order trig
            terms.append(np.sin(xj))
            term_descs.append(f"sin(x{j})")
            terms.append(np.cos(xj))
            term_descs.append(f"cos(x{j})")
            # x * sin, x * cos
            terms.append(xj * np.sin(xj))
            term_descs.append(f"x{j}*sin(x{j})")
            terms.append(xj * np.cos(xj))
            term_descs.append(f"x{j}*cos(x{j})")

        # Pairwise interactions for d <= 4
        if d >= 2:
            for i in range(d):
                for j in range(i + 1, d):
                    xi = Xn[:, i]
                    xj = Xn[:, j]
                    # Polynomial interactions
                    terms.append(xi * xj)
                    term_descs.append(f"x{i}*x{j}")
                    terms.append(xi * xj ** 2)
                    term_descs.append(f"x{i}*x{j}^2")
                    terms.append(xi ** 2 * xj)
                    term_descs.append(f"x{i}^2*x{j}")
                    terms.append(xi ** 2 * xj ** 2)
                    term_descs.append(f"x{i}^2*x{j}^2")
                    terms.append(xi ** 3 * xj)
                    term_descs.append(f"x{i}^3*x{j}")
                    terms.append(xi * xj ** 3)
                    term_descs.append(f"x{i}*x{j}^3")
                    terms.append(xi ** 3 * xj ** 3)
                    term_descs.append(f"x{i}^3*x{j}^3")
                    # Trig interactions
                    terms.append(np.sin(xi + xj))
                    term_descs.append(f"sin(x{i}+x{j})")
                    terms.append(np.cos(xi + xj))
                    term_descs.append(f"cos(x{i}+x{j})")
                    terms.append(np.sin(xi - xj))
                    term_descs.append(f"sin(x{i}-x{j})")
                    terms.append(np.cos(xi - xj))
                    term_descs.append(f"cos(x{i}-x{j})")
                    terms.append(np.sin(xi * xj))
                    term_descs.append(f"sin(x{i}*x{j})")
                    terms.append(np.cos(xi * xj))
                    term_descs.append(f"cos(x{i}*x{j})")
                    # Exp interactions
                    terms.append(np.exp(xi + xj))
                    term_descs.append(f"exp(x{i}+x{j})")
                    terms.append(np.exp(xi - xj))
                    term_descs.append(f"exp(x{i}-x{j})")
                    terms.append(np.exp(xi * xj))
                    term_descs.append(f"exp(x{i}*x{j})")
                    # Log interactions
                    terms.append(np.log(xi * xj + 1e-10))
                    term_descs.append(f"log(x{i}*x{j})")
                    # Sqrt interactions
                    terms.append(np.sqrt(xi * xj))
                    term_descs.append(f"sqrt(x{i}*x{j})")
                    # Ratio
                    terms.append(xi / (xj + 1e-10))
                    term_descs.append(f"x{i}/(x{j}+eps)")
                    terms.append(xj / (xi + 1e-10))
                    term_descs.append(f"x{j}/(x{i}+eps)")

        # Triple interactions for d == 3
        if d == 3:
            x0, x1, x2 = Xn[:, 0], Xn[:, 1], Xn[:, 2]
            terms.append(x0 * x1 * x2)
            term_descs.append("x0*x1*x2")
            terms.append(x0 ** 2 * x1 * x2)
            term_descs.append("x0^2*x1*x2")
            terms.append(x0 * x1 ** 2 * x2)
            term_descs.append("x0*x1^2*x2")
            terms.append(x0 * x1 * x2 ** 2)
            term_descs.append("x0*x1*x2^2")
            terms.append(x0 ** 2 * x1 ** 2 * x2)
            term_descs.append("x0^2*x1^2*x2")
            terms.append(x0 ** 2 * x1 * x2 ** 2)
            term_descs.append("x0^2*x1*x2^2")
            terms.append(x0 * x1 ** 2 * x2 ** 2)
            term_descs.append("x0*x1^2*x2^2")
            terms.append(x0 ** 2 * x1 ** 2 * x2 ** 2)
            term_descs.append("x0^2*x1^2*x2^2")
            terms.append(np.sin(x0 + x1 + x2))
            term_descs.append("sin(x0+x1+x2)")
            terms.append(np.cos(x0 + x1 + x2))
            term_descs.append("cos(x0+x1+x2)")
            terms.append(np.exp(x0 + x1 + x2))
            term_descs.append("exp(x0+x1+x2)")
            terms.append(np.exp(x0 * x1 * x2))
            term_descs.append("exp(x0*x1*x2)")
            terms.append(np.sqrt(x0 * x1 * x2))
            term_descs.append("sqrt(x0*x1*x2)")
            terms.append(x0 / (x1 + x2 + 1e-10))
            term_descs.append("x0/(x1+x2+eps)")
            terms.append(x1 / (x0 + x2 + 1e-10))
            term_descs.append("x1/(x0+x2+eps)")
            terms.append(x2 / (x0 + x1 + 1e-10))
            term_descs.append("x2/(x0+x1+eps)")

        F = np.column_stack(terms)
        # Remove columns that are all NaN/Inf
        col_ok = np.all(np.isfinite(F), axis=0)
        F = F[:, col_ok]
        term_descs = [t for t, ok in zip(term_descs, col_ok) if ok]
        return F, term_descs

    # Build the basis
    F_train, term_descs = build_basis(Xn_train, d)
    n_terms = F_train.shape[1]

    def safe_evaluate(model, X, y):
        try:
            preds = model(X)
            preds = np.asarray(preds, dtype=float)
            if preds.shape != y.shape:
                return np.inf
            if np.any(np.isnan(preds)) or np.any(np.isinf(preds)):
                return np.inf
            mse = np.mean((preds - y) ** 2)
            if np.isnan(mse) or np.isinf(mse):
                return np.inf
            return mse
        except Exception:
            return np.inf

    best_predict = None
    best_mse = np.inf

    def update_best(model):
        nonlocal best_predict, best_mse
        if model is None:
            return
        mse = safe_evaluate(model, X_train, y_train)
        if mse < best_mse:
            best_mse = mse
            best_predict = model

    # Strategy 1: Iterative thresholded least squares (sparse symbolic regression)
    # This is the main strategy: discover sparse combinations of basis terms

    def iterative_thresholded_ls(F, y, lam, max_iter=200):
        """Iterative soft-thresholded least squares for sparse recovery."""
        n_samples, n_feats = F.shape
        w = np.zeros(n_feats)
        active = np.ones(n_feats, dtype=bool)
        
        for it in range(max_iter):
            if not np.any(active):
                break
            F_active = F[:, active]
            # Ridge solve
            lam_local = 1e-6
            A = F_active.T @ F_active + lam_local * np.eye(F_active.shape[1])
            try:
                w_active = solve(A, F_active.T @ y)
            except Exception:
                break
            
            # Soft threshold
            w_active = np.sign(w_active) * np.maximum(np.abs(w_active) - lam, 0)
            
            new_active = np.abs(w_active) > 1e-10
            if not np.any(new_active):
                w = np.zeros(n_feats)
                break
            
            # Update active set
            new_w = np.zeros(n_feats)
            new_w[active] = w_active[new_active]
            
            if np.allclose(w, new_w, atol=1e-12):
                w = new_w
                break
            w = new_w
            active = np.abs(w) > 1e-10
        
        return w

    # Try multiple lambda values for sparsity
    for lam in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]:
        if not check_time():
            break
        try:
            w = iterative_thresholded_ls(F_train, yn, lam, max_iter=100)
            if np.all(w == 0):
                continue
            
            # Evaluate
            preds = F_train @ w
            mse = np.mean((preds - yn) ** 2)
            if mse < 1e-6:  # Very good fit
                def predict_sparse(X, F_train, w, y_mean, y_std, X_min, X_range, d, term_descs):
                    Xn = (X - X_min) / X_range
                    F_test, _ = build_basis(Xn, d)
                    return y_mean + y_std * (F_test @ w)
                update_best(lambda X: predict_sparse(X, F_train, w, y_mean, y_std, X_min, X_range, d, term_descs))
        except Exception:
            pass

    # Strategy 2: L1-regularized (Lasso-like) via iterative coordinate descent
    def lasso_coordinate_descent(F, y, alpha, max_iter=500):
        """Simple Lasso via coordinate descent."""
        n_samples, n_feats = F.shape
        w = np.zeros(n_feats)
        F_sq = np.sum(F ** 2, axis=0)
        
        for it in range(max_iter):
            for j in range(n_feats):
                w_old = w[j]
                w[j] = 0
                residual = y - F @ w
                rho = F[:, j] @ residual
                if rho > alpha:
                    w[j] = (rho + alpha) / F_sq[j]
                elif rho < -alpha:
                    w[j] = (rho - alpha) / F_sq[j]
                else:
                    w[j] = 0
        return w

    for alpha in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]:
        if not check_time():
            break
        try:
            w = lasso_coordinate_descent(F_train, yn, alpha, max_iter=200)
            if np.all(w == 0):
                continue
            preds = F_train @ w
            mse = np.mean((preds - yn) ** 2)
            if mse < 1e-4:
                def predict_lasso(X, w, y_mean, y_std, X_min, X_range, d, term_descs):
                    Xn = (X - X_min) / X_range
                    F_test, _ = build_basis(Xn, d)
                    return y_mean + y_std * (F_test @ w)
                update_best(lambda X: predict_lasso(X, w, y_mean, y_std, X_min, X_range, d, term_descs))
        except Exception:
            pass

    # Strategy 3: Full least squares with strong regularization (dense)
    # As a baseline, in case the function is not sparse
    try:
        lam = 1e-3
        A = F_train.T @ F_train + lam * np.eye(n_terms)
        w_dense = solve(A, F_train.T @ yn)
        def predict_dense(X, w, y_mean, y_std, X_min, X_range, d, term_descs):
            Xn = (X - X_min) / X_range
            F_test, _ = build_basis(Xn, d)
            return y_mean + y_std * (F_test @ w)
        update_best(lambda X: predict_dense(X, w_dense, y_mean, y_std, X_min, X_range, d, term_descs))
    except Exception:
        pass

    # Strategy 4: RBF with multiple gamma values
    def make_rbf_model(gamma, n_centers):
        if n > n_centers:
            idx = np.random.choice(n, n_centers, replace=False)
            centers = X_train[idx]
        else:
            centers = X_train
        
        diffs = X_train[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sum(diffs ** 2, axis=2)
        F = np.exp(-gamma * dists)
        
        lam = 1e-5
        A = F.T @ F + lam * np.eye(F.shape[1])
        A[0, 0] -= lam
        w = solve(A, F.T @ yn)
        
        def predict_rbf(X, centers, gamma, w, y_mean, y_std):
            diffs = X[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists = np.sum(diffs ** 2, axis=2)
            F_test = np.exp(-gamma * dists)
            return y_mean + y_std * (F_test @ w)
        
        return lambda X: predict_rbf(X, centers, gamma, w, y_mean, y_std)

    X_range_mean = np.mean(X_range)
    base_gamma = 1.0 / (d * X_range_mean ** 2 + 1e-10)
    
    for g_mult in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]:
        if not check_time():
            break
        try:
            model = make_rbf_model(base_gamma * g_mult, min(50, n))
            update_best(model)
        except Exception:
            pass

    # Strategy 5: Polynomial regression (univariate and multivariate)
    def make_poly_features(X, degree):
        Xn = (X - X_min) / X_range
        cols = [np.ones(len(Xn))]
        for j in range(Xn.shape[1]):
            for k in range(1, degree + 1):
                cols.append(Xn[:, j] ** k)
        for i in range(Xn.shape[1]):
            for j in range(i, Xn.shape[1]):
                if degree >= 2:
                    cols.append(Xn[:, i] * Xn[:, j])
                if degree >= 3:
                    for k in range(j, Xn.shape[1]):
                        cols.append(Xn[:, i] * Xn[:, j] * Xn[:, k])
        return np.column_stack(cols)

    for deg in [1, 2, 3, 4, 5]:
        if not check_time():
            break
        try:
            F = make_poly_features(X_train, deg)
            lam = 1e-5
            A = F.T @ F + lam * np.eye(F.shape[1])
            A[0, 0] -= lam
            w = solve(A, F.T @ yn)
            
            def predict_poly(X, w, y_mean, y_std, deg):
                F_test = make_poly_features(X, deg)
                return y_mean + y_std * (F_test @ w)
            
            update_best(lambda X, w=w, deg=deg: predict_poly(X, w, y_mean, y_std, deg))
        except Exception:
            pass

    # Strategy 6: Neural network-like with ReLU
    def make_relu_model(n_hidden, alpha, seed=42):
        np.random.seed(seed)
        W1 = np.random.randn(d, n_hidden) * alpha
        b1 = np.zeros(n_hidden)
        
        def forward(Xn):
            z1 = Xn @ W1 + b1
            a1 = np.maximum(0, z1)
            return a1
        
        F = forward(Xn_train)
        F = np.column_stack([np.ones(len(F)), F])
        
        lam = 1e-5
        A = F.T @ F + lam * np.eye(F.shape[1])
        A[0, 0] -= lam
        w = solve(A, F.T @ yn)
        
        def predict_relu(X, W1, b1, w, y_mean, y_std):
            Xn = (X - X_min) / X_range
            z1 = Xn @ W1 + b1
            a1 = np.maximum(0, z1)
            F_test = np.column_stack([np.ones(len(a1)), a1])
            return y_mean + y_std * (F_test @ w)
        
        return lambda X: predict_relu(X, W1, b1, w, y_mean, y_std)

    for n_hid in [10, 20, 50, 100]:
        for alpha in [0.5, 1.0, 2.0]:
            if not check_time():
                break
            try:
                model = make_relu_model(n_hid, alpha, seed=42 + n_hid * 10 + int(alpha * 10))
                update_best(model)
            except Exception:
                pass

    # Fallback
    if best_predict is None:
        return lambda X: np.full(len(X), y_mean)

    return best_predict

# EVOLVE-BLOCK-END
