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
        """Build basis matrix with many candidate terms. Deterministic and safe."""
        # Clamp Xn to [0, 1] to ensure safety for log, sqrt, arcsin
        Xn = np.clip(Xn, 0.0, 1.0)
        
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
            # Exponential (clamped to avoid overflow, though xj in [0,1] is safe)
            terms.append(np.exp(xj))
            term_descs.append(f"exp(x{j})")
            terms.append(np.exp(-xj))
            term_descs.append(f"exp(-x{j})")
            # Log (safe because xj >= 0, add small eps)
            terms.append(np.log(xj + 1e-8))
            term_descs.append(f"log(x{j}+eps)")
            # Sqrt
            terms.append(np.sqrt(xj))
            term_descs.append(f"sqrt(x{j})")
            # Inverse (safe because xj >= 0, add small eps)
            terms.append(1.0 / (xj + 1e-8))
            term_descs.append(f"1/(x{j}+eps)")
            # Tanh
            terms.append(np.tanh(xj))
            term_descs.append(f"tanh(x{j})")
            # Arcsin/arccos (clamped to [0,1] already, but arcsin needs [-1,1], so it's safe)
            terms.append(np.arcsin(xj))
            term_descs.append(f"arcsin(x{j})")
            terms.append(np.arccos(xj))
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
                    # Trig interactions
                    terms.append(np.sin(xi + xj))
                    term_descs.append(f"sin(x{i}+x{j})")
                    terms.append(np.cos(xi + xj))
                    term_descs.append(f"cos(x{i}+x{j})")
                    # Exp interactions
                    terms.append(np.exp(xi + xj))
                    term_descs.append(f"exp(x{i}+x{j})")
                    # Log interactions
                    terms.append(np.log(xi * xj + 1e-8))
                    term_descs.append(f"log(x{i}*x{j})")
                    # Sqrt interactions
                    terms.append(np.sqrt(xi * xj))
                    term_descs.append(f"sqrt(x{i}*x{j})")
                    # Ratio
                    terms.append(xi / (xj + 1e-8))
                    term_descs.append(f"x{i}/(x{j}+eps)")
                    terms.append(xj / (xi + 1e-8))
                    term_descs.append(f"x{j}/(x{i}+eps)")

        # Triple interactions for d == 3
        if d == 3:
            x0, x1, x2 = Xn[:, 0], Xn[:, 1], Xn[:, 2]
            terms.append(x0 * x1 * x2)
            term_descs.append("x0*x1*x2")
            terms.append(np.sin(x0 + x1 + x2))
            term_descs.append("sin(x0+x1+x2)")
            terms.append(np.exp(x0 + x1 + x2))
            term_descs.append("exp(x0+x1+x2)")
            terms.append(np.sqrt(x0 * x1 * x2))
            term_descs.append("sqrt(x0*x1*x2)")
            terms.append(x0 / (x1 + x2 + 1e-8))
            term_descs.append("x0/(x1+x2+eps)")
            terms.append(x1 / (x0 + x2 + 1e-8))
            term_descs.append("x1/(x0+x2+eps)")
            terms.append(x2 / (x0 + x1 + 1e-8))
            term_descs.append("x2/(x0+x1+eps)")

        F = np.column_stack(terms)
        # Force finite values just in case, though clamping should prevent this
        F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
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

    # Strategy 1: Ridge Regression on the fixed basis
    for lam in [1e-4, 1e-3, 1e-2, 0.1, 1.0]:
        if not check_time():
            break
        try:
            A = F_train.T @ F_train + lam * np.eye(n_terms)
            w = solve(A, F_train.T @ yn)
            
            def predict_ridge(X, w, y_mean, y_std, X_min, X_range, d):
                Xn = (X - X_min) / X_range
                F_test, _ = build_basis(Xn, d)
                return y_mean + y_std * (F_test @ w)
            
            update_best(lambda X: predict_ridge(X, w, y_mean, y_std, X_min, X_range, d))
        except Exception:
            pass

    # Strategy 2: RBF (Gaussian Process-like)
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
    
    for g_mult in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        if not check_time():
            break
        try:
            model = make_rbf_model(base_gamma * g_mult, min(50, n))
            update_best(model)
        except Exception:
            pass

    # Fallback
    if best_predict is None:
        return lambda X: np.full(len(X), y_mean)

    return best_predict

# EVOLVE-BLOCK-END
