"""Symbolic regression via multi-strategy ensemble with time-bounded search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

# Global deadline: must finish all 4 fits within 330s wall-clock.
# We'll use a per-fit budget of ~80s to be safe (4 * 80 = 320s).
_FIT_BUDGET = 75.0  # seconds per fit call


def _safe_predict(X, model):
    """Safely evaluate a model, returning finite values."""
    try:
        pred = model(X)
        pred = np.asarray(pred, dtype=float)
        if pred.ndim == 0:
            pred = np.full(X.shape[0], float(pred))
        pred = np.where(np.isfinite(pred), pred, np.nan)
        if np.any(np.isnan(pred)):
            # Replace NaNs with mean of valid predictions
            valid = pred[~np.isnan(pred)]
            fill = np.mean(valid) if len(valid) > 0 else 0.0
            pred = np.where(np.isnan(pred), fill, pred)
        return pred
    except Exception:
        return np.zeros(X.shape[0])


def _make_poly_predict(X_train, y_train, degree):
    """Polynomial regression with feature expansion."""
    d = X_train.shape[1]
    
    def features(X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        cols = [np.ones(X.shape[0])]
        # All monomials up to 'degree'
        for total_deg in range(1, degree + 1):
            for i in range(d):
                for j in range(d):
                    if i <= j and (i + j) <= total_deg:
                        # Only include monomials x_i^a * x_j^b where a+b=total_deg and a<=b
                        pass
        # Simpler: just do all monomials x_i^p * x_j^q with p+q <= degree
        cols = [np.ones(X.shape[0])]
        for p in range(1, degree + 1):
            for i in range(d):
                cols.append(X[:, i] ** p)
        for p in range(1, degree + 1):
            for q in range(1, degree + 1 - p):
                for i in range(d):
                    for j in range(i + 1, d):
                        cols.append(X[:, i] ** p * X[:, j] ** q)
        # Also cross terms with same variable already covered
        # Remove duplicates
        arr = np.column_stack(cols) if len(cols) > 1 else np.array(cols[0]).reshape(-1, 1)
        return arr
    
    Ft = features(X_train)
    try:
        w, *_ = np.linalg.lstsq(Ft, y_train, rcond=None)
    except Exception:
        return lambda X: np.full(X.shape[0], np.mean(y_train))
    
    def predict(X):
        return features(X) @ w
    
    return predict


def _make_rbf_predict(X_train, y_train, n_centers, gamma):
    """RBF kernel regression."""
    d = X_train.shape[1]
    if d == 1:
        Xf = X_train
    else:
        Xf = X_train
    
    # Select centers
    if n_centers >= len(X_train):
        centers = Xf
    else:
        idx = np.random.choice(len(X_train), n_centers, replace=False)
        centers = Xf[idx]
    
    def rbf_kernel(A, B):
        # Squared Euclidean distance
        A2 = np.sum(A ** 2, axis=1, keepdims=True)
        B2 = np.sum(B ** 2, axis=1, keepdims=True)
        D2 = A2 + B2.T - 2 * A @ B.T
        D2 = np.maximum(D2, 0)
        return np.exp(-gamma * D2)
    
    K_train = rbf_kernel(Xf, centers)
    # Add small ridge for stability
    K_train += 1e-6 * np.eye(len(centers))
    
    try:
        w = np.linalg.solve(K_train, y_train)
    except Exception:
        w, *_ = np.linalg.lstsq(K_train, y_train, rcond=None)
    
    def predict(X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        K_test = rbf_kernel(X, centers)
        return K_test @ w
    
    return predict


def _make_gp_predict(X_train, y_train, length_scale, sigma_f, noise_var):
    """Gaussian Process regression with RBF kernel."""
    d = X_train.shape[1]
    n = len(X_train)
    
    def rbf_kernel(A, B, ls, sf):
        if d == 1:
            A = A.reshape(-1, 1)
            B = B.reshape(-1, 1)
        A2 = np.sum(A ** 2, axis=1, keepdims=True)
        B2 = np.sum(B ** 2, axis=1, keepdims=True)
        D2 = A2 + B2.T - 2 * A @ B.T
        D2 = np.maximum(D2, 0)
        return sf ** 2 * np.exp(-0.5 * D2 / (ls ** 2))
    
    K = rbf_kernel(X_train, X_train, length_scale, sigma_f)
    K += noise_var * np.eye(n)
    
    try:
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
    except Exception:
        K += 1e-4 * np.eye(n)
        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
        except Exception:
            # Fallback to ridge
            K += 0.01 * np.eye(n)
            w, *_ = np.linalg.lstsq(K, y_train, rcond=None)
            def predict(X):
                X = np.asarray(X, dtype=float)
                if X.ndim == 1:
                    X = X.reshape(-1, 1)
                Kt = rbf_kernel(X, X_train, length_scale, sigma_f)
                return Kt @ w
            return predict
    
    def predict(X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        K_star = rbf_kernel(X, X_train, length_scale, sigma_f)
        return K_star @ alpha
    
    return predict


def _optimize_gp_hyperparams(X_train, y_train, deadline):
    """Optimize GP hyperparameters using L-BFGS-B."""
    d = X_train.shape[1]
    n = len(X_train)
    
    # Normalize inputs to [0,1] for better conditioning
    x_min = X_train.min(axis=0)
    x_max = X_train.max(axis=0)
    x_range = x_max - x_min
    x_range[x_range < 1e-10] = 1.0
    X_norm = (X_train - x_min) / x_range
    
    y_mean = np.mean(y_train)
    y_std = np.std(y_train)
    if y_std < 1e-10:
        y_std = 1.0
    y_norm = (y_train - y_mean) / y_std
    
    def rbf_kernel_norm(A, B, ls, sf):
        A2 = np.sum(A ** 2, axis=1, keepdims=True)
        B2 = np.sum(B ** 2, axis=1, keepdims=True)
        D2 = A2 + B2.T - 2 * A @ B.T
        D2 = np.maximum(D2, 0)
        return sf ** 2 * np.exp(-0.5 * D2 / (ls ** 2))
    
    def neg_log_marginal(log_params):
        log_ls, log_sf, log_noise = log_params
        ls = np.exp(log_ls)
        sf = np.exp(log_sf)
        noise = np.exp(log_noise)
        
        K = rbf_kernel_norm(X_norm, X_norm, ls, sf)
        K += noise * np.eye(n)
        
        try:
            L = np.linalg.cholesky(K)
        except Exception:
            return 1e10
        
        try:
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))
            log_det = 2 * np.sum(np.log(np.diag(L)))
            marginal = -0.5 * y_norm @ alpha - 0.5 * log_det - 0.5 * n * np.log(2 * np.pi)
            return -marginal
        except Exception:
            return 1e10
    
    # Multi-start
    best_score = 1e10
    best_params = None
    start_time = time.time()
    
    # Grid of starting points
    ls_starts = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
    sf_starts = [0.5, 1.0, 2.0]
    noise_starts = [1e-4, 1e-3, 1e-2]
    
    count = 0
    for ls in ls_starts:
        for sf in sf_starts:
            for noise in noise_starts:
                if time.time() - start_time > deadline * 0.6:
                    break
                x0 = [np.log(ls), np.log(sf), np.log(noise)]
                try:
                    res = minimize(neg_log_marginal, x0, method='L-BFGS-B',
                                  options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6})
                    if res.fun < best_score:
                        best_score = res.fun
                        best_params = res.x
                except Exception:
                    pass
                count += 1
    
    if best_params is None:
        best_params = [np.log(1.0), np.log(1.0), np.log(1e-3)]
    
    ls = np.exp(best_params[0])
    sf = np.exp(best_params[1])
    noise = np.exp(best_params[2])
    
    # Build final GP with original-scale kernel
    def predict(X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        X_n = (X - x_min) / x_range
        K_star = rbf_kernel_norm(X_n, X_norm, ls, sf)
        return K_star @ alpha * y_std + y_mean
    
    return predict


def _fit_nn(X_train, y_train, deadline):
    """Simple neural network fit using gradient descent."""
    d = X_train.shape[1]
    n = len(X_train)
    
    # Normalize
    x_mean = X_train.mean(axis=0)
    x_std = X_train.std(axis=0)
    x_std[x_std < 1e-10] = 1.0
    X_norm = (X_train - x_mean) / x_std
    
    y_mean = y_train.mean()
    y_std = y_train.std()
    if y_std < 1e-10:
        y_std = 1.0
    y_norm = (y_train - y_mean) / y_std
    
    # Hidden layer size
    hidden = 32
    
    # Initialize
    rng = np.random.default_rng(42)
    W1 = rng.normal(0, 1.0 / np.sqrt(d), (d, hidden))
    b1 = np.zeros(hidden)
    W2 = rng.normal(0, 1.0 / np.sqrt(hidden), (hidden, 1))
    b2 = np.zeros(1)
    
    lr = 0.01
    epochs = 2000
    start_time = time.time()
    
    for epoch in range(epochs):
        if time.time() - start_time > deadline * 0.4:
            break
        
        # Forward
        Z1 = X_norm @ W1 + b1
        A1 = np.tanh(Z1)
        Z2 = A1 @ W2 + b2
        A2 = Z2.flatten()
        
        # Loss
        loss = np.mean((A2 - y_norm) ** 2)
        
        # Backward
        dA2 = 2 * (A2 - y_norm) / n
        dW2 = A1.T @ dA2.reshape(-1, 1)
        db2 = np.sum(dA2, axis=0)
        dZ2 = dA2.reshape(-1, 1)
        dA1 = dZ2 @ W2.T
        dZ1 = dA1 * (1 - A1 ** 2)
        dW1 = X_norm.T @ dZ1
        db1 = np.sum(dZ1, axis=0)
        
        # Update
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2
    
    def predict(X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        X_n = (X - x_mean) / x_std
        Z1 = X_n @ W1 + b1
        A1 = np.tanh(Z1)
        Z2 = A1 @ W2 + b2
        return Z2.flatten() * y_std + y_mean
    
    return predict


def fit(X_train, y_train):
    """Multi-strategy symbolic regression fit."""
    start_time = time.time()
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    
    n = X_train.shape[0]
    d = X_train.shape[1]
    
    candidates = []  # list of (name, predict_fn, train_mse)
    
    # 1. Polynomial regression (degree 2, 3, 4)
    for deg in [2, 3, 4]:
        try:
            pred_fn = _make_poly_predict(X_train, y_train, deg)
            train_pred = pred_fn(X_train)
            mse = np.mean((train_pred - y_train) ** 2)
            candidates.append((f'poly{deg}', pred_fn, mse))
        except Exception:
            pass
    
    # 2. RBF kernel regression with various gammas
    for gamma in [0.1, 0.5, 1.0, 2.0, 5.0]:
        try:
            pred_fn = _make_rbf_predict(X_train, y_train, n, gamma)
            train_pred = pred_fn(X_train)
            mse = np.mean((train_pred - y_train) ** 2)
            candidates.append((f'rbf_{gamma}', pred_fn, mse))
        except Exception:
            pass
    
    # 3. GP with optimized hyperparameters
    try:
        gp_deadline = min(_FIT_BUDGET * 0.35, max(5.0, _FIT_BUDGET - (time.time() - start_time)))
        gp_pred = _optimize_gp_hyperparams(X_train, y_train, gp_deadline)
        train_pred = gp_pred(X_train)
        mse = np.mean((train_pred - y_train) ** 2)
        candidates.append(('gp', gp_pred, mse))
    except Exception:
        pass
    
    # 4. Simple NN
    try:
        nn_deadline = min(_FIT_BUDGET * 0.2, max(3.0, _FIT_BUDGET - (time.time() - start_time)))
        nn_pred = _fit_nn(X_train, y_train, nn_deadline)
        train_pred = nn_pred(X_train)
        mse = np.mean((train_pred - y_train) ** 2)
        candidates.append(('nn', nn_pred, mse))
    except Exception:
        pass
    
    if not candidates:
        # Fallback: mean
        y_mean = np.mean(y_train)
        return lambda X: np.full(np.asarray(X, dtype=float).shape[0], y_mean)
    
    # Select top candidates by training MSE and ensemble them
    candidates.sort(key=lambda x: x[2])
    
    # Use top 3 for ensemble (weighted by inverse MSE)
    top_k = min(3, len(candidates))
    top_candidates = candidates[:top_k]
    
    # If the best is much better than others, just use it
    best_mse = top_candidates[0][2]
    if len(top_candidates) > 1 and best_mse < 0.01 * top_candidates[1][2]:
        best_pred = top_candidates[0][1]
        def predict(X):
            return _safe_predict(X, best_pred)
        return predict
    
    # Ensemble: weighted average
    weights = []
    for name, pred_fn, mse in top_candidates:
        w = 1.0 / (mse + 1e-10)
        weights.append(w)
    total_w = sum(weights)
    weights = [w / total_w for w in weights]
    
    def predict(X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        result = np.zeros(X.shape[0])
        for (name, pred_fn, mse), w in zip(top_candidates, weights):
            result += w * _safe_predict(X, pred_fn)
        return result
    
    return predict
# EVOLVE-BLOCK-END
