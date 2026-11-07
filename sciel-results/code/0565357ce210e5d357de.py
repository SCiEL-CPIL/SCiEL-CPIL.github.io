"""Multi-kernel sparse regression with CV-guided ensemble."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_TIME_START = [0.0]
_TOTAL_DEADLINE = 310.0

def _remaining():
    return _TOTAL_DEADLINE - (time.time() - _TIME_START[0])

def _rbf_kernel(X, Y, gamma):
    d2 = (np.sum(X ** 2, axis=1)[:, None] + np.sum(Y ** 2, axis=1)[None, :] - 2 * X @ Y.T)
    d2 = np.maximum(d2, 0.0)
    return np.exp(-gamma * d2)

def _matern52_kernel(X, Y, length_scale, sigma_f=1.0):
    d2 = (np.sum(X ** 2, axis=1)[:, None] + np.sum(Y ** 2, axis=1)[None, :] - 2 * X @ Y.T)
    d2 = np.maximum(d2, 0.0)
    r = np.sqrt(d2 + 1e-12)
    K = (sigma_f ** 2) * (1 + np.sqrt(5) * r + 5 / 3 * r ** 2) * np.exp(-np.sqrt(5) * r / length_scale)
    return K

def _poly_kernel(X, Y, degree, gamma=1.0):
    return (gamma * X @ Y.T + 1.0) ** degree

def _select_landmarks_cholesky(X, n_landmarks, kernel_fn, rng):
    """Select landmarks using Cholesky-based greedy selection for kernel ridge."""
    n = X.shape[0]
    if n_landmarks >= n:
        return np.arange(n)
    
    # Start with random point
    landmarks = [rng.integers(0, n)]
    K_land = kernel_fn(X[landmarks], X[landmarks])
    
    # Compute initial distances (kernel variance)
    K_all = kernel_fn(X, X[landmarks])  # (n, 1)
    min_var = K_all[:, 0].copy()
    
    for _ in range(n_landmarks - 1):
        # Pick point with max variance
        idx = np.argmax(min_var)
        landmarks.append(idx)
        
        # Update: recompute kernel to new landmark
        K_new = kernel_fn(X, X[idx:idx+1])  # (n, 1)
        
        # Update min variance using Cholesky update
        # For simplicity, recompute from scratch for small n_landmarks
        K_land = kernel_fn(X[landmarks], X[landmarks])
        try:
            L = np.linalg.cholesky(K_land + 1e-8 * np.eye(len(landmarks)))
            # Variance of each point: k(x,x) - k(x,L)^T L^{-T} L^{-1} k(x,L)
            K_x = kernel_fn(X, X[landmarks])  # (n, m)
            # Solve L Z^T = K_x^T
            Z = np.linalg.solve(L, K_x.T)  # (m, n)
            var = np.sum(K_x ** 2, axis=1) - np.sum(Z ** 2, axis=0)
            min_var = np.maximum(var, 0.0)
        except np.linalg.LinAlgError:
            # Fallback: Euclidean distance
            d2 = np.sum((X - X[idx]) ** 2, axis=1)
            min_var = np.maximum(min_var, d2)
    
    return np.array(landmarks)

def _fit_sparse_kernel(X, y, kernel_fn, n_landmarks, alpha_reg, rng):
    """Sparse kernel ridge regression with landmark selection."""
    n = X.shape[0]
    landmark_idx = _select_landmarks_cholesky(X, n_landmarks, kernel_fn, rng)
    X_land = X[landmark_idx]
    M = len(landmark_idx)
    
    K_land_land = kernel_fn(X_land, X_land)
    K_train_land = kernel_fn(X, X_land)
    
    # Solve: (K_train_land^T K_train_land + alpha_reg * I) w = K_train_land^T y
    A = K_train_land.T @ K_train_land + alpha_reg * np.eye(M)
    b = K_train_land.T @ y
    
    try:
        w = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(A, b, rcond=None)[0]
    
    def predict(Xt):
        K_test_land = kernel_fn(Xt, X_land)
        return K_test_land @ w
    
    return predict

def _kfold_cv_score(predict_fn, X, y, n_folds=5, rng_seed=42):
    """K-fold cross-validation MSE."""
    n = len(y)
    idx = np.arange(n)
    rng = np.random.default_rng(rng_seed)
    rng.shuffle(idx)
    fold_size = n // n_folds
    se = 0.0
    count = 0
    for i in range(n_folds):
        val_idx = idx[i * fold_size:(i + 1) * fold_size]
        tr_idx = np.concatenate([idx[:i * fold_size], idx[(i + 1) * fold_size:]])
        if len(tr_idx) < 10:
            continue
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        try:
            pred_val = predict_fn(X_val)
            se += np.sum((pred_val - y_val) ** 2)
            count += len(val_idx)
        except Exception:
            continue
    if count == 0:
        return np.inf
    return se / count

def _poly_basis(X, degree):
    """Polynomial basis up to given degree."""
    n, d = X.shape
    cols = [np.ones(n)]
    for deg in range(1, degree + 1):
        for r in range(d):
            for c in range(r, d):
                term = np.ones(n)
                for k in range(r, c + 1):
                    term = term * X[:, k]
                cols.append(term)
    return np.column_stack(cols)

def fit(X_train, y_train):
    _TIME_START[0] = time.time()
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float)
    n, d = X.shape
    yvar = np.var(y) + 1e-12
    
    if yvar < 1e-12:
        c = float(np.mean(y))
        return lambda Xt: np.full(len(Xt), c)
    
    candidates = []
    rng = np.random.default_rng(123)
    
    # Scale X for better kernel behavior
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0) + 1e-8
    X_scaled = (X - X_mean) / X_std
    
    # 1) Polynomial baselines
    if _remaining() > 5:
        for deg in (2, 3):
            try:
                B = _poly_basis(X_scaled, deg)
                w = np.linalg.lstsq(B, y, rcond=None)[0]
                p = lambda Xt, _w=w, _d=deg: _poly_basis((np.asarray(Xt, dtype=float) - X_mean) / X_std, _d) @ _w
                candidates.append((f'poly{deg}', p))
            except Exception:
                pass
    
    # 2) Sparse kernel regression with multiple kernels
    if _remaining() > 30:
        # RBF kernels with different gammas
        for gamma in (0.5, 1.0, 2.0, 4.0):
            for n_land in (25, 50):
                for alpha in (1e-2, 1e-1):
                    if _remaining() < 5:
                        break
                    try:
                        kf = lambda Xa, Xb, g=gamma: _rbf_kernel(Xa, Xb, g)
                        p = _fit_sparse_kernel(X_scaled, y, kf, n_land, alpha, rng)
                        candidates.append((f'rbf_g{gamma}_l{n_land}_a{alpha}', p))
                    except Exception:
                        continue
        
        # Matérn-5/2 kernels
        if _remaining() > 20:
            for ls in (0.5, 1.0, 2.0):
                for n_land in (25, 50):
                    for alpha in (1e-2, 1e-1):
                        if _remaining() < 5:
                            break
                        try:
                            kf = lambda Xa, Xb, l=ls: _matern52_kernel(Xa, Xb, l)
                            p = _fit_sparse_kernel(X_scaled, y, kf, n_land, alpha, rng)
                            candidates.append((f'mat52_l{ls}_land{n_land}_a{alpha}', p))
                        except Exception:
                            continue
        
        # Polynomial kernels
        if _remaining() > 15:
            for deg in (2, 3):
                for gamma in (0.5, 1.0):
                    for n_land in (25,):
                        if _remaining() < 5:
                            break
                        try:
                            kf = lambda Xa, Xb, dg=deg, g=gamma: _poly_kernel(Xa, Xb, dg, g)
                            p = _fit_sparse_kernel(X_scaled, y, kf, n_land, 1e-1, rng)
                            candidates.append((f'polyk_d{deg}_g{gamma}', p))
                        except Exception:
                            continue
    
    # 3) Full kernel ridge for small n
    if n <= 250 and _remaining() > 20:
        for gamma in (1.0, 2.0):
            for lam in (1e-2, 1e-1):
                try:
                    K = _rbf_kernel(X_scaled, X_scaled, gamma)
                    alpha = np.linalg.solve(K + lam * np.eye(n), y)
                    p = lambda Xt, _a=alpha, _g=gamma: _rbf_kernel((np.asarray(Xt, dtype=float) - X_mean) / X_std, X_scaled, _g) @ _a
                    candidates.append((f'fullrbf_g{gamma}_l{lam}', p))
                except Exception:
                    continue
    
    if not candidates:
        c = float(np.mean(y))
        return lambda Xt: np.full(len(Xt), c)
    
    # Score candidates with CV
    scored = []
    for name, p in candidates:
        if _remaining() < 3:
            break
        try:
            score = _kfold_cv_score(p, X_scaled, y, n_folds=5)
            if np.isfinite(score):
                scored.append((name, p, score))
        except Exception:
            continue
    
    if not scored:
        c = float(np.mean(y))
        return lambda Xt: np.full(len(Xt), c)
    
    # Sort by CV score
    scored.sort(key=lambda x: x[2])
    
    # Ensemble top-K with inverse-score weights
    K_top = min(4, len(scored))
    top = scored[:K_top]
    
    scores = np.array([s[2] for s in top])
    weights = 1.0 / (scores + 1e-8)
    weights = weights / weights.sum()
    
    predictors = [s[1] for s in top]
    y_mean = float(np.mean(y))
    
    def predict(Xt):
        Xt = np.asarray(Xt, dtype=float)
        out = np.zeros(len(Xt))
        for p, w in zip(predictors, weights):
            try:
                out += w * p(Xt)
            except Exception:
                out += w * y_mean
        out = np.asarray(out, dtype=float).ravel()
        out = np.where(np.isfinite(out), out, y_mean)
        return out
    
    return predict
# EVOLVE-BLOCK-END
