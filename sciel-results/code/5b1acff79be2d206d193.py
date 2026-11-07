"""Multi-stage symbolic regression with ensemble selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_TIME_START = [0.0]
_TOTAL_DEADLINE = 310.0  # Leave 20s margin before 330s kill

def _remaining():
    return _TOTAL_DEADLINE - (time.time() - _TIME_START[0])

def _poly_basis(X, degree):
    """Return a basis matrix of all monomials up to `degree` (incl. constant)."""
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

def _rich_basis(X):
    """Richer basis: poly up to degree 3 + sin, cos, exp, reciprocal."""
    n, d = X.shape
    cols = [np.ones(n)]
    # Linear
    for j in range(d):
        cols.append(X[:, j])
    # Quadratic
    for i in range(d):
        for j in range(i, d):
            cols.append(X[:, i] * X[:, j])
    # Cubic
    for i in range(d):
        for j in range(i, d):
            for k in range(j, d):
                cols.append(X[:, i] * X[:, j] * X[:, k])
    # Trig
    for j in range(d):
        cols.append(np.sin(X[:, j]))
        cols.append(np.cos(X[:, j]))
    # Exp (scaled to avoid overflow)
    for j in range(d):
        x_scaled = (X[:, j] - np.mean(X[:, j])) / (np.std(X[:, j]) + 1e-8)
        cols.append(np.exp(np.clip(x_scaled, -5, 5)))
    # Reciprocal
    for j in range(d):
        cols.append(1.0 / (X[:, j] + 1e-8))
    return np.column_stack(cols)

def _fit_ridge(X, y, lam):
    """Fit ridge regression on precomputed features."""
    n, m = X.shape
    A = X.T @ X + lam * np.eye(m)
    w = np.linalg.solve(A, X.T @ y)
    return w

def _rbf_kernel(X, Y, gamma):
    d2 = (np.sum(X ** 2, axis=1)[:, None] + np.sum(Y ** 2, axis=1)[None, :] - 2 * X @ Y.T)
    d2 = np.maximum(d2, 0.0)
    return np.exp(-gamma * d2)

def _fit_ridge_rbf(X, y, gamma, lam):
    K = _rbf_kernel(X, X, gamma)
    n = K.shape[0]
    alpha = np.linalg.solve(K + lam * np.eye(n), y)
    def predict(Xt):
        return _rbf_kernel(Xt, X, gamma) @ alpha
    return predict

def _fit_nystrom_kernel(X, y, gamma, n_landmarks, alpha_reg):
    """Nyström approximate kernel regression with elastic-net regularization."""
    n = X.shape[0]
    rng = np.random.default_rng(42)
    
    # Select landmarks using farthest-point sampling
    if n_landmarks >= n:
        landmark_idx = np.arange(n)
    else:
        landmarks = [rng.integers(0, n)]
        d2_first = np.sum((X - X[landmarks[0]]) ** 2, axis=1)
        d2_first = np.maximum(d2_first, 0.0)
        min_dist = d2_first.copy()
        
        for _ in range(n_landmarks - 1):
            idx = np.argmax(min_dist)
            landmarks.append(idx)
            d2_new = np.sum((X - X[idx]) ** 2, axis=1)
            d2_new = np.maximum(d2_new, 0.0)
            min_dist = np.minimum(min_dist, d2_new)
        
        landmark_idx = np.array(landmarks)
    
    X_landmarks = X[landmark_idx]
    
    # Compute kernel matrices
    K_landmarks = _rbf_kernel(X_landmarks, X_landmarks, gamma)  # (M, M)
    K_train_landmarks = _rbf_kernel(X, X_landmarks, gamma)  # (N, M)
    
    # Nyström approximation: K ≈ K_train_landmarks @ K_landmarks^{-1} @ K_train_landmarks.T
    # Solve for coefficients with elastic-net regularization
    # Minimize: ||K_train_landmarks @ w - y||^2 + alpha_reg * ||w||^2
    M = len(landmark_idx)
    A = K_train_landmarks.T @ K_train_landmarks + alpha_reg * np.eye(M)
    b = K_train_landmarks.T @ y
    
    try:
        w = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(A, b, rcond=None)[0]
    
    def predict(Xt):
        K_test_landmarks = _rbf_kernel(Xt, X_landmarks, gamma)  # (Nt, M)
        return K_test_landmarks @ w
    
    return predict

def _fit_gp_m52(X, y, max_iter=50):
    """Matern-5/2 GP regression with hyperparameter optimization."""
    n, d = X.shape
    
    def build_K(theta):
        l, sigma_f, sigma_n = theta
        l = max(l, 1e-4)
        d2 = (np.sum(X ** 2, axis=1)[:, None] + np.sum(X ** 2, axis=1)[None, :] - 2 * X @ X.T)
        d2 = np.maximum(d2, 0.0)
        r = np.sqrt(d2 + 1e-12)
        K = (sigma_f ** 2) * (1 + np.sqrt(5) * r + 5 / 3 * r ** 2) * np.exp(-np.sqrt(5) * r / l)
        K[np.diag_indices(n)] += sigma_n ** 2
        return K
    
    def log_marginal(theta):
        K = build_K(theta)
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return 1e10
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        lmc = 0.5 * np.sum(y * alpha) + np.sum(np.log(np.diag(L))) + 0.5 * n * np.log(2 * np.pi)
        return lmc
    
    scale = np.max(np.ptp(X, axis=0)) + 1e-6
    yvar = np.var(y) + 1e-9
    theta0 = np.array([scale, np.sqrt(yvar), np.sqrt(yvar) * 1e-2])
    
    res = minimize(log_marginal, theta0, method='Nelder-Mead',
                   options={'maxiter': max_iter, 'xatol': 1e-4, 'fatol': 1e-4})
    l, sigma_f, sigma_n = res.x
    l = max(l, 1e-4)
    K = build_K(res.x)
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    
    def predict(Xt):
        d2 = (np.sum(Xt ** 2, axis=1)[:, None] + np.sum(X ** 2, axis=1)[None, :] - 2 * Xt @ X.T)
        d2 = np.maximum(d2, 0.0)
        r = np.sqrt(d2 + 1e-12)
        Ks = (sigma_f ** 2) * (1 + np.sqrt(5) * r + 5 / 3 * r ** 2) * np.exp(-np.sqrt(5) * r / l)
        return Ks @ alpha
    return predict

def _fit_rational(X, y):
    """Fit rational function approximator: ratio of two linear basis expansions."""
    n, d = X.shape
    B = _poly_basis(X, 1)  # constant + linear
    m = B.shape[1]
    
    def rational_predict(coeffs, Xt):
        cnum = coeffs[:m]
        cden = coeffs[m:]
        num = Xt @ cnum
        den = Xt @ cden
        den = np.where(np.abs(den) < 1e-6, np.sign(den) * 1e-6 + 1e-6, den)
        return num / den
    
    x0 = np.zeros(2 * m)
    x0[m] = 1.0  # constant term of denominator
    
    def obj(coeffs):
        pred = rational_predict(coeffs, X)
        return np.sum((pred - y) ** 2)
    
    res = minimize(obj, x0, method='Nelder-Mead', options={'maxiter': 300})
    p = lambda Xt, _c=res.x: rational_predict(_c, Xt)
    return p

def _loo_cv_score(predict_fn, X, y, n_folds=5):
    """K-fold cross-validation score."""
    n = len(y)
    idx = np.arange(n)
    rng = np.random.default_rng(42)
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

def _ensemble_predict(predictors, weights, Xt):
    """Weighted ensemble prediction."""
    total = np.zeros(len(Xt))
    for p, w in zip(predictors, weights):
        total += w * p(Xt)
    return total

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
    
    # 1) Degree-2 polynomial (fast baseline)
    if _remaining() > 5:
        try:
            B = _poly_basis(X, 2)
            w = np.linalg.lstsq(B, y, rcond=None)[0]
            p = lambda Xt: _poly_basis(Xt, 2) @ w
            candidates.append(('poly2', p))
        except Exception:
            pass
    
    # 2) Rich basis with ridge
    if _remaining() > 10:
        try:
            B = _rich_basis(X)
            for lam in (1e-3, 1e-2, 1e-1):
                w = _fit_ridge(B, y, lam)
                p = lambda Xt, _w=w: _rich_basis(Xt) @ _w
                candidates.append((f'rich_l{lam}', p))
        except Exception:
            pass
    
    # 3) Nyström sparse kernel regression (primary approach)
    if _remaining() > 25:
        for gamma in (0.5, 1.0, 2.0, 5.0):
            for n_land in (20, 40, 60):
                for alpha_reg in (1e-2, 1e-1, 1.0):
                    if _remaining() < 5:
                        break
                    try:
                        p = _fit_nystrom_kernel(X, y, gamma, n_land, alpha_reg)
                        candidates.append((f'nystrom_g{gamma}_l{n_land}_a{alpha_reg}', p))
                    except Exception:
                        continue
    
    # 4) Full RBF kernel ridge regression (backup)
    if _remaining() > 30:
        for gamma in (0.5, 1.0, 2.0):
            for lam in (1e-2, 1e-1):
                if _remaining() < 5:
                    break
                try:
                    p = _fit_ridge_rbf(X, y, gamma, lam)
                    candidates.append((f'rbf_g{gamma}_l{lam}', p))
                except Exception:
                    continue
    
    # 5) Matern-5/2 Gaussian Process
    if _remaining() > 60:
        try:
            p = _fit_gp_m52(X, y, max_iter=40)
            candidates.append(('gp_m52', p))
        except Exception:
            pass
    
    # 6) Rational function approximator
    if _remaining() > 30:
        try:
            p = _fit_rational(X, y)
            candidates.append(('rational', p))
        except Exception:
            pass
    
    if not candidates:
        c = float(np.mean(y))
        return lambda Xt: np.full(len(Xt), c)
    
    # Score all candidates using K-fold CV
    scored_candidates = []
    for name, p in candidates:
        if _remaining() < 5:
            break
        try:
            if _remaining() > 20:
                score = _loo_cv_score(p, X, y, n_folds=5)
            else:
                pred = p(X)
                score = np.mean((pred - y) ** 2)
            if np.isfinite(score):
                scored_candidates.append((name, p, score))
        except Exception:
            continue
    
    if not scored_candidates:
        c = float(np.mean(y))
        return lambda Xt: np.full(len(Xt), c)
    
    # Sort by CV score
    scored_candidates.sort(key=lambda x: x[2])
    
    # Ensemble of top-K models (weighted by inverse CV score)
    K_top = min(3, len(scored_candidates))
    top_models = scored_candidates[:K_top]
    
    # Weight by inverse CV score (better models get higher weight)
    scores = np.array([s[2] for s in top_models])
    weights = 1.0 / (scores + 1e-8)
    weights = weights / weights.sum()
    
    predictors = [s[1] for s in top_models]
    
    def predict(Xt):
        Xt = np.asarray(Xt, dtype=float)
        out = _ensemble_predict(predictors, weights, Xt)
        out = np.asarray(out, dtype=float).ravel()
        out = np.where(np.isfinite(out), out, np.nanmean(y))
        return out
    
    return predict
# EVOLVE-BLOCK-END
