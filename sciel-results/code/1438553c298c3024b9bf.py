"""Symbolic regression with Chebyshev+Fourier basis, RBF kernel, and efficient selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

_START_TIME = None
TIME_LIMIT = 330.0
SAFETY_MARGIN = 25.0


def _time_left():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
        return TIME_LIMIT - SAFETY_MARGIN
    return TIME_LIMIT - SAFETY_MARGIN - (time.time() - _START_TIME)


def _chebyshev_features(X, n_terms):
    """Chebyshev polynomial features for 1D input, mapped to [-1,1]."""
    x = X[:, 0]
    x_min, x_max = -3.5, 3.5
    # Map to [-1, 1]
    x_norm = (x - (x_min + x_max) / 2) / ((x_max - x_min) / 2)
    x_norm = np.clip(x_norm, -1, 1)
    
    cols = [np.ones(len(x))]
    T0 = np.ones(len(x))
    T1 = x_norm.copy()
    cols.append(T1)
    for k in range(2, n_terms + 1):
        Tk = 2 * x_norm * T1 - T0
        cols.append(Tk)
        T0, T1 = T1, Tk
    return np.column_stack(cols)


def _fourier_features(X, n_harmonics, x_min=-3.5, x_max=3.5):
    """Fourier features for 1D input."""
    x = X[:, 0]
    cols = [np.ones(len(x))]
    L = x_max - x_min
    for k in range(1, n_harmonics + 1):
        freq = 2 * np.pi * k / L
        cols.append(np.sin(freq * x))
        cols.append(np.cos(freq * x))
    return np.column_stack(cols)


def _combined_features(X, n_cheb=15, n_fourier=8):
    """Combined Chebyshev and Fourier features."""
    F1 = _chebyshev_features(X, n_cheb)
    F2 = _fourier_features(X, n_harmonics=n_fourier)
    return np.hstack([F1, F2])


def _rbf_kernel(X1, X2, gamma):
    """Compute RBF kernel matrix efficiently."""
    X1_sq = np.sum(X1 ** 2, axis=1)
    X2_sq = np.sum(X2 ** 2, axis=1)
    d2 = X1_sq[:, None] + X2_sq[None, :] - 2 * X1 @ X2.T
    d2 = np.maximum(d2, 0)
    return np.exp(-gamma * d2)


def _fit_ridge(Xf, y, alpha):
    """Fit ridge regression and return weights."""
    p = Xf.shape[1]
    AtA = Xf.T @ Xf + alpha * np.eye(p)
    Aty = Xf.T @ y
    try:
        w = np.linalg.solve(AtA, Aty)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(AtA, Aty, rcond=None)[0]
    return w


def _loocv_ridge(Xf, y, alpha):
    """Efficient LOOCV for ridge regression using hat matrix."""
    n, p = Xf.shape
    AtA = Xf.T @ Xf + alpha * np.eye(p)
    try:
        AtA_inv = np.linalg.inv(AtA)
    except np.linalg.LinAlgError:
        AtA_inv = np.linalg.pinv(AtA)
    
    # Hat matrix H = X (X'X + alpha*I)^{-1} X'
    H = Xf @ AtA_inv @ Xf.T
    # LOOCV residuals
    y_hat = H @ y
    h_diag = np.diag(H)
    residuals = y - y_hat
    loocv_residuals = residuals / (1 - h_diag)
    loocv_mse = np.mean(loocv_residuals ** 2)
    return loocv_mse


def _fit_rbf_kernel(X_train, y_train, gamma, alpha):
    """Fit RBF kernel ridge regression."""
    K = _rbf_kernel(X_train, X_train, gamma)
    n = K.shape[0]
    K_reg = K + alpha * np.eye(n)
    try:
        w = np.linalg.solve(K_reg, y_train)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(K_reg, y_train, rcond=None)[0]
    
    def predict(X):
        K_test = _rbf_kernel(X, X_train, gamma)
        return K_test @ w
    
    y_pred = K @ w
    mse = np.mean((y_pred - y_train) ** 2)
    return mse, predict, w, K


def _loocv_rbf(X_train, y_train, gamma, alpha):
    """Efficient LOOCV for RBF kernel ridge regression."""
    K = _rbf_kernel(X_train, X_train, gamma)
    n = K.shape[0]
    K_reg = K + alpha * np.eye(n)
    try:
        K_reg_inv = np.linalg.inv(K_reg)
    except np.linalg.LinAlgError:
        K_reg_inv = np.linalg.pinv(K_reg)
    
    # Hat matrix H = K (K + alpha*I)^{-1}
    H = K @ K_reg_inv
    y_hat = H @ y_train
    h_diag = np.diag(H)
    residuals = y_train - y_hat
    loocv_residuals = residuals / (1 - h_diag)
    loocv_mse = np.mean(loocv_residuals ** 2)
    return loocv_mse


def _fit_gaussian_process(X_train, y_train, length_scale, noise_var, signal_var):
    """Fit a simple Gaussian Process regression."""
    X = X_train
    y = y_train
    n = len(X)
    
    # RBF kernel with given hyperparameters
    gamma = 1.0 / (2 * length_scale ** 2)
    K = _rbf_kernel(X, X, gamma)
    K += signal_var * np.eye(n)
    K += noise_var * np.eye(n)
    
    try:
        L = np.linalg.cholesky(K)
    except np.linalg.LinAlgError:
        K += 1e-6 * np.eye(n)
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return None, float('inf')
    
    alpha_vec = np.linalg.solve(L.T, np.linalg.solve(L, y))
    
    def predict(X_new):
        K_star = _rbf_kernel(X_new, X, gamma)
        return K_star @ alpha_vec
    
    y_pred = K @ alpha_vec
    mse = np.mean((y_pred - y) ** 2)
    return predict, mse


def _fit(X_train, y_train):
    """Main fitting function that tries multiple model families."""
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    
    d = X_train.shape[1]
    n = len(X_train)
    
    if d != 1:
        # Fallback for multi-dimensional
        cols = [np.ones(n)]
        for j in range(d):
            cols.append(X_train[:, j])
        for i in range(d):
            for j in range(i, d):
                cols.append(X_train[:, i] * X_train[:, j])
        Xf = np.column_stack(cols)
        w, *_ = np.linalg.lstsq(Xf, y_train, rcond=None)
        
        def predict(X):
            X = np.asarray(X, dtype=np.float64)
            cols = [np.ones(len(X))]
            for j in range(d):
                cols.append(X[:, j])
            for i in range(d):
                for j in range(i, d):
                    cols.append(X[:, i] * X[:, j])
            Xf_test = np.column_stack(cols)
            return Xf_test @ w
        return predict
    
    # 1D case: try multiple model families
    
    best_cv_mse = float('inf')
    best_predict = None
    best_name = "none"
    
    # Track time
    t_start = time.time()
    
    # 1. Try Chebyshev + Fourier with ridge regression
    for n_cheb in [10, 15, 20, 25]:
        for n_fourier in [5, 8, 12]:
            if _time_left() < 15.0:
                break
            for alpha in [1e-6, 1e-4, 1e-3, 1e-2, 0.1]:
                if _time_left() < 5.0:
                    break
                try:
                    Xf = _combined_features(X_train, n_cheb, n_fourier)
                    cv_mse = _loocv_ridge(Xf, y_train, alpha)
                    if cv_mse < best_cv_mse:
                        best_cv_mse = cv_mse
                        w = _fit_ridge(Xf, y_train, alpha)
                        
                        def make_pred(w, nc, nf):
                            def predict(X):
                                Xf_test = _combined_features(X, nc, nf)
                                return Xf_test @ w
                            return predict
                        best_predict = make_pred(w, n_cheb, n_fourier)
                        best_name = f"cheb_fourier({n_cheb},{n_fourier},a={alpha})"
                except Exception:
                    continue
    
    # 2. Try pure Chebyshev
    for n_cheb in [8, 12, 16, 20, 30]:
        if _time_left() < 10.0:
            break
        for alpha in [1e-6, 1e-4, 1e-3, 1e-2]:
            if _time_left() < 3.0:
                break
            try:
                Xf = _chebyshev_features(X_train, n_cheb)
                cv_mse = _loocv_ridge(Xf, y_train, alpha)
                if cv_mse < best_cv_mse:
                    best_cv_mse = cv_mse
                    w = _fit_ridge(Xf, y_train, alpha)
                    
                    def make_pred(w, nc):
                        def predict(X):
                            Xf_test = _chebyshev_features(X, nc)
                            return Xf_test @ w
                        return predict
                    best_predict = make_pred(w, n_cheb)
                    best_name = f"cheb({n_cheb},a={alpha})"
            except Exception:
                continue
    
    # 3. Try RBF kernel ridge regression
    for gamma in [0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0]:
        if _time_left() < 10.0:
            break
        for alpha in [1e-4, 1e-3, 1e-2, 0.1, 1.0]:
            if _time_left() < 3.0:
                break
            try:
                cv_mse = _loocv_rbf(X_train, y_train, gamma, alpha)
                if cv_mse < best_cv_mse:
                    best_cv_mse = cv_mse
                    train_mse, predict, w, K = _fit_rbf_kernel(X_train, y_train, gamma, alpha)
                    if predict is not None:
                        best_predict = predict
                        best_name = f"rbf(g={gamma},a={alpha})"
            except Exception:
                continue
    
    # 4. Try Gaussian Process with simple hyperparameter search
    for length_scale in [0.3, 0.5, 1.0, 2.0]:
        if _time_left() < 10.0:
            break
        for noise_var in [1e-4, 1e-3, 1e-2]:
            if _time_left() < 3.0:
                break
            try:
                predict, mse = _fit_gaussian_process(X_train, y_train, length_scale, noise_var, 1.0)
                if predict is not None and mse < best_cv_mse:
                    # Use training MSE as proxy for CV
                    best_cv_mse = mse
                    best_predict = predict
                    best_name = f"gp(ls={length_scale},nv={noise_var})"
            except Exception:
                continue
    
    # 5. Fallback: simple polynomial
    if best_predict is None:
        Xf = _chebyshev_features(X_train, 10)
        w, *_ = np.linalg.lstsq(Xf, y_train, rcond=None)
        
        def predict(X):
            Xf_test = _chebyshev_features(X, 10)
            return Xf_test @ w
        best_predict = predict
        best_name = "fallback_poly"
    
    # Ensure predictions are finite
    def safe_predict(X):
        X = np.asarray(X, dtype=np.float64)
        try:
            y_pred = best_predict(X)
            y_pred = np.asarray(y_pred, dtype=np.float64)
            # Replace non-finite values with 0
            y_pred = np.where(np.isfinite(y_pred), y_pred, 0.0)
            return y_pred
        except Exception:
            return np.zeros(len(X))
    
    return safe_predict


def fit(X_train, y_train):
    global _START_TIME
    _START_TIME = time.time()
    
    try:
        return _fit(X_train, y_train)
    except Exception:
        # Ultimate fallback
        X_train = np.asarray(X_train, dtype=np.float64)
        y_train = np.asarray(y_train, dtype=np.float64)
        n = len(X_train)
        mean_y = np.mean(y_train)
        
        def predict(X):
            X = np.asarray(X, dtype=np.float64)
            return np.full(len(X), mean_y)
        return predict
# EVOLVE-BLOCK-END
