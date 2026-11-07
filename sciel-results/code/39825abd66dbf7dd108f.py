"""Symbolic regression over 10 hidden equations: multi-stage oracle selection."""
# EVOLVE-BLOCK-START
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def _cholesky_diag_inverse(K_reg, y):
    """Compute LOO predictions efficiently using Cholesky decomposition.
    Returns (loo_predictions, alpha) where alpha = K_reg^{-1} y.
    """
    n = len(y)
    L = np.linalg.cholesky(K_reg)
    # Solve L v = I to get columns of L^{-1}; we need diag(L^{-T} L^{-1})
    # More efficiently: solve L z = e_i for each i, but that's O(n^3)
    # Alternative: compute alpha = K_reg^{-1} y via two triangular solves
    Ly = np.linalg.solve(L, y)
    alpha = np.linalg.solve(L.T, Ly)
    # For LOO: yhat_loo_i = y_i - K_ii / (K_reg^{-1})_ii
    # (K_reg^{-1})_ii = e_i^T K_reg^{-1} e_i
    # We can compute this as: solve K_reg v_i = e_i, then v_i[i]
    # But that's n solves. Instead, use: if we have L, then
    # K_reg^{-1} = L^{-T} L^{-1}
    # Let W = L^{-1} (solved via L W = I), then (K_reg^{-1})_ii = sum_k W[k,i]^2
    # Computing full L^{-1} is O(n^3) but for n=200 it's fine
    W = np.linalg.solve(L, np.eye(n))
    diag_inv = np.sum(W ** 2, axis=0)
    return alpha, diag_inv


def _loocv_krr(X, y, kernel_fn, alpha_grid):
    """LOO-CV for kernel ridge regression. Returns (best_mse, best_alpha, K_train)."""
    n = len(y)
    K = kernel_fn(X, X)
    best_mse = np.inf
    best_alpha = None
    
    for alpha in alpha_grid:
        K_reg = K + alpha * np.eye(n)
        try:
            alpha_vec, diag_inv = _cholesky_diag_inverse(K_reg, y)
            yhat_loo = y - np.diag(K) * diag_inv
            mse = np.mean((y - yhat_loo) ** 2)
            if mse < best_mse:
                best_mse = mse
                best_alpha = alpha
        except (np.linalg.LinAlgError, ValueError):
            continue
    return best_mse, best_alpha, K


def _loocv_gp(X, y, length_scale_grid, variance_grid, noise_grid):
    """LOO-CV for Matern-5/2 GP. Returns (best_mse, best_params)."""
    n = len(y)
    best_mse = np.inf
    best_params = None
    
    for ls in length_scale_grid:
        for var in variance_grid:
            for noise in noise_grid:
                X_sq = np.sum(X ** 2, axis=1)
                sq_dists = X_sq[:, None] + X_sq[None, :] - 2 * X @ X.T
                sq_dists = np.maximum(sq_dists, 0)
                dists = np.sqrt(sq_dists)
                r = dists / ls
                K_prior = var * (1 + r + r ** 2 / 3) * np.exp(-r)
                K_obs = K_prior + noise * np.eye(n)
                try:
                    alpha_vec, diag_inv = _cholesky_diag_inverse(K_obs, y)
                    yhat_loo = y - var * diag_inv
                    mse = np.mean((y - yhat_loo) ** 2)
                    if mse < best_mse:
                        best_mse = mse
                        best_params = (ls, var, noise)
                except (np.linalg.LinAlgError, ValueError):
                    continue
    return best_mse, best_params


def _loocv_rational(X, y, alpha_grid):
    """LOO-CV for rational function approximator: y ≈ (a^T φ(x)) / (b^T φ(x))
    where φ includes intercept and linear terms. We fit numerator and denominator
    separately using ridge regression, then compute LOO via the hat matrix approach.
    
    Actually, for a true rational model, LOO is tricky. Instead, we'll use a
    weighted approach: fit numerator with weights 1/denom, but that's circular.
    
    Better approach: Use a product/sum basis that can approximate rational functions:
    φ(x) = [1, x_1, ..., x_d, 1/x_1, ..., 1/x_d, x_i*x_j, ...]
    This captures common rational patterns without the nonlinearity of a true ratio.
    """
    n, d = X.shape
    # Build rich basis: intercept, linear, inverse linear, quadratic, cross terms
    cols = [np.ones(n)]
    for j in range(d):
        cols.append(X[:, j])
    for j in range(d):
        # Avoid division by zero; inputs are in [1, ...] so safe
        cols.append(1.0 / np.maximum(X[:, j], 0.1))
    # Quadratic and cross terms
    for i in range(d):
        for j in range(i, d):
            cols.append(X[:, i] * X[:, j])
    # Inverse quadratic
    for j in range(d):
        cols.append(1.0 / np.maximum(X[:, j], 0.1) ** 2)
    
    B = np.column_stack(cols)
    m = B.shape[1]
    best_mse = np.inf
    best_alpha = None
    
    for alpha in alpha_grid:
        A = B.T @ B + alpha * np.eye(m)
        b = B.T @ y
        try:
            w = np.linalg.solve(A, b)
            yhat = B @ w
            # Hat matrix for ridge: H = B (B^T B + alpha I)^{-1} B^T
            # We already have A = B^T B + alpha I, so H = B A^{-1} B^T
            A_inv = np.linalg.inv(A)
            H = B @ A_inv @ B.T
            h_ii = np.diag(H)
            h_ii_safe = np.where(np.abs(1 - h_ii) < 1e-10, 1e-10, 1 - h_ii)
            yhat_loo = y - (y - yhat) / h_ii_safe
            mse = np.mean((y - yhat_loo) ** 2)
            if mse < best_mse:
                best_mse = mse
                best_alpha = alpha
        except (np.linalg.LinAlgError, ValueError):
            continue
    return best_mse, best_alpha, B


def fit(X_train, y_train):
    X = np.array(X_train, dtype=np.float64)
    y = np.array(y_train, dtype=np.float64)
    n, d = X.shape
    
    # Standardize inputs
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std < 1e-10] = 1.0
    X_norm = (X - X_mean) / X_std
    
    models = []
    cv_scores = []
    
    # Model 1: Polynomial KRR degree 2
    def poly_kernel_deg2(X1, X2, deg=2):
        return (1 + X1 @ X2.T / d) ** deg
    
    alpha_grid_krr = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    mse1, alpha1, K1 = _loocv_krr(X_norm, y, lambda a, b: poly_kernel_deg2(a, b, 2), alpha_grid_krr)
    if mse1 < np.inf and alpha1 is not None:
        K_reg = K1 + alpha1 * np.eye(n)
        w = np.linalg.solve(K_reg, y)
        def pred_krr2(X_test):
            X_test = np.array(X_test, dtype=np.float64)
            X_test_norm = (X_test - X_mean) / X_std
            K_test = poly_kernel_deg2(X_test_norm, X_norm, 2)
            return K_test @ w
        models.append(('krr_poly2', pred_krr2))
        cv_scores.append(mse1)
    
    # Model 2: Polynomial KRR degree 3
    mse2, alpha2, K2 = _loocv_krr(X_norm, y, lambda a, b: poly_kernel_deg2(a, b, 3), alpha_grid_krr)
    if mse2 < np.inf and alpha2 is not None:
        K_reg = K2 + alpha2 * np.eye(n)
        w = np.linalg.solve(K_reg, y)
        def pred_krr3(X_test):
            X_test = np.array(X_test, dtype=np.float64)
            X_test_norm = (X_test - X_mean) / X_std
            K_test = poly_kernel_deg2(X_test_norm, X_norm, 3)
            return K_test @ w
        models.append(('krr_poly3', pred_krr3))
        cv_scores.append(mse2)
    
    # Model 3: Polynomial KRR degree 4
    mse3, alpha3, K3 = _loocv_krr(X_norm, y, lambda a, b: poly_kernel_deg2(a, b, 4), alpha_grid_krr)
    if mse3 < np.inf and alpha3 is not None:
        K_reg = K3 + alpha3 * np.eye(n)
        w = np.linalg.solve(K_reg, y)
        def pred_krr4(X_test):
            X_test = np.array(X_test, dtype=np.float64)
            X_test_norm = (X_test - X_mean) / X_std
            K_test = poly_kernel_deg2(X_test_norm, X_norm, 4)
            return K_test @ w
        models.append(('krr_poly4', pred_krr4))
        cv_scores.append(mse3)
    
    # Model 4: Matern-5/2 Gaussian Process
    length_scale_grid = [0.2, 0.4, 0.7, 1.0, 1.5, 2.5]
    variance_grid = [0.5, 1.0, 2.0, 5.0]
    noise_grid = [0.001, 0.01, 0.1, 1.0]
    mse4, gp_params = _loocv_gp(X_norm, y, length_scale_grid, variance_grid, noise_grid)
    if mse4 < np.inf and gp_params is not None:
        ls, var, noise = gp_params
        X_sq = np.sum(X_norm ** 2, axis=1)
        sq_dists = X_sq[:, None] + X_sq[None, :] - 2 * X_norm @ X_norm.T
        sq_dists = np.maximum(sq_dists, 0)
        dists = np.sqrt(sq_dists)
        r = dists / ls
        K_prior = var * (1 + r + r ** 2 / 3) * np.exp(-r)
        K_obs = K_prior + noise * np.eye(n)
        L = np.linalg.cholesky(K_obs)
        alpha_gp = np.linalg.solve(L.T, np.linalg.solve(L, y))
        
        def pred_gp(X_test):
            X_test = np.array(X_test, dtype=np.float64)
            X_test_norm = (X_test - X_mean) / X_std
            X_test_sq = np.sum(X_test_norm ** 2, axis=1)
            sq_dists_test = X_test_sq[:, None] + X_sq[None, :] - 2 * X_test_norm @ X_norm.T
            sq_dists_test = np.maximum(sq_dists_test, 0)
            dists_test = np.sqrt(sq_dists_test)
            r_test = dists_test / ls
            K_test = var * (1 + r_test + r_test ** 2 / 3) * np.exp(-r_test)
            return K_test @ alpha_gp
        models.append(('gp_matern52', pred_gp))
        cv_scores.append(mse4)
    
    # Model 5: Rational-style basis (intercept, linear, inverse, quadratic, inverse quadratic)
    alpha_grid_rat = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    mse5, alpha5, B5 = _loocv_rational(X, y, alpha_grid_rat)
    if mse5 < np.inf and alpha5 is not None:
        A5 = B5.T @ B5 + alpha5 * np.eye(B5.shape[1])
        w5 = np.linalg.solve(A5, B5.T @ y)
        
        def build_rational_basis(X_in):
            cols = [np.ones(len(X_in))]
            for j in range(X_in.shape[1]):
                cols.append(X_in[:, j])
            for j in range(X_in.shape[1]):
                cols.append(1.0 / np.maximum(X_in[:, j], 0.1))
            for i in range(X_in.shape[1]):
                for j in range(i, X_in.shape[1]):
                    cols.append(X_in[:, i] * X_in[:, j])
            for j in range(X_in.shape[1]):
                cols.append(1.0 / np.maximum(X_in[:, j], 0.1) ** 2)
            return np.column_stack(cols)
        
        def pred_rat(X_test):
            X_test = np.array(X_test, dtype=np.float64)
            B_test = build_rational_basis(X_test)
            return B_test @ w5
        models.append(('rational_basis', pred_rat))
        cv_scores.append(mse5)
    
    # Model 6: Linear Ridge (simple baseline)
    alpha_grid_lin = [0.001, 0.01, 0.1, 1.0, 10.0]
    B_lin = np.column_stack([np.ones(n), X_norm])
    m_lin = B_lin.shape[1]
    mse6 = np.inf
    best_alpha_lin = None
    for alpha in alpha_grid_lin:
        A = B_lin.T @ B_lin + alpha * np.eye(m_lin)
        b = B_lin.T @ y
        try:
            w = np.linalg.solve(A, b)
            yhat = B_lin @ w
            A_inv = np.linalg.inv(A)
            H = B_lin @ A_inv @ B_lin.T
            h_ii = np.diag(H)
            h_ii_safe = np.where(np.abs(1 - h_ii) < 1e-10, 1e-10, 1 - h_ii)
            yhat_loo = y - (y - yhat) / h_ii_safe
            mse = np.mean((y - yhat_loo) ** 2)
            if mse < mse6:
                mse6 = mse
                best_alpha_lin = alpha
        except (np.linalg.LinAlgError, ValueError):
            continue
    if mse6 < np.inf and best_alpha_lin is not None:
        A = B_lin.T @ B_lin + best_alpha_lin * np.eye(m_lin)
        w = np.linalg.solve(A, B_lin.T @ y)
        def pred_lin(X_test):
            X_test = np.array(X_test, dtype=np.float64)
            X_test_norm = (X_test - X_mean) / X_std
            B_test = np.column_stack([np.ones(len(X_test)), X_test_norm])
            return B_test @ w
        models.append(('linear_ridge', pred_lin))
        cv_scores.append(mse6)
    
    # Select best model based on LOO CV score
    if not models:
        y_mean = np.mean(y)
        return lambda X: np.full(len(X), y_mean)
    
    cv_scores = np.array(cv_scores)
    best_idx = np.argmin(cv_scores)
    best_predict_fn = models[best_idx][1]
    best_mse = cv_scores[best_idx]
    
    # Check if we should ensemble: if top-2 models are within 20% of each other,
    # average their predictions for robustness
    sorted_indices = np.argsort(cv_scores)
    if len(sorted_indices) >= 2 and cv_scores[sorted_indices[1]] < 1.2 * cv_scores[sorted_indices[0]]:
        # Ensemble top 2
        pred_fn1 = models[sorted_indices[0]][1]
        pred_fn2 = models[sorted_indices[1]][1]
        def predict_ensemble(X_test):
            X_test = np.array(X_test, dtype=np.float64)
            try:
                p1 = pred_fn1(X_test)
                p2 = pred_fn2(X_test)
                p1 = np.where(np.isfinite(p1), p1, np.mean(y))
                p2 = np.where(np.isfinite(p2), p2, np.mean(y))
                return 0.5 * p1 + 0.5 * p2
            except Exception:
                return np.full(len(X_test), np.mean(y))
        return predict_ensemble
    
    def predict(X_test):
        X_test = np.array(X_test, dtype=np.float64)
        try:
            preds = best_predict_fn(X_test)
            preds = np.where(np.isfinite(preds), preds, np.mean(y))
            return preds
        except Exception:
            return np.full(len(X_test), np.mean(y))
    
    return predict
# EVOLVE-BLOCK-END
