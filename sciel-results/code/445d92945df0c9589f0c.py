"""Symbolic regression using Gaussian Process with Cholesky decomposition for noiseless 1D data."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings
from scipy.optimize import minimize

warnings.filterwarnings('ignore')


def _safe_predict(predict_fn, X):
    """Ensure predictions are finite."""
    X = np.asarray(X, dtype=float)
    try:
        pred = predict_fn(X)
    except Exception:
        return np.zeros(len(np.asarray(X).reshape(-1)))
    pred = np.asarray(pred, dtype=float)
    pred = np.where(np.isfinite(pred), pred, 0.0)
    return pred


def _rbf_kernel(x1, x2, length_scale, signal_var):
    """Compute RBF kernel matrix between two 1D arrays."""
    x1 = np.asarray(x1, dtype=float).reshape(-1, 1)
    x2 = np.asarray(x2, dtype=float).reshape(-1, 1)
    sq_dist = np.sum(x1**2, axis=1)[:, None] + np.sum(x2**2, axis=1)[None, :] - 2.0 * x1 @ x2.T
    sq_dist = np.maximum(sq_dist, 0.0)
    return signal_var * np.exp(-sq_dist / (2.0 * length_scale**2))


def _matern_kernel(x1, x2, length_scale, signal_var, nu=2.5):
    """Compute Matern kernel matrix."""
    x1 = np.asarray(x1, dtype=float).reshape(-1, 1)
    x2 = np.asarray(x2, dtype=float).reshape(-1, 1)
    sq_dist = np.sum(x1**2, axis=1)[:, None] + np.sum(x2**2, axis=1)[None, :] - 2.0 * x1 @ x2.T
    dist = np.sqrt(np.maximum(sq_dist, 0.0))
    if nu == 2.5:
        sqrt3 = np.sqrt(3.0)
        K = (1.0 + sqrt3 * dist / length_scale) * np.exp(-sqrt3 * dist / length_scale)
    elif nu == 1.5:
        sqrt3 = np.sqrt(3.0)
        K = (1.0 + sqrt3 * dist / length_scale) * np.exp(-sqrt3 * dist / length_scale)
    elif nu == 0.5:
        K = np.exp(-dist / length_scale)
    else:
        K = np.exp(-sq_dist / (2.0 * length_scale**2))
    return signal_var * K


def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 290

    X = np.asarray(X_train, dtype=float).reshape(-1, 1)
    y = np.asarray(y_train, dtype=float).flatten()
    n = len(y)

    if n < 2:
        mean_y = np.mean(y) if n > 0 else 0.0
        return lambda X: _safe_predict(lambda x: np.full(len(np.asarray(x).reshape(-1)), mean_y), X)

    x_vals = X[:, 0].copy()
    y_vals = y.copy()

    if np.std(y_vals) < 1e-12:
        mean_y = np.mean(y_vals)
        return lambda X: _safe_predict(lambda x: np.full(len(np.asarray(x).reshape(-1)), mean_y), X)

    y_var = np.var(y_vals)
    if y_var < 1e-15:
        mean_y = np.mean(y_vals)
        return lambda X: _safe_predict(lambda x: np.full(len(np.asarray(x).reshape(-1)), mean_y), X)

    x_min, x_max = np.min(x_vals), np.max(x_vals)
    x_range = x_max - x_min
    if x_range < 1e-10:
        x_range = 1.0

    sort_idx = np.argsort(x_vals)
    x_sorted = x_vals[sort_idx]
    y_sorted = y_vals[sort_idx]

    best_model = None
    best_score = np.inf

    def _optimize_gp(kernel_type):
        """Optimize GP hyperparameters for given kernel type."""
        def _get_kernel(x1, x2, length_scale, signal_var):
            if kernel_type == 'rbf':
                return _rbf_kernel(x1, x2, length_scale, signal_var)
            elif kernel_type == 'matern25':
                return _matern_kernel(x1, x2, length_scale, signal_var, nu=2.5)
            elif kernel_type == 'matern15':
                return _matern_kernel(x1, x2, length_scale, signal_var, nu=1.5)
            else:
                return _rbf_kernel(x1, x2, length_scale, signal_var)

        def _neg_log_marginal_likelihood(params):
            if time.time() > deadline - 5:
                return 1e10

            log_ls, log_sv, log_nv = params
            length_scale = np.exp(log_ls)
            signal_var = np.exp(log_sv)
            noise_var = np.exp(log_nv)

            if length_scale < 1e-8 or signal_var < 1e-12 or noise_var < 1e-12:
                return 1e10

            try:
                K = _get_kernel(x_sorted, x_sorted, length_scale, signal_var)
                K[np.diag_indices_from(K)] += noise_var

                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L, y_sorted)
                alpha = np.linalg.solve(L.T, alpha)

                log_lik = -0.5 * y_sorted @ alpha - np.sum(np.log(np.diag(L))) - 0.5 * n * np.log(2 * np.pi)

                if not np.isfinite(log_lik):
                    return 1e10

                return -log_lik
            except Exception:
                return 1e10

        best_neg_ll = np.inf
        best_params = None

        ls_candidates = [x_range * f for f in [0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]]
        sv_candidates = [y_var * f for f in [0.1, 1.0, 10.0]]
        nv_candidates = [1e-12, 1e-10, 1e-8]

        max_inits = 20
        n_inits = 0

        for ls_c in ls_candidates:
            for sv_c in sv_candidates:
                for nv_c in nv_candidates:
                    if n_inits >= max_inits or time.time() > deadline - 30:
                        break

                    x_init = np.log([max(ls_c, 1e-6), max(sv_c, 1e-10), max(nv_c, 1e-15)])
                    try:
                        result = minimize(
                            _neg_log_marginal_likelihood,
                            x_init,
                            method='L-BFGS-B',
                            bounds=[(np.log(1e-6), np.log(x_range * 10)),
                                    (np.log(1e-10), np.log(y_var * 100)),
                                    (np.log(1e-15), np.log(max(y_var, 1e-6) * 0.1))],
                            options={'maxiter': 40, 'ftol': 1e-6, 'gtol': 1e-4}
                        )
                        if result.fun < best_neg_ll:
                            best_neg_ll = result.fun
                            best_params = result.x.copy()
                    except Exception:
                        pass
                    n_inits += 1

                if time.time() > deadline - 30:
                    break
            if time.time() > deadline - 30:
                break

        if best_params is not None:
            best_ls = np.clip(np.exp(best_params[0]), 1e-4, x_range * 10)
            best_sv = np.clip(np.exp(best_params[1]), 1e-6, y_var * 100)
            best_nv = np.clip(np.exp(best_params[2]), 1e-12, max(y_var, 1e-6) * 0.1)

            try:
                K = _get_kernel(x_sorted, x_sorted, best_ls, best_sv)
                K[np.diag_indices_from(K)] += best_nv
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L, y_sorted)
                alpha = np.linalg.solve(L.T, alpha)

                pred_train = _get_kernel(x_sorted, x_sorted, best_ls, best_sv) @ alpha
                mse_train = np.mean((pred_train - y_sorted)**2)
                nrmse_train = mse_train / y_var if y_var > 1e-15 else mse_train

                return {
                    'type': kernel_type,
                    'ls': best_ls,
                    'sv': best_sv,
                    'nv': best_nv,
                    'alpha': alpha,
                    'score': nrmse_train,
                    'neg_ll': best_neg_ll
                }
            except Exception:
                return None
        return None

    # Try RBF kernel
    rbf_model = _optimize_gp('rbf')
    if rbf_model is not None:
        best_model = rbf_model
        best_score = rbf_model['score']

    # Try Matern 2.5 kernel
    if time.time() < deadline - 60:
        matern25_model = _optimize_gp('matern25')
        if matern25_model is not None and matern25_model['score'] < best_score:
            best_model = matern25_model
            best_score = matern25_model['score']

    # Try Matern 1.5 kernel
    if time.time() < deadline - 30:
        matern15_model = _optimize_gp('matern15')
        if matern15_model is not None and matern15_model['score'] < best_score:
            best_model = matern15_model
            best_score = matern15_model['score']

    # If no GP model succeeded, fall back to polynomial fit
    if best_model is None:
        mean_y = np.mean(y_vals)
        try:
            deg = min(5, n - 1)
            pc = np.polyfit(x_vals, y_vals, deg)
            def predict(X_pred):
                X_pred = np.asarray(X_pred, dtype=float).reshape(-1, 1)[:, 0]
                try:
                    pred = np.polyval(pc, X_pred)
                    pred = np.where(np.isfinite(pred), pred, mean_y)
                    return pred
                except Exception:
                    return np.full(len(X_pred), mean_y)
            return predict
        except Exception:
            return lambda X: _safe_predict(lambda x: np.full(len(np.asarray(x).reshape(-1)), mean_y), X)

    # Create prediction function based on best model
    kernel_type = best_model['type']
    ls = best_model['ls']
    sv = best_model['sv']
    alpha = best_model['alpha']

    def _get_kernel_for_predict(x1, x2, length_scale, signal_var):
        if kernel_type == 'rbf':
            return _rbf_kernel(x1, x2, length_scale, signal_var)
        elif kernel_type == 'matern25':
            return _matern_kernel(x1, x2, length_scale, signal_var, nu=2.5)
        elif kernel_type == 'matern15':
            return _matern_kernel(x1, x2, length_scale, signal_var, nu=1.5)
        else:
            return _rbf_kernel(x1, x2, length_scale, signal_var)

    def predict(X_pred):
        X_pred = np.asarray(X_pred, dtype=float).reshape(-1, 1)[:, 0]
        try:
            K_star = _get_kernel_for_predict(X_pred, x_sorted, ls, sv)
            pred_mean = K_star @ alpha
            pred_mean = np.where(np.isfinite(pred_mean), pred_mean, np.mean(y_sorted))
            return pred_mean
        except Exception:
            try:
                pred = np.interp(X_pred, x_sorted, y_sorted)
                pred = np.where(np.isfinite(pred), pred, np.mean(y_sorted))
                return pred
            except Exception:
                return np.full(len(X_pred), np.mean(y_sorted))

    return predict
# EVOLVE-BLOCK-END
