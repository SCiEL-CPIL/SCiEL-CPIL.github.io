"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve


def fit(X_train, y_train):
    """Fit using Gaussian Process regression for 1D data,
    with a polynomial fallback for multi-dimensional inputs."""
    t0 = time.time()
    deadline = 325.0  # safety margin before 330s kill

    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape

    if d == 1:
        x = X[:, 0].copy()
        # Sort by x
        order = np.argsort(x)
        x_sorted = x[order]
        y_sorted = y[order]

        # GP with RBF kernel: k(x, x') = sigma_f^2 * exp(-0.5 * (x - x')^2 / l^2)
        # For noiseless data, we can use exact interpolation with small noise for numerical stability
        
        # Optimize hyperparameters: length_scale l, signal variance sigma_f
        # Use log-space for optimization
        x_range = x_sorted[-1] - x_sorted[0] if x_sorted[-1] > x_sorted[0] else 1.0
        
        def neg_log_likelihood(log_params):
            log_l = log_params[0]
            log_sigma_f = log_params[1]
            
            l = np.exp(log_l)
            sigma_f = np.exp(log_sigma_f)
            
            # Build kernel matrix
            diff = x_sorted[:, None] - x_sorted[None, :]
            K = sigma_f**2 * np.exp(-0.5 * (diff / l)**2)
            
            # Add small noise for numerical stability
            K += 1e-8 * np.eye(n)
            
            try:
                L, _ = cho_factor(K)
                alpha = cho_solve((L, True), y_sorted)
                # Marginal log likelihood: -0.5 * y^T K^-1 y - 0.5 * log det(K) - n/2 * log(2*pi)
                log_det = 2.0 * np.sum(np.log(np.diag(L)))
                nll = 0.5 * y_sorted @ alpha + 0.5 * log_det + 0.5 * n * np.log(2 * np.pi)
                return nll
            except:
                return 1e10

        # Initial guess: length_scale ~ range/5, sigma_f ~ std(y)
        y_std = np.std(y_sorted)
        if y_std < 1e-10:
            y_std = 1.0
        log_l0 = np.log(x_range / 5.0)
        log_sf0 = np.log(y_std)
        
        best_params = np.array([log_l0, log_sf0])
        best_nll = neg_log_likelihood(best_params)
        
        # Try several starting points
        starts = [
            [np.log(x_range * f) for f in [0.1, 0.2, 0.5, 1.0, 2.0]] + [[log_sf0] for _ in range(5)]
        ]
        start_list = []
        for f in [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]:
            start_list.append([np.log(x_range * f), log_sf0])
        for sf_mult in [0.5, 2.0, 5.0]:
            start_list.append([log_l0, log_sf0 + np.log(sf_mult)])
        
        for start in start_list:
            if time.time() > deadline:
                break
            try:
                res = minimize(neg_log_likelihood, np.array(start), method='L-BFGS-B',
                              options={'maxiter': 50, 'ftol': 1e-8})
                if res.fun < best_nll:
                    best_nll = res.fun
                    best_params = res.x
            except:
                pass

        # Use best parameters
        log_l, log_sigma_f = best_params
        l = np.exp(log_l)
        sigma_f = np.exp(log_sigma_f)
        
        # Build final kernel and Cholesky factorization
        diff = x_sorted[:, None] - x_sorted[None, :]
        K = sigma_f**2 * np.exp(-0.5 * (diff / l)**2)
        K += 1e-8 * np.eye(n)
        
        try:
            L, _ = cho_factor(K)
            alpha = cho_solve((L, True), y_sorted)
            gp_fitted = True
        except:
            gp_fitted = False
            # Fallback to interpolation
            from scipy.interpolate import interp1d
            f_interp = interp1d(x_sorted, y_sorted, kind='cubic', fill_value='extrapolate')
            def predict_interp(X_test):
                X_test = np.asarray(X_test, dtype=np.float64)
                if X_test.ndim == 1:
                    X_test = X_test.reshape(-1, 1)
                return f_interp(X_test[:, 0])
            return predict_interp

        def predict_gp(X_test):
            X_test = np.asarray(X_test, dtype=np.float64)
            if X_test.ndim == 1:
                X_test = X_test.reshape(-1, 1)
            xt = X_test[:, 0]
            
            # Compute kernel between test points and training points
            diff_test = xt[:, None] - x_sorted[None, :]
            K_star = sigma_f**2 * np.exp(-0.5 * (diff_test / l)**2)
            
            # Prediction: K_star @ alpha
            preds = K_star @ alpha
            return preds

        return predict_gp

    else:
        # Multi-dimensional fallback: polynomial regression
        def features(X):
            cols = [np.ones(len(X))]
            for j in range(X.shape[1]):
                cols.append(X[:, j])
            for i in range(X.shape[1]):
                for j in range(i, X.shape[1]):
                    cols.append(X[:, i] * X[:, j])
            return np.column_stack(cols)

        w, *_ = np.linalg.lstsq(features(X), y, rcond=None)

        def predict_poly(X_test):
            X_test = np.asarray(X_test, dtype=np.float64)
            if X_test.ndim == 1:
                X_test = X_test.reshape(-1, d)
            return features(X_test) @ w

        return predict_poly

# EVOLVE-BLOCK-END
