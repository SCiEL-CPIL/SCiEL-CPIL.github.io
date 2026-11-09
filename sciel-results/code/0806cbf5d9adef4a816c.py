"""Symbolic regression via LOESS with adaptive bandwidth, combined with
Gaussian Process fallback and rich feature-space least squares."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


def fit(X_train, y_train):
    """Fit using LOESS (local polynomial regression) with adaptive bandwidth,
    combined with GP and feature-space least squares as fallbacks."""
    t0 = time.time()
    deadline = t0 + 280.0  # safety margin before 300s kill

    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape

    # Handle constant y
    y_var = np.var(y)
    if y_var < 1e-12:
        y_mean = np.mean(y)
        return lambda X_test: np.full(len(np.asarray(X_test).shape[0]), y_mean)

    if d == 1:
        x = X[:, 0].copy()
        order = np.argsort(x)
        x_sorted = x[order]
        y_sorted = y[order]
        x_range = x_sorted[-1] - x_sorted[0]
        if x_range < 1e-10:
            x_range = 1.0

        # --- Strategy 1: LOESS with adaptive bandwidth via GCV ---
        def loess_predict(X_test, bandwidth_frac, poly_deg=2):
            """LOESS prediction for 1D data."""
            X_test = np.asarray(X_test, dtype=np.float64)
            if X_test.ndim == 1:
                X_test = X_test.reshape(-1, 1)
            xt = X_test[:, 0]
            
            # Number of points to use in local window
            n_points = max(5, int(bandwidth_frac * n))
            preds = np.zeros(len(xt))
            
            for i in range(len(xt)):
                # Find nearest neighbors
                dists = np.abs(xt[i] - x_sorted)
                idx = np.argpartition(dists, n_points - 1)[:n_points]
                # Sort by distance for weighting
                idx = idx[np.argsort(dists[idx])]
                
                xi = x_sorted[idx]
                yi = y_sorted[idx]
                di = dists[idx]
                
                # Tricube weights
                max_d = di[-1] if di[-1] > 1e-10 else 1.0
                w = (1.0 - (di / max_d)**3)**3
                w[di > max_d] = 0.0
                if np.sum(w) < 1e-10:
                    preds[i] = np.mean(yi)
                    continue
                
                # Local polynomial fit (degree 2)
                x_local = xi - xt[i]
                # Build design matrix
                cols = [np.ones_like(x_local)]
                for k in range(1, poly_deg + 1):
                    cols.append(x_local**k)
                A = np.column_stack(cols)
                
                # Weighted least squares
                W = np.diag(w)
                try:
                    # Solve (A^T W A) beta = A^T W y
                    ATA = A.T @ (w[:, None] * A)
                    ATy = A.T @ (w * yi)
                    # Add small ridge for stability
                    ATA += 1e-10 * np.eye(poly_deg + 1)
                    beta = np.linalg.solve(ATA, ATy)
                    preds[i] = beta[0]  # value at x=0 (i.e., xt[i])
                except:
                    preds[i] = np.average(yi, weights=w)
            
            return preds

        def gcv_score(bandwidth_frac, poly_deg=2):
            """Generalized Cross-Validation score for LOESS."""
            try:
                preds = loess_predict(x_sorted.reshape(-1, 1), bandwidth_frac, poly_deg)
                if not np.all(np.isfinite(preds)):
                    return 1e10
                residuals = y_sorted - preds
                n_eff = bandwidth_frac * n
                h = 1.0 / n_eff  # effective smoothing parameter
                # GCV = (1/n) * sum(residuals^2) / (1 - h)^2
                mse = np.mean(residuals**2)
                gcv_val = mse / (1.0 - h)**2
                return gcv_val
            except:
                return 1e10

        # Search over bandwidth fractions
        best_bw = 0.3
        best_gcv = 1e10
        bw_candidates = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8]
        
        for bw in bw_candidates:
            if time.time() > deadline:
                break
            gcv_val = gcv_score(bw)
            if gcv_val < best_gcv:
                best_gcv = gcv_val
                best_bw = bw

        # --- Strategy 2: Gaussian Process with RBF kernel ---
        def rbf_kernel(x1, x2, length_scale):
            sq_dists = (x1[:, None] - x2[None, :])**2
            return np.exp(-sq_dists / (2 * length_scale**2))

        def gp_predict(X_test, ls, sv, noise):
            X_test = np.asarray(X_test, dtype=np.float64)
            if X_test.ndim == 1:
                X_test = X_test.reshape(-1, 1)
            xt = X_test[:, 0]
            K_star = rbf_kernel(xt, x_sorted, ls) * sv
            return K_star @ alpha_gp

        # Fit GP
        K = rbf_kernel(x_sorted, x_sorted, x_range / 3.0) * max(y_var, 1e-6)
        K += 1e-6 * np.eye(n)
        try:
            L = np.linalg.cholesky(K)
            alpha_gp = np.linalg.solve(L.T, np.linalg.solve(L, y_sorted))
            gp_ls = x_range / 3.0
            gp_sv = max(y_var, 1e-6)
            gp_noise = 1e-6
        except:
            alpha_gp = None
            gp_ls = gp_sv = gp_noise = None

        # --- Strategy 3: Feature-space least squares ---
        def build_features_1d(x):
            feats = [np.ones_like(x), x.copy()]
            for deg in range(2, 10):
                feats.append(x**deg)
            feats.append(np.sin(x))
            feats.append(np.cos(x))
            feats.append(np.sin(2 * x))
            feats.append(np.cos(2 * x))
            feats.append(np.sin(3 * x))
            feats.append(np.cos(3 * x))
            feats.append(np.tanh(x))
            feats.append(np.tanh(2 * x))
            feats.append(np.abs(x))
            with np.errstate(divide='ignore', invalid='ignore'):
                inv = np.where(np.abs(x) > 1e-8, 1.0 / x, 0.0)
            feats.append(inv)
            with np.errstate(divide='ignore', invalid='ignore'):
                logx = np.where(x > 1e-8, np.log(x), 0.0)
            feats.append(logx)
            return np.column_stack(feats)

        F_train = build_features_1d(x_sorted)
        try:
            w_ls, *_ = np.linalg.lstsq(F_train, y_sorted, rcond=None)
        except:
            w_ls = None

        # Evaluate all strategies on training data and pick best
        candidates = []

        # LOESS
        try:
            preds_loess = loess_predict(x_sorted.reshape(-1, 1), best_bw)
            if np.all(np.isfinite(preds_loess)):
                mse_loess = np.mean((preds_loess - y_sorted)**2)
                candidates.append(('loess', mse_loess, best_bw))
        except:
            pass

        # GP
        if alpha_gp is not None:
            try:
                preds_gp = gp_predict(x_sorted.reshape(-1, 1), gp_ls, gp_sv, gp_noise)
                if np.all(np.isfinite(preds_gp)):
                    mse_gp = np.mean((preds_gp - y_sorted)**2)
                    candidates.append(('gp', mse_gp, (gp_ls, gp_sv, gp_noise)))
            except:
                pass

        # Least squares
        if w_ls is not None:
            try:
                preds_ls = F_train @ w_ls
                if np.all(np.isfinite(preds_ls)):
                    mse_ls = np.mean((preds_ls - y_sorted)**2)
                    candidates.append(('ls', mse_ls, w_ls))
            except:
                pass

        # Fallback: interpolation
        from scipy.interpolate import interp1d
        try:
            f_interp = interp1d(x_sorted, y_sorted, kind='cubic', fill_value='extrapolate')
            preds_interp = f_interp(x_sorted)
            if np.all(np.isfinite(preds_interp)):
                mse_interp = np.mean((preds_interp - y_sorted)**2)
                candidates.append(('interp', mse_interp, f_interp))
        except:
            pass

        if not candidates:
            # Last resort: constant
            y_mean = np.mean(y_sorted)
            return lambda X_test: np.full(len(np.asarray(X_test).shape[0]), y_mean)

        # Pick best strategy
        candidates.sort(key=lambda c: c[1])
        best_strategy, best_mse, best_params = candidates[0]

        def predict_final(X_test):
            X_test = np.asarray(X_test, dtype=np.float64)
            if X_test.ndim == 1:
                X_test = X_test.reshape(-1, 1)
            xt = X_test[:, 0]
            
            if best_strategy == 'loess':
                return loess_predict(X_test, best_bw)
            elif best_strategy == 'gp':
                ls, sv, noise = best_params
                K_star = rbf_kernel(xt, x_sorted, ls) * sv
                return K_star @ alpha_gp
            elif best_strategy == 'ls':
                F_test = build_features_1d(xt)
                return F_test @ best_params
            elif best_strategy == 'interp':
                return best_params(xt)
            else:
                return np.full(len(xt), np.mean(y_sorted))

        return predict_final

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
