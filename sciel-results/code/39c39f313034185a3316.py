"""
Program: Ensemble GPR denoising + weighted degree-5 polynomial fit.

Strategy:
1. Standardize X and Y for numerical stability.
2. Estimate noise variance robustly.
3. Use ensemble of GPRs with different RBF length scales to estimate E[Y|X].
4. Weight each GPR by its validation performance.
5. Fit a weighted degree-5 polynomial to the ensemble pseudo-labels.
6. Transform coefficients back to original scale.
"""
import numpy as np
import scipy.linalg


def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-5 polynomial to the unresolved subgrid tendency.

    Parameters:
        training_x: array of local slow-state values x
        training_subgrid_tendency: array of corresponding unresolved tendencies

    Returns:
        coeff: array of 6 coefficients in ascending powers of x.
    """
    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)

    n = len(x)
    if n == 0:
        return np.zeros(6)

    # Direct fit for very small samples
    if n < 20:
        try:
            coeff = np.polynomial.polynomial.polyfit(x, y, 5)
            coeff = np.asarray(coeff, dtype=np.float64)
            if np.all(np.isfinite(coeff)):
                return coeff
        except Exception:
            pass
        return np.zeros(6)

    # Standardize X and Y
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-10:
        x_std = 1.0
    y_mean = np.mean(y)
    y_std = np.std(y)
    if y_std < 1e-10:
        y_std = 1.0

    x_norm = (x - x_mean) / x_std
    y_norm = (y - y_mean) / y_std

    # Estimate noise variance in normalized space using robust local regression
    sorted_idx = np.argsort(x_norm)
    x_sorted = x_norm[sorted_idx]
    y_sorted = y_norm[sorted_idx]

    window = max(10, n // 20)
    noise_var_estimates = []
    step = max(1, window // 4)
    for i in range(0, n - window, step):
        xw = x_sorted[i:i + window]
        yw = y_sorted[i:i + window]
        if len(np.unique(xw)) >= 2:
            try:
                p1 = np.polyfit(xw, yw, 1)
                resid = yw - np.polyval(p1, xw)
                v = np.var(resid)
                if v > 0:
                    noise_var_estimates.append(v)
            except Exception:
                pass

    if noise_var_estimates:
        noise_var = np.median(noise_var_estimates)
    else:
        noise_var = 0.1
    noise_var = max(noise_var, 1e-6)
    
    # Signal variance in normalized space is 1.0
    sigma_f2 = 1.0

    # Length scale candidates in normalized space
    # Use a broader range to capture different smoothness scales
    length_scales = np.array([0.15, 0.25, 0.4, 0.6, 0.9, 1.4, 2.2, 3.5])

    # Cross-Validation to evaluate each length scale
    # Use 5-fold CV
    n_folds = 5
    indices = np.arange(n)
    np.random.shuffle(indices)
    fold_sizes = np.full(n_folds, n // n_folds)
    fold_sizes[:n % n_folds] += 1
    folds = []
    current = 0
    for fs in fold_sizes:
        folds.append(indices[current:current + fs])
        current += fs

    # Store CV MSE for each length scale
    cv_mse_per_ls = {}

    for ls in length_scales:
        cv_mses = []
        for i in range(n_folds):
            val_idx = folds[i]
            train_idx = np.setdiff1d(indices, val_idx)
            
            x_train = x_norm[train_idx]
            y_train = y_norm[train_idx]
            x_val = x_norm[val_idx]
            y_val = y_norm[val_idx]
            
            n_train = len(x_train)
            if n_train < 10:
                continue
                
            try:
                diff = x_train[:, None] - x_train[None, :]
                K = np.exp(-diff ** 2 / (2.0 * ls ** 2))
                K_noisy = K + (noise_var / sigma_f2) * np.eye(n_train)
                
                try:
                    L = scipy.linalg.cholesky(K_noisy, lower=True)
                except Exception:
                    L = scipy.linalg.cholesky(K_noisy + 1e-8 * np.eye(n_train), lower=True)
                    
                Ly = scipy.linalg.solve(L, y_train, lower=True)
                alpha = scipy.linalg.solve(L.T, Ly)
                
                # Predict on val
                K_star = np.exp(-(x_val[:, None] - x_train[None, :]) ** 2 / (2.0 * ls ** 2))
                y_pred = K_star @ alpha
                
                mse = np.mean((y_pred - y_val) ** 2)
                cv_mses.append(mse)
            except Exception:
                continue
        
        if cv_mses:
            mean_mse = np.mean(cv_mses)
            cv_mse_per_ls[ls] = mean_mse

    # If no valid CV results, use direct fit
    if not cv_mse_per_ls:
        try:
            coeff = np.polynomial.polynomial.polyfit(x, y, 5)
            coeff = np.asarray(coeff, dtype=np.float64)
            if np.all(np.isfinite(coeff)):
                return coeff
        except Exception:
            pass
        return np.zeros(6)

    # Select top-3 length scales by CV performance for ensemble
    sorted_ls = sorted(cv_mse_per_ls.keys(), key=lambda ls: cv_mse_per_ls[ls])
    n_ensemble = min(3, len(sorted_ls))
    ensemble_ls = sorted_ls[:n_ensemble]

    # Compute weights based on inverse CV MSE (softmax-like)
    cv_mses_arr = np.array([cv_mse_per_ls[ls] for ls in ensemble_ls])
    # Use inverse MSE as weights, normalized
    inv_mse = 1.0 / np.maximum(cv_mses_arr, 1e-10)
    ensemble_weights = inv_mse / np.sum(inv_mse)

    # Evaluation points for polynomial fit
    # Use a grid of points in the range of x_norm, extended slightly
    x_min_norm, x_max_norm = float(np.min(x_norm)), float(np.max(x_norm))
    margin = 0.5 * (x_max_norm - x_min_norm)
    eval_points_norm = np.linspace(x_min_norm - margin * 0.5, x_max_norm + margin * 0.5, 300)
    
    # Fit GPR for each length scale in the ensemble and compute ensemble prediction
    y_pseudo_norm = np.zeros(len(eval_points_norm))
    pred_var_ensemble = np.zeros(len(eval_points_norm))

    for idx, ls in enumerate(ensemble_ls):
        w_ens = ensemble_weights[idx]
        try:
            diff = x_norm[:, None] - x_norm[None, :]
            K = np.exp(-diff ** 2 / (2.0 * ls ** 2))
            K_noisy = K + (noise_var / sigma_f2) * np.eye(n)
            
            try:
                L = scipy.linalg.cholesky(K_noisy, lower=True)
            except Exception:
                L = scipy.linalg.cholesky(K_noisy + 1e-8 * np.eye(n), lower=True)
            
            Ly = scipy.linalg.solve(L, y_norm, lower=True)
            alpha = scipy.linalg.solve(L.T, Ly)
            
            # Predict pseudo-labels
            K_star = np.exp(-(eval_points_norm[:, None] - x_norm[None, :]) ** 2 / (2.0 * ls ** 2))
            y_pred = K_star @ alpha
            
            # Predictive variance
            try:
                V = scipy.linalg.solve(L, K_star.T, lower=True)
                pred_var = sigma_f2 - np.sum(V ** 2, axis=0)
                pred_var = np.maximum(pred_var, 1e-6)
            except Exception:
                pred_var = np.ones(len(eval_points_norm)) * noise_var
            
            y_pseudo_norm += w_ens * y_pred
            pred_var_ensemble += w_ens * pred_var
        except Exception:
            continue

    # If ensemble failed completely, fallback to best single GPR
    if np.all(y_pseudo_norm == 0):
        best_ls = sorted_ls[0]
        try:
            diff = x_norm[:, None] - x_norm[None, :]
            K = np.exp(-diff ** 2 / (2.0 * best_ls ** 2))
            K_noisy = K + (noise_var / sigma_f2) * np.eye(n)
            
            try:
                L = scipy.linalg.cholesky(K_noisy, lower=True)
            except Exception:
                L = scipy.linalg.cholesky(K_noisy + 1e-8 * np.eye(n), lower=True)
            
            Ly = scipy.linalg.solve(L, y_norm, lower=True)
            alpha = scipy.linalg.solve(L.T, Ly)
            
            K_star = np.exp(-(eval_points_norm[:, None] - x_norm[None, :]) ** 2 / (2.0 * best_ls ** 2))
            y_pseudo_norm = K_star @ alpha
            
            try:
                V = scipy.linalg.solve(L, K_star.T, lower=True)
                pred_var_ensemble = sigma_f2 - np.sum(V ** 2, axis=0)
                pred_var_ensemble = np.maximum(pred_var_ensemble, 1e-6)
            except Exception:
                pred_var_ensemble = np.ones(len(eval_points_norm)) * noise_var
        except Exception:
            try:
                coeff = np.polynomial.polynomial.polyfit(x, y, 5)
                coeff = np.asarray(coeff, dtype=np.float64)
                if np.all(np.isfinite(coeff)):
                    return coeff
            except Exception:
                pass
            return np.zeros(6)

    # Weights for polynomial fit based on predictive variance
    weights = 1.0 / pred_var_ensemble
    weights = weights / np.sum(weights)

    # Fit polynomial in normalized space with weighted least squares
    try:
        # w in polyfit is sqrt of weights for the least squares problem
        # min sum w_i (y_i - p(x_i))^2
        coeff_norm = np.polynomial.polynomial.polyfit(eval_points_norm, y_pseudo_norm, 5, w=np.sqrt(weights))
        coeff_norm = np.asarray(coeff_norm, dtype=np.float64)
        if not np.all(np.isfinite(coeff_norm)):
            raise ValueError("Non-finite coefficients")
    except Exception:
        try:
            coeff_norm = np.polynomial.polynomial.polyfit(eval_points_norm, y_pseudo_norm, 5)
            coeff_norm = np.asarray(coeff_norm, dtype=np.float64)
            if not np.all(np.isfinite(coeff_norm)):
                raise ValueError("Non-finite coefficients")
        except Exception:
            # Final fallback
            try:
                coeff = np.polynomial.polynomial.polyfit(x, y, 5)
                coeff = np.asarray(coeff, dtype=np.float64)
                if np.all(np.isfinite(coeff)):
                    return coeff
            except Exception:
                pass
            return np.zeros(6)

    # Transform coefficients from normalized space to original space
    # p(x_norm) = sum c_i x_norm^i
    # x_norm = (x - x_mean) / x_std
    # p((x - x_mean)/x_std) = sum c_i ((x - x_mean)/x_std)^i
    # Let u = x - x_mean.
    # p(u/x_std) = sum c_i u^i / x_std^i
    # We want q(x) = sum d_i x^i
    # q(x) = p((x - x_mean)/x_std)
    # This is a polynomial transformation.
    # We can use numpy.polynomial.polynomial.polymul and polyadd or just expand.
    # Since degree is 5, we can do this manually or use a helper.
    
    # p(x) = c0 + c1*x + c2*x^2 + c3*x^3 + c4*x^4 + c5*x^5
    # Substitute x = (x_orig - x_mean) / x_std
    # Let z = x_orig.
    # term i: c_i * (z - x_mean)^i / x_std^i
    # Expand (z - x_mean)^i using binomial theorem: sum_{k=0}^i binom(i, k) z^k (-x_mean)^(i-k)
    # So coefficient of z^k in term i is c_i / x_std^i * binom(i, k) * (-x_mean)^(i-k)
    
    coeff_orig = np.zeros(6)
    for i in range(6):
        if abs(coeff_norm[i]) < 1e-15:
            continue
        ci = coeff_norm[i] / (x_std ** i)
        for k in range(i + 1):
            binom_coeff = 1.0
            for j in range(i - k):
                binom_coeff *= (i - j) / (j + 1)
            # binom(i, k)
            coeff_orig[k] += ci * binom_coeff * ((-x_mean) ** (i - k))
            
    # Scale Y back
    # y_orig = y_mean + y_std * y_norm
    # q(z) = y_mean + y_std * p_norm(z)
    coeff_orig = coeff_orig * y_std
    coeff_orig[0] += y_mean

    if not np.all(np.isfinite(coeff_orig)):
        return np.zeros(6)
        
    return coeff_orig
