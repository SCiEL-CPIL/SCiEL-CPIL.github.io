import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    t_start = time.time()
    deadline = t_start + 135.0  # safety margin before 150s kill

    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)

    # Remove any NaN/Inf
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    n = len(x)
    if n < 6:
        return np.array([np.mean(y) if n > 0 else 0.0, 0, 0, 0, 0, 0], dtype=np.float64)

    # Standardize x for better conditioning
    mu = np.mean(x)
    sigma = np.std(x)
    if sigma < 1e-12:
        sigma = 1.0
    xc = (x - mu) / sigma

    y_var = np.var(y)
    if y_var < 1e-15:
        y_var = 1e-15

    best_score = np.inf
    best_deg = 5
    best_alpha = 1e-4

    # Precompute powers of xc
    max_deg = 5
    xc_powers = np.ones((n, max_deg + 1), dtype=np.float64)
    for k in range(1, max_deg + 1):
        xc_powers[:, k] = xc_powers[:, k - 1] * xc

    # Efficient ridge fit using normal equations
    def _fit_ridge(xc_powers_sub, y_sub, degree, alpha):
        V = xc_powers_sub[:, :degree + 1]
        if alpha > 0:
            A = V.T @ V
            b = V.T @ y_sub
            reg = alpha * np.eye(degree + 1)
            try:
                c = np.linalg.solve(A + reg, b)
            except np.linalg.LinAlgError:
                c = np.linalg.lstsq(V, y_sub, rcond=None)[0]
        else:
            try:
                c = np.linalg.lstsq(V, y_sub, rcond=None)[0]
            except Exception:
                c = np.zeros(degree + 1)
        return c

    # Convert centered coefficients to original basis
    def _coeffs_in_original_basis(c_centered, mu, sigma, deg):
        coeffs_orig = np.zeros(6, dtype=np.float64)
        for i in range(deg + 1):
            ci = c_centered[i]
            inv_si = (1.0 / sigma) ** i
            for k in range(i + 1):
                # Compute binomial coefficient C(i, k)
                if k == 0 or k == i:
                    Ck = 1.0
                else:
                    mm = min(k, i - k)
                    num = 1
                    den = 1
                    for t in range(1, mm + 1):
                        num *= (i - mm + t)
                        den *= t
                    Ck = num / den
                coeffs_orig[k] += ci * Ck * ((-mu) ** (i - k)) * inv_si
        return coeffs_orig

    # Cross-validation using k-fold
    def _cv_mse(xc_powers, y, degree, alpha, k_folds=5):
        n_samples = len(xc_powers)
        if n_samples < (degree + 2) * k_folds:
            k_folds = 2
        
        idx = np.random.RandomState(42).permutation(n_samples)
        fold_size = n_samples // k_folds
        mse_sum = 0.0
        
        for k in range(k_folds):
            lo = k * fold_size
            hi = lo + fold_size
            if k == k_folds - 1:
                hi = n_samples
            
            val_idx = idx[lo:hi]
            train_idx = np.concatenate([idx[:lo], idx[hi:]])
            
            xc_train = xc_powers[train_idx]
            y_train = y[train_idx]
            xc_val = xc_powers[val_idx]
            y_val = y[val_idx]
            
            c = _fit_ridge(xc_train, y_train, degree, alpha)
            pred = xc_val[:, :degree + 1] @ c
            mse_sum += np.mean((pred - y_val)**2)
        
        return mse_sum / k_folds

    # Grid search over degree and regularization
    degrees = [1, 2, 3, 4, 5]
    alphas = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0]

    # First pass: coarse grid
    for deg in degrees:
        if time.time() > deadline:
            break
        for alpha in alphas:
            if time.time() > deadline:
                break
            try:
                cv_mse = _cv_mse(xc_powers, y, deg, alpha)
                score = cv_mse / y_var
                if score < best_score:
                    best_score = score
                    best_deg = deg
                    best_alpha = alpha
            except Exception:
                continue

    # Second pass: refine around the best degree with finer alphas
    if time.time() < deadline - 10.0:
        refined_alphas = np.logspace(-7, 4, 30)
        for deg in [max(1, best_deg - 1), best_deg, min(5, best_deg + 1)]:
            if time.time() > deadline:
                break
            for alpha in refined_alphas:
                if time.time() > deadline:
                    break
                try:
                    cv_mse = _cv_mse(xc_powers, y, deg, float(alpha))
                    score = cv_mse / y_var
                    if score < best_score:
                        best_score = score
                        best_deg = deg
                        best_alpha = float(alpha)
                except Exception:
                    continue

    # Third pass: even finer refinement if time permits
    if time.time() < deadline - 5.0:
        ultra_fine_alphas = np.logspace(np.log10(max(1e-8, best_alpha / 10)), np.log10(min(1e5, best_alpha * 10)), 20)
        for alpha in ultra_fine_alphas:
            if time.time() > deadline:
                break
            try:
                cv_mse = _cv_mse(xc_powers, y, best_deg, float(alpha))
                score = cv_mse / y_var
                if score < best_score:
                    best_score = score
                    best_alpha = float(alpha)
            except Exception:
                continue

    # Final fit on all data with best settings
    c_centered = _fit_ridge(xc_powers, y, best_deg, best_alpha)

    # Convert from centered-standardized basis to original x basis
    coeffs_orig = _coeffs_in_original_basis(c_centered, mu, sigma, best_deg)

    # Ensure finite
    if not np.all(np.isfinite(coeffs_orig)):
        coeffs_orig = np.zeros(6, dtype=np.float64)
        coeffs_orig[0] = np.mean(y)

    return coeffs_orig
