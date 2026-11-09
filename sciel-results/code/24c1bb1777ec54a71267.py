"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def construct(training_x, training_subgrid_tendency):
    t_start = time.time()
    deadline = 110.0  # Safety margin before 120s limit

    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()

    n = min(x.size, y.size)
    if n == 0:
        return np.zeros(6, dtype=np.float64)

    x = x[:n]
    y = y[:n]

    # Remove non-finite values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = x.size

    if n == 0:
        return np.zeros(6, dtype=np.float64)

    # Standardize x for numerical stability
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-12:
        x_std = 1.0

    x_norm = (x - x_mean) / x_std

    def fit_ridge(x_fit, y_fit, degree, lam):
        """Fit polynomial of given degree with ridge regularization in normalized space."""
        V = np.vander(x_fit, N=degree + 1, increasing=True)

        # Ridge regression: (V^T V + lam*I) c = V^T y
        A = V.T @ V + lam * np.eye(degree + 1)
        b = V.T @ y_fit

        try:
            coeff = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            try:
                coeff = np.linalg.lstsq(A, b, rcond=None)[0]
            except:
                return np.zeros(degree + 1)

        return coeff

    def poly_eval(coeff, x_eval):
        """Evaluate polynomial with given coefficients."""
        pred = np.zeros_like(x_eval)
        for i in range(len(coeff)):
            pred += coeff[i] * (x_eval ** i)
        return pred

    def mse(pred, target):
        return np.mean((pred - target) ** 2)

    # Strategy: Use K-Fold Cross-Validation to select the best degree and regularization
    # This is more robust than a single holdout split, especially for unseen regimes.

    # We will try degrees 5, 4, 3, 2 and a range of lambdas.
    # To save time, we'll do a coarse grid search first, then refine if time permits.
    # Or just a reasonably sized grid.

    degrees_to_try = [5, 4, 3, 2]
    
    # Log-spaced lambdas. 
    # Small lambda = less regularization (more flexible)
    # Large lambda = more regularization (smoother)
    lambdas_to_try = np.logspace(-6, 4, 25)

    # K-Fold setup
    k_folds = 5
    np.random.seed(42)  # Reproducibility
    indices = np.random.permutation(n)
    
    # Handle case where n < k_folds
    if n < k_folds:
        k_folds = max(2, n)
        
    fold_sizes = np.full(k_folds, n // k_folds, dtype=int)
    fold_sizes[:n % k_folds] += 1
    folds = np.split(indices, np.cumsum(fold_sizes)[:-1])

    best_score = np.inf
    best_coeff_norm = None
    best_degree = 5
    best_lam = 1e-3

    # Coarse Search
    for degree in degrees_to_try:
        if time.time() - t_start > deadline - 15:
            break
            
        for lam in lambdas_to_try:
            if time.time() - t_start > deadline - 15:
                break
                
            cv_mse_sum = 0.0
            valid_fold = True
            
            for i in range(k_folds):
                val_idx = folds[i]
                train_idx = np.concatenate([folds[j] for j in range(k_folds) if j != i])
                
                x_train = x_norm[train_idx]
                y_train = y[train_idx]
                x_val = x_norm[val_idx]
                y_val = y[val_idx]
                
                coeff = fit_ridge(x_train, y_train, degree, lam)
                
                if not np.all(np.isfinite(coeff)):
                    valid_fold = False
                    break
                
                pred = poly_eval(coeff, x_val)
                cv_mse_sum += mse(pred, y_val)
            
            if not valid_fold:
                continue
                
            avg_cv_mse = cv_mse_sum / k_folds
            
            if avg_cv_mse < best_score:
                best_score = avg_cv_mse
                best_degree = degree
                best_lam = lam

    if best_coeff_norm is None or not np.isfinite(best_score):
        # Fallback if CV failed or found nothing good
        # Just use a very regularized degree 5
        best_degree = 5
        best_lam = 1.0
        best_score = np.inf

    # Refinement Step: If time permits, do a finer search around the best lambda found
    if time.time() - t_start < deadline - 20 and best_lam > 0:
        # Create a finer grid around best_lam
        # Also try slightly different degrees if close
        local_lambdas = np.array([best_lam * (0.5 ** i) for i in range(-2, 3)])
        local_lambdas = local_lambdas[local_lambdas > 0]
        
        current_best = best_score
        current_best_lam = best_lam
        current_best_deg = best_degree

        for degree in [best_degree, best_degree - 1, best_degree + 1]:
            if degree < 1 or degree > 5:
                continue
            for lam in local_lambdas:
                if time.time() - t_start > deadline - 10:
                    break
                
                cv_mse_sum = 0.0
                valid_fold = True
                
                for i in range(k_folds):
                    val_idx = folds[i]
                    train_idx = np.concatenate([folds[j] for j in range(k_folds) if j != i])
                    
                    x_train = x_norm[train_idx]
                    y_train = y[train_idx]
                    x_val = x_norm[val_idx]
                    y_val = y[val_idx]
                    
                    coeff = fit_ridge(x_train, y_train, degree, lam)
                    
                    if not np.all(np.isfinite(coeff)):
                        valid_fold = False
                        break
                    
                    pred = poly_eval(coeff, x_val)
                    cv_mse_sum += mse(pred, y_val)
                
                if not valid_fold:
                    continue
                    
                avg_cv_mse = cv_mse_sum / k_folds
                
                if avg_cv_mse < current_best:
                    current_best = avg_cv_mse
                    current_best_lam = lam
                    current_best_deg = degree

        best_lam = current_best_lam
        best_degree = current_best_deg
        best_score = current_best

    # Final Fit: Use ALL data with the best selected hyperparameters
    # This utilizes all information for the final coefficients
    final_coeff_norm = fit_ridge(x_norm, y, best_degree, best_lam)
    
    if not np.all(np.isfinite(final_coeff_norm)):
        # Fallback to a simple robust fit
        final_coeff_norm = fit_ridge(x_norm, y, 5, 1.0)

    # Pad to 6 coefficients if needed
    if len(final_coeff_norm) < 6:
        temp = np.zeros(6)
        temp[:len(final_coeff_norm)] = final_coeff_norm
        final_coeff_norm = temp

    # Convert from normalized x to original x scale
    # x_norm = (x - x_mean) / x_std
    # We have f(x_norm) = sum(c_i * x_norm^i)
    # Need g(x) = f((x - x_mean)/x_std) = sum(c_i * ((x - x_mean)/x_std)^i)
    
    max_deg = min(5, len(final_coeff_norm) - 1)
    coeff_orig = np.zeros(6, dtype=np.float64)
    
    for i in range(max_deg + 1):
        if abs(final_coeff_norm[i]) < 1e-15:
            continue
        
        ci_over_std_pow = final_coeff_norm[i] / (x_std ** i)
        
        # Expand (x - x_mean)^i = sum_{j=0}^{i} binom(i,j) * x^j * (-x_mean)^(i-j)
        for j in range(i + 1):
            # Compute binomial coefficient
            binom_val = 1.0
            for k in range(j):
                binom_val = binom_val * (i - k) / (k + 1)
            
            term = binom_val * ((-x_mean) ** (i - j)) * ci_over_std_pow
            coeff_orig[j] += term
    
    # Ensure finite
    if not np.all(np.isfinite(coeff_orig)):
        coeff_orig = np.nan_to_num(coeff_orig, nan=0.0, posinf=0.0, neginf=0.0)
    
    return coeff_orig
# EVOLVE-BLOCK-END
