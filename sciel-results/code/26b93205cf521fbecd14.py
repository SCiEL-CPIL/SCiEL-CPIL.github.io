"""Direct approach: Fit degree-5 polynomial with regularization, leveraging the
linear-in-X structure of the Lorenz-96 subgrid forcing. Use train/val split
to select between model orders and regularization strengths. Includes a
'force-proxy' model where y = x * g(x) captures F-dependent scaling.
Added: normalized x scaling, efficient CV, and final polish pass."""
import numpy as np
import time
import math

def construct(training_x, training_subgrid_tendency):
    t0 = time.time()
    deadline = t0 + 130.0

    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)

    # Filter non-finite values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if x.size == 0:
        return np.zeros(6)

    n = x.size

    # If very few points, fall back to linear
    if n < 10:
        try:
            c_lin = np.polynomial.polynomial.polyfit(x, y, 1)
            coeff = np.zeros(6)
            coeff[:2] = c_lin
            if not np.all(np.isfinite(coeff)):
                return np.zeros(6)
            return coeff
        except Exception:
            return np.zeros(6)

    # Normalize x for better conditioning during fitting,
    # then convert coefficients back to original scale at the end.
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-12:
        x_std = 1.0
    x_norm = (x - x_mean) / x_std

    # The subgrid forcing h*c*mean(Y) is approximately linear in X
    # because Y is linearly driven by X in the fast equations.
    # Try both linear and full degree-5 fits, pick best via validation.

    # Create train/val split
    if n > 50:
        rng = np.random.default_rng(42)
        indices = rng.permutation(n)
        val_size = max(10, n // 10)
        val_idx = indices[:val_size]
        train_idx = indices[val_size:]

        x_train = x[train_idx]
        y_train = y[train_idx]
        x_val = x[val_idx]
        y_val = y[val_idx]
        x_train_norm = (x_train - x_mean) / x_std
        x_val_norm = (x_val - x_mean) / x_std
    else:
        x_train = x
        y_train = y
        x_val = x
        y_val = y
        x_train_norm = x_norm
        x_val_norm = x_norm

    best_coeff_norm = None  # coefficients in normalized space
    best_val_err = np.inf
    best_model_type = None

    # Pre-compute design matrices for normalized x to speed up CV
    def build_design_matrix(xn, model_type):
        """Build design matrix for given model type using normalized x."""
        if model_type == 'linear':
            return np.column_stack([np.ones_like(xn), xn])
        elif model_type == 'full':
            return np.column_stack([xn**i for i in range(6)])
        elif model_type == 'odd':
            return np.column_stack([xn, xn**3, xn**5])
        elif model_type == 'force_proxy':
            return np.column_stack([xn**(i+1) for i in range(5)])
        elif model_type == 'force_proxy_intercept':
            return np.column_stack([np.ones_like(xn), xn**1, xn**2, xn**3, xn**4])
        else:
            return None

    def get_penalty(model_type):
        """Get penalty matrix excluding intercept if present."""
        if model_type == 'linear':
            return np.diag([0.0, 1.0])
        elif model_type == 'full':
            return np.diag([0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        elif model_type == 'odd':
            return np.eye(3)
        elif model_type == 'force_proxy':
            return np.eye(5)
        elif model_type == 'force_proxy_intercept':
            return np.diag([0.0, 1.0, 1.0, 1.0, 1.0])
        else:
            return np.eye(6)

    def kfold_cv(xn_data, y_data, model_type, lam, n_folds=5):
        """Perform k-fold CV to estimate generalization error using normalized x."""
        n_samples = len(xn_data)
        if n_samples < n_folds * 2:
            n_folds = max(2, n_samples // 2)
        
        rng_cv = np.random.default_rng(123)
        indices = rng_cv.permutation(n_samples)
        fold_size = n_samples // n_folds
        fold_errors = []
        
        # Pre-compute full design matrix
        A_full_data = build_design_matrix(xn_data, model_type)
        penalty = get_penalty(model_type)
        n_params = A_full_data.shape[1]
        
        for fold in range(n_folds):
            if time.time() > deadline:
                return np.inf
            
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < n_folds - 1 else n_samples
            val_idx = indices[val_start:val_end]
            train_idx = np.concatenate([indices[:val_start], indices[val_end:]])
            
            A_tr = A_full_data[train_idx]
            y_tr = y_data[train_idx]
            A_vl = A_full_data[val_idx]
            y_vl = y_data[val_idx]
            
            try:
                # Scale lambda based on data magnitude for better conditioning
                # Use trace of A^T A for non-intercept columns
                if model_type in ('linear', 'full', 'force_proxy_intercept'):
                    # Exclude intercept from scaling
                    non_intercept = slice(1, n_params)
                    trace_val = np.trace(A_tr[:, non_intercept].T @ A_tr[:, non_intercept]) / (n_params - 1)
                else:
                    trace_val = np.trace(A_tr.T @ A_tr) / n_params
                
                lam_scaled = lam * max(trace_val, 1e-10)
                coeff = np.linalg.solve(A_tr.T @ A_tr + lam_scaled * penalty, A_tr.T @ y_tr)
                pred = A_vl @ coeff
            except np.linalg.LinAlgError:
                return np.inf
            
            fold_errors.append(np.mean((pred - y_vl)**2))
        
        return np.mean(fold_errors)

    def fit_model(xn_data, y_data, model_type, lam):
        """Fit model on normalized x and return coefficients in normalized space."""
        A = build_design_matrix(xn_data, model_type)
        penalty = get_penalty(model_type)
        n_params = A.shape[1]
        
        if model_type in ('linear', 'full', 'force_proxy_intercept'):
            non_intercept = slice(1, n_params)
            trace_val = np.trace(A[:, non_intercept].T @ A[:, non_intercept]) / (n_params - 1)
        else:
            trace_val = np.trace(A.T @ A) / n_params
        
        lam_scaled = lam * max(trace_val, 1e-10)
        return np.linalg.solve(A.T @ A + lam_scaled * penalty, A.T @ y_data)

    def convert_to_original_space(coeff_norm, model_type):
        """Convert coefficients from normalized space to original x space.
        x_norm = (x - x_mean) / x_std, so x = x_std * x_norm + x_mean
        We need to express sum(c_i * x_norm^i) in terms of x.
        Use binomial expansion: x_norm^i = ((x - x_mean)/x_std)^i
        """
        coeff_orig = np.zeros(6)
        
        if model_type == 'linear':
            # c0 + c1 * x_norm = c0 + c1 * (x - x_mean) / x_std
            # = (c0 - c1*x_mean/x_std) + (c1/x_std) * x
            coeff_orig[0] = coeff_norm[0] - coeff_norm[1] * x_mean / x_std
            coeff_orig[1] = coeff_norm[1] / x_std
        elif model_type == 'full':
            # sum_{i=0}^{5} c_i * x_norm^i where x_norm = (x - x_mean)/x_std
            # Expand each term using binomial theorem
            for i in range(6):
                ci = coeff_norm[i]
                if abs(ci) < 1e-15:
                    continue
                # x_norm^i = (x - x_mean)^i / x_std^i
                # (x - x_mean)^i = sum_{j=0}^{i} binom(i,j) * x^j * (-x_mean)^(i-j)
                for j in range(i + 1):
                    if j > 5:
                        continue
                    binom_coeff = float(math.comb(i, j))
                    coeff_orig[j] += ci * binom_coeff * ((-x_mean) ** (i - j)) / (x_std ** i)
        elif model_type == 'odd':
            # coeff_norm = [b1, b3, b5] for x_norm, x_norm^3, x_norm^5
            for power, ci in zip([1, 3, 5], coeff_norm):
                if abs(ci) < 1e-15:
                    continue
                for j in range(power + 1):
                    if j > 5:
                        continue
                    binom_coeff = float(math.comb(power, j))
                    coeff_orig[j] += ci * binom_coeff * ((-x_mean) ** (power - j)) / (x_std ** power)
        elif model_type == 'force_proxy':
            # y = b0*x_norm + b1*x_norm^2 + ... + b4*x_norm^5
            for power, ci in zip(range(1, 6), coeff_norm):
                if abs(ci) < 1e-15:
                    continue
                for j in range(power + 1):
                    if j > 5:
                        continue
                    binom_coeff = float(math.comb(power, j))
                    coeff_orig[j] += ci * binom_coeff * ((-x_mean) ** (power - j)) / (x_std ** power)
        elif model_type == 'force_proxy_intercept':
            # y = a0 + b0*x_norm + b1*x_norm^2 + ... + b3*x_norm^4
            coeff_orig[0] = coeff_norm[0]
            for power, ci in zip(range(1, 5), coeff_norm[1:]):
                if abs(ci) < 1e-15:
                    continue
                for j in range(power + 1):
                    if j > 5:
                        continue
                    binom_coeff = float(math.comb(power, j))
                    coeff_orig[j] += ci * binom_coeff * ((-x_mean) ** (power - j)) / (x_std ** power)
        
        return coeff_orig

    # Lambda grid - use wider range since we're now scaling by data magnitude
    lambda_grid = [1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e0, 1e2, 1e4, 1e6, 1e8]

    # Pre-compute Gram matrices and RHS for each model type to speed up CV
    # For each model type, we precompute A_full and its Gram components
    model_types = ['linear', 'full', 'odd', 'force_proxy', 'force_proxy_intercept']
    
    # Precompute for each model type: full design matrix, penalty, trace scaling
    precomputed = {}
    for mt in model_types:
        if time.time() > deadline - 20.0:
            break
        A_full = build_design_matrix(x_train_norm, mt)
        penalty = get_penalty(mt)
        n_params = A_full.shape[1]
        
        if mt in ('linear', 'full', 'force_proxy_intercept'):
            non_intercept = slice(1, n_params)
            trace_val = np.trace(A_full[:, non_intercept].T @ A_full[:, non_intercept]) / (n_params - 1)
        else:
            trace_val = np.trace(A_full.T @ A_full) / n_params
        
        precomputed[mt] = {
            'A_full': A_full,
            'penalty': penalty,
            'n_params': n_params,
            'trace_val': trace_val
        }
    
    # Fast CV using precomputed matrices
    def kfold_cv_fast(xn_data, y_data, model_type, lam):
        """Perform k-fold CV using precomputed design matrix."""
        if model_type not in precomputed:
            return np.inf
        
        pc = precomputed[model_type]
        A_full_data = pc['A_full']
        penalty = pc['penalty']
        n_params = pc['n_params']
        trace_val = pc['trace_val']
        
        n_samples = len(xn_data)
        n_folds = 5
        if n_samples < n_folds * 2:
            n_folds = max(2, n_samples // 2)
        
        rng_cv = np.random.default_rng(123)
        indices = rng_cv.permutation(n_samples)
        fold_size = n_samples // n_folds
        fold_errors = []
        
        for fold in range(n_folds):
            if time.time() > deadline - 5.0:
                return np.inf
            
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < n_folds - 1 else n_samples
            val_idx = indices[val_start:val_end]
            train_idx = np.concatenate([indices[:val_start], indices[val_end:]])
            
            A_tr = A_full_data[train_idx]
            y_tr = y_data[train_idx]
            A_vl = A_full_data[val_idx]
            y_vl = y_data[val_idx]
            
            try:
                lam_scaled = lam * max(trace_val, 1e-10)
                coeff = np.linalg.solve(A_tr.T @ A_tr + lam_scaled * penalty, A_tr.T @ y_tr)
                pred = A_vl @ coeff
            except np.linalg.LinAlgError:
                return np.inf
            
            fold_errors.append(np.mean((pred - y_vl)**2))
        
        return np.mean(fold_errors)

    # Try each model type with various regularizations using fast CV
    for mt in model_types:
        if time.time() > deadline - 15.0:
            break
        if mt not in precomputed:
            continue
        for lam in lambda_grid:
            if time.time() > deadline - 15.0:
                break
            try:
                cv_err = kfold_cv_fast(x_train_norm, y_train, mt, lam)
                if cv_err < best_val_err:
                    best_val_err = cv_err
                    best_coeff_norm = fit_model(x_train_norm, y_train, mt, lam)
                    best_model_type = mt
            except Exception:
                continue

    # Convert best coefficients from normalized space to original space
    if best_coeff_norm is not None and np.all(np.isfinite(best_coeff_norm)):
        best_coeff = convert_to_original_space(best_coeff_norm, best_model_type)
    else:
        # Fall back to least squares linear in original space
        try:
            c_lin = np.polynomial.polynomial.polyfit(x, y, 1)
            best_coeff = np.zeros(6)
            best_coeff[:2] = c_lin
        except Exception:
            best_coeff = np.zeros(6)

    # Final polish: refine the best model using all training data with the
    # selected model type and a fine-tuned lambda via local search
    if time.time() < deadline - 5.0:
        try:
            # First, do a fine-grained lambda search around the best grid value
            # We need to track which lambda was best; re-derive it
            best_lam = None
            best_fine_err = np.inf
            
            for lam in lambda_grid:
                if time.time() > deadline - 5.0:
                    break
                try:
                    cv_err = kfold_cv_fast(x_train_norm, y_train, best_model_type, lam)
                    if cv_err < best_fine_err:
                        best_fine_err = cv_err
                        best_lam = lam
                except Exception:
                    continue
            
            if best_lam is not None:
                # Fine-grained local search around best_lam
                log_lam = np.log10(best_lam) if best_lam > 0 else -12.0
                lam_candidates = [10**(log_lam + d) for d in [-0.5, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.5]]
                best_polish_err = np.inf
                best_polish_lam = best_lam
                
                for lam_p in lam_candidates:
                    if time.time() > deadline - 5.0:
                        break
                    try:
                        cv_err = kfold_cv_fast(x_train_norm, y_train, best_model_type, lam_p)
                        if cv_err < best_polish_err:
                            best_polish_err = cv_err
                            best_polish_lam = lam_p
                    except Exception:
                        continue
                
                # Final high-precision refit: use ALL training data (not just train split)
                # with the best lambda for maximum accuracy
                all_coeff = fit_model(x_norm, y, best_model_type, best_polish_lam)
                if np.all(np.isfinite(all_coeff)):
                    polished_coeff = convert_to_original_space(all_coeff, best_model_type)
                    if np.all(np.isfinite(polished_coeff)):
                        best_coeff = polished_coeff
                else:
                    # Fall back to train-split fit
                    polished_coeff_norm = fit_model(x_train_norm, y_train, best_model_type, best_polish_lam)
                    if np.all(np.isfinite(polished_coeff_norm)):
                        polished_coeff = convert_to_original_space(polished_coeff_norm, best_model_type)
                        if np.all(np.isfinite(polished_coeff)):
                            best_coeff = polished_coeff
        except Exception:
            pass

    # Final verification
    if not np.all(np.isfinite(best_coeff)):
        best_coeff = np.zeros(6)

    return best_coeff
