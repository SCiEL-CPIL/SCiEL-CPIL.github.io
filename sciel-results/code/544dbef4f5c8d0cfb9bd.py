"""Direct approach: Fit degree-5 polynomial with regularization, leveraging the
linear-in-X structure of the Lorenz-96 subgrid forcing. Use train/val split
to select between model orders and regularization strengths. Includes a
'force-proxy' model where y = x * g(x) captures F-dependent scaling."""
import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    t0 = time.time()
    deadline = t0 + 140.0

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
    else:
        x_train = x
        y_train = y
        x_val = x
        y_val = y

    best_coeff = None
    best_val_err = np.inf

    # K-fold cross-validation for regularization strength selection
    def kfold_cv(x_data, y_data, model_type, lam, n_folds=5):
        """Perform k-fold CV to estimate generalization error."""
        n_samples = len(x_data)
        if n_samples < n_folds * 2:
            n_folds = max(2, n_samples // 2)
        
        rng_cv = np.random.default_rng(123)
        indices = rng_cv.permutation(n_samples)
        fold_size = n_samples // n_folds
        fold_errors = []
        
        for fold in range(n_folds):
            if time.time() > deadline:
                return np.inf
            
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < n_folds - 1 else n_samples
            val_idx = indices[val_start:val_end]
            train_idx = np.concatenate([indices[:val_start], indices[val_end:]])
            
            x_tr = x_data[train_idx]
            y_tr = y_data[train_idx]
            x_vl = x_data[val_idx]
            y_vl = y_data[val_idx]
            
            try:
                if model_type == 'linear':
                    A_tr = np.column_stack([np.ones_like(x_tr), x_tr])
                    A_vl = np.column_stack([np.ones_like(x_vl), x_vl])
                    # Exclude intercept from penalty
                    penalty = np.diag([0.0, 1.0])
                    lam_scaled = lam * np.trace(A_tr[:, 1:] @ A_tr[:, 1:].T) / 1.0
                    coeff = np.linalg.solve(A_tr.T @ A_tr + lam_scaled * penalty, A_tr.T @ y_tr)
                    pred = A_vl @ coeff
                elif model_type == 'full':
                    A_tr = np.column_stack([x_tr**i for i in range(6)])
                    A_vl = np.column_stack([x_vl**i for i in range(6)])
                    # Exclude intercept from penalty
                    penalty = np.diag([0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
                    lam_scaled = lam * np.trace(A_tr[:, 1:] @ A_tr[:, 1:].T) / 5.0
                    coeff = np.linalg.solve(A_tr.T @ A_tr + lam_scaled * penalty, A_tr.T @ y_tr)
                    pred = A_vl @ coeff
                elif model_type == 'odd':
                    A_tr = np.column_stack([x_tr, x_tr**3, x_tr**5])
                    A_vl = np.column_stack([x_vl, x_vl**3, x_vl**5])
                    penalty = np.eye(3)
                    lam_scaled = lam * np.trace(A_tr.T @ A_tr) / 3.0
                    coeff = np.linalg.solve(A_tr.T @ A_tr + lam_scaled * penalty, A_tr.T @ y_tr)
                    pred = A_vl @ coeff
                elif model_type == 'force_proxy':
                    # y = x * g(x) where g(x) = b0 + b1*x + b2*x^2 + b3*x^3 + b4*x^4
                    # This gives y = b0*x + b1*x^2 + b2*x^3 + b3*x^4 + b4*x^5
                    # Design matrix columns: x, x^2, x^3, x^4, x^5
                    A_tr = np.column_stack([x_tr**(i+1) for i in range(5)])
                    A_vl = np.column_stack([x_vl**(i+1) for i in range(5)])
                    # Regularize all coefficients
                    penalty = np.eye(5)
                    lam_scaled = lam * np.trace(A_tr.T @ A_tr) / 5.0
                    coeff = np.linalg.solve(A_tr.T @ A_tr + lam_scaled * penalty, A_tr.T @ y_tr)
                    pred = A_vl @ coeff
                elif model_type == 'force_proxy_intercept':
                    # y = a0 + x * g(x) where g(x) = b0 + b1*x + b2*x^2 + b3*x^3
                    # This gives y = a0 + b0*x + b1*x^2 + b2*x^3 + b3*x^4
                    # Design matrix columns: 1, x, x^2, x^3, x^4
                    A_tr = np.column_stack([np.ones_like(x_tr), x_tr**1, x_tr**2, x_tr**3, x_tr**4])
                    A_vl = np.column_stack([np.ones_like(x_vl), x_vl**1, x_vl**2, x_vl**3, x_vl**4])
                    # Exclude intercept from penalty
                    penalty = np.diag([0.0, 1.0, 1.0, 1.0, 1.0])
                    lam_scaled = lam * np.trace(A_tr[:, 1:] @ A_tr[:, 1:].T) / 4.0
                    coeff = np.linalg.solve(A_tr.T @ A_tr + lam_scaled * penalty, A_tr.T @ y_tr)
                    pred = A_vl @ coeff
                else:
                    return np.inf
            except np.linalg.LinAlgError:
                return np.inf
            
            fold_errors.append(np.mean((pred - y_vl)**2))
        
        return np.mean(fold_errors)

    # Try linear model (degree 1) with various regularizations using CV
    for lam in [1e-8, 1e-6, 1e-4, 1e-2, 1e0, 1e2, 1e4, 1e6]:
        if time.time() > deadline:
            break
        try:
            cv_err = kfold_cv(x_train, y_train, 'linear', lam)
            if cv_err < best_val_err:
                best_val_err = cv_err
                A_lin = np.column_stack([np.ones_like(x_train), x_train])
                penalty = np.diag([0.0, 1.0])
                lam_scaled = lam * np.trace(A_lin[:, 1:] @ A_lin[:, 1:].T) / 1.0
                coeff_lin = np.linalg.solve(A_lin.T @ A_lin + lam_scaled * penalty, A_lin.T @ y_train)
                coeff_full = np.zeros(6)
                coeff_full[:2] = coeff_lin
                best_coeff = coeff_full
        except Exception:
            continue

    # Try degree-5 polynomial with various regularizations using CV
    for lam in [1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e0, 1e2, 1e4, 1e6, 1e8]:
        if time.time() > deadline:
            break
        try:
            cv_err = kfold_cv(x_train, y_train, 'full', lam)
            if cv_err < best_val_err:
                best_val_err = cv_err
                A_full = np.column_stack([x_train**i for i in range(6)])
                penalty = np.diag([0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
                lam_scaled = lam * np.trace(A_full[:, 1:] @ A_full[:, 1:].T) / 5.0
                coeff_full = np.linalg.solve(A_full.T @ A_full + lam_scaled * penalty, A_full.T @ y_train)
                best_coeff = coeff_full
        except Exception:
            continue

    # Try odd-polynomial model (x, x^3, x^5) with various regularizations using CV
    for lam in [1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e0, 1e2, 1e4, 1e6, 1e8]:
        if time.time() > deadline:
            break
        try:
            cv_err = kfold_cv(x_train, y_train, 'odd', lam)
            if cv_err < best_val_err:
                best_val_err = cv_err
                A_odd = np.column_stack([x_train, x_train**3, x_train**5])
                penalty = np.eye(3)
                lam_scaled = lam * np.trace(A_odd.T @ A_odd) / 3.0
                coeff_odd = np.linalg.solve(A_odd.T @ A_odd + lam_scaled * penalty, A_odd.T @ y_train)
                coeff_full = np.zeros(6)
                coeff_full[1] = coeff_odd[0]
                coeff_full[3] = coeff_odd[1]
                coeff_full[5] = coeff_odd[2]
                best_coeff = coeff_full
        except Exception:
            continue

    # Try force-proxy model: y = x * g(x) = b0*x + b1*x^2 + b2*x^3 + b3*x^4 + b4*x^5
    for lam in [1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e0, 1e2, 1e4, 1e6, 1e8]:
        if time.time() > deadline:
            break
        try:
            cv_err = kfold_cv(x_train, y_train, 'force_proxy', lam)
            if cv_err < best_val_err:
                best_val_err = cv_err
                A_fp = np.column_stack([x_train**(i+1) for i in range(5)])
                penalty = np.eye(5)
                lam_scaled = lam * np.trace(A_fp.T @ A_fp) / 5.0
                coeff_fp = np.linalg.solve(A_fp.T @ A_fp + lam_scaled * penalty, A_fp.T @ y_train)
                coeff_full = np.zeros(6)
                coeff_full[1:] = coeff_fp  # x^1 through x^5
                best_coeff = coeff_full
        except Exception:
            continue

    # Try force-proxy with intercept: y = a0 + b0*x + b1*x^2 + b2*x^3 + b3*x^4
    for lam in [1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e0, 1e2, 1e4, 1e6, 1e8]:
        if time.time() > deadline:
            break
        try:
            cv_err = kfold_cv(x_train, y_train, 'force_proxy_intercept', lam)
            if cv_err < best_val_err:
                best_val_err = cv_err
                A_fpi = np.column_stack([np.ones_like(x_train), x_train**1, x_train**2, x_train**3, x_train**4])
                penalty = np.diag([0.0, 1.0, 1.0, 1.0, 1.0])
                lam_scaled = lam * np.trace(A_fpi[:, 1:] @ A_fpi[:, 1:].T) / 4.0
                coeff_fpi = np.linalg.solve(A_fpi.T @ A_fpi + lam_scaled * penalty, A_fpi.T @ y_train)
                coeff_full = np.zeros(6)
                coeff_full[:5] = coeff_fpi  # x^0 through x^4
                best_coeff = coeff_full
        except Exception:
            continue

    # If no model worked, fall back to least squares linear
    if best_coeff is None or not np.all(np.isfinite(best_coeff)):
        try:
            c_lin = np.polynomial.polynomial.polyfit(x, y, 1)
            best_coeff = np.zeros(6)
            best_coeff[:2] = c_lin
        except Exception:
            best_coeff = np.zeros(6)

    # Final verification
    if not np.all(np.isfinite(best_coeff)):
        best_coeff = np.zeros(6)

    return best_coeff
