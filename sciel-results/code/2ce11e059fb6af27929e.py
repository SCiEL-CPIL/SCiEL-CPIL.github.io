"""Direct approach: Fit degree-5 polynomial with regularization, leveraging the
linear-in-X structure of the Lorenz-96 subgrid forcing. Use train/val split
to select between model orders and regularization strengths."""
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

    # K-fold cross-validation helper for more robust model selection
    def kfold_cv(x_data, y_data, power_list, lam, n_folds=5):
        """Compute mean CV error for given power list and regularization."""
        n_pts = len(x_data)
        if n_pts < n_folds * 2:
            n_folds = max(2, n_pts // 2)
        rng_cv = np.random.default_rng(123)
        indices = rng_cv.permutation(n_pts)
        fold_size = n_pts // n_folds
        errors = []
        for k in range(n_folds):
            val_start = k * fold_size
            val_end = val_start + fold_size if k < n_folds - 1 else n_pts
            val_idx = indices[val_start:val_end]
            train_mask = np.ones(n_pts, dtype=bool)
            train_mask[val_idx] = False
            train_idx = np.where(train_mask)[0]
            
            x_tr = x_data[train_idx]
            y_tr = y_data[train_idx]
            x_va = x_data[val_idx]
            y_va = y_data[val_idx]
            
            A_tr = np.column_stack([x_tr**p for p in power_list])
            A_va = np.column_stack([x_va**p for p in power_list])
            lam_scaled = lam * np.trace(A_tr.T @ A_tr) / len(power_list)
            try:
                coeff = np.linalg.solve(A_tr.T @ A_tr + lam_scaled * np.eye(len(power_list)), A_tr.T @ y_tr)
                pred = A_va @ coeff
                err = np.mean((pred - y_va)**2)
                if np.isfinite(err):
                    errors.append(err)
            except np.linalg.LinAlgError:
                continue
        return np.mean(errors) if errors else np.inf

    # Model variants to try
    model_variants = [
        # (power_list, description)
        ([0, 1], "linear"),
        ([0, 1, 2], "quadratic"),
        ([0, 1, 2, 3], "cubic"),
        ([0, 1, 2, 3, 4, 5], "full_degree5"),
        ([1, 3, 5], "odd_powers"),
        ([0, 1, 3, 5], "even+odd_mixed"),
        ([0, 2, 4], "even_powers"),
        ([1, 2, 3, 4, 5], "no_intercept"),
    ]
    
    lam_candidates = [1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e0, 1e2, 1e4, 1e6, 1e8, 1e10]
    
    # First pass: use train/val split for quick screening
    for power_list, desc in model_variants:
        if time.time() > deadline:
            break
        for lam in lam_candidates:
            if time.time() > deadline:
                break
            try:
                A_tr = np.column_stack([x_train**p for p in power_list])
                A_va = np.column_stack([x_val**p for p in power_list])
                lam_scaled = lam * np.trace(A_tr.T @ A_tr) / len(power_list)
                coeff_partial = np.linalg.solve(A_tr.T @ A_tr + lam_scaled * np.eye(len(power_list)), A_tr.T @ y_train)
                pred_val = A_va @ coeff_partial
                val_err = np.mean((pred_val - y_val)**2)
                if val_err < best_val_err and np.all(np.isfinite(coeff_partial)):
                    best_val_err = val_err
                    coeff_full = np.zeros(6)
                    for i, p in enumerate(power_list):
                        coeff_full[p] = coeff_partial[i]
                    best_coeff = coeff_full
            except np.linalg.LinAlgError:
                continue
    
    # Second pass: refine top candidates with k-fold CV
    # Re-collect candidates from first pass
    candidate_configs = []
    for power_list, desc in model_variants:
        for lam in lam_candidates[:6]:  # Focus on middle range for refinement
            if time.time() > deadline:
                break
            try:
                A_tr = np.column_stack([x_train**p for p in power_list])
                A_va = np.column_stack([x_val**p for p in power_list])
                lam_scaled = lam * np.trace(A_tr.T @ A_tr) / len(power_list)
                coeff_partial = np.linalg.solve(A_tr.T @ A_tr + lam_scaled * np.eye(len(power_list)), A_tr.T @ y_train)
                pred_val = A_va @ coeff_partial
                val_err = np.mean((pred_val - y_val)**2)
                if np.isfinite(val_err) and val_err < best_val_err * 1.5:
                    candidate_configs.append((power_list, lam, val_err))
            except np.linalg.LinAlgError:
                continue
    
    # Sort by val error and refine top ones with k-fold CV
    candidate_configs.sort(key=lambda c: c[2])
    for power_list, lam, _ in candidate_configs[:8]:
        if time.time() > deadline:
            break
        cv_err = kfold_cv(x, y, power_list, lam, n_folds=5)
        if np.isfinite(cv_err):
            # Use CV error as a better estimate
            if cv_err < best_val_err:
                best_val_err = cv_err
                A_full = np.column_stack([x**p for p in power_list])
                lam_scaled = lam * np.trace(A_full.T @ A_full) / len(power_list)
                try:
                    coeff_partial = np.linalg.solve(A_full.T @ A_full + lam_scaled * np.eye(len(power_list)), A_full.T @ y)
                    coeff_full = np.zeros(6)
                    for i, p in enumerate(power_list):
                        coeff_full[p] = coeff_partial[i]
                    if np.all(np.isfinite(coeff_full)):
                        best_coeff = coeff_full
                except np.linalg.LinAlgError:
                    continue

    # If no model worked, fall back to least squares linear
    if best_coeff is None or not np.all(np.isfinite(best_coeff)):
        try:
            c_lin = np.polynomial.polynomial.polyfit(x, y, 1)
            best_coeff = np.zeros(6)
            best_coeff[:2] = c_lin
        except Exception:
            best_coeff = np.zeros(6)

    # Final verification and polish
    if not np.all(np.isfinite(best_coeff)):
        best_coeff = np.zeros(6)
    
    # Optional: try a final refinement with all data for the best model structure
    # This can help if the val split was unlucky
    if best_coeff is not None and np.all(np.isfinite(best_coeff)) and time.time() < deadline - 10:
        # Determine which powers are active in best_coeff
        active_powers = [i for i in range(6) if abs(best_coeff[i]) > 1e-15]
        if len(active_powers) >= 1:
            for lam in [1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2]:
                if time.time() > deadline:
                    break
                try:
                    A_all = np.column_stack([x**p for p in active_powers])
                    lam_scaled = lam * np.trace(A_all.T @ A_all) / len(active_powers)
                    coeff_partial = np.linalg.solve(A_all.T @ A_all + lam_scaled * np.eye(len(active_powers)), A_all.T @ y)
                    coeff_candidate = np.zeros(6)
                    for i, p in enumerate(active_powers):
                        coeff_candidate[p] = coeff_partial[i]
                    if np.all(np.isfinite(coeff_candidate)):
                        # Check if this is better on the full training set
                        pred_all = np.column_stack([x**p for p in active_powers]) @ coeff_partial
                        train_mse = np.mean((pred_all - y)**2)
                        pred_best = np.column_stack([x**p for p in active_powers]) @ [best_coeff[p] for p in active_powers]
                        train_mse_best = np.mean((pred_best - y)**2)
                        if train_mse < train_mse_best * 0.99:  # Only accept if significantly better
                            best_coeff = coeff_candidate
                except np.linalg.LinAlgError:
                    continue

    return best_coeff
