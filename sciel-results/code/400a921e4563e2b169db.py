"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.interpolate import interp1d


def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 290  # Safety margin before 300s limit
    
    X = X_train.astype(np.float64)
    y = y_train.astype(np.float64)
    n, d = X.shape
    
    # For 1D case, use the single input dimension
    if d == 1:
        x = X[:, 0].copy()
    else:
        x = X  # fallback for multi-dim (shouldn't happen per task spec)
    
    # Sort by x for better behavior
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    
    best_predict = None
    best_score = np.inf
    
    def make_gaussian_predict(length_scale, signal_var, x_train, y_train):
        """Create a predictor using Gaussian process-like interpolation."""
        def predict(X):
            x_test = X[:, 0] if X.shape[1] == 1 else X.ravel()
            # Compute kernel matrix K(x_test, x_train)
            diff = x_test[:, np.newaxis] - x_train[np.newaxis, :]
            K = np.exp(-0.5 * (diff / length_scale) ** 2) * signal_var
            # For noiseless data, we can use the training values directly
            # Use weighted interpolation
            weights = K
            # Normalize weights
            weight_sums = weights.sum(axis=1, keepdims=True)
            weight_sums[weight_sums == 0] = 1
            predictions = (weights * y_train[np.newaxis, :]).sum(axis=1) / weight_sums[:, 0]
            return predictions
        return predict
    
    def make_gp_predict(length_scale, x_train, y_train):
        """Create a predictor using Gaussian process interpolation (exact for noiseless)."""
        def predict(X):
            x_test = X[:, 0] if X.shape[1] == 1 else X.ravel()
            # Compute kernel matrices
            K_train = np.exp(-0.5 * ((x_train[:, np.newaxis] - x_train[np.newaxis, :]) / length_scale) ** 2)
            K_test = np.exp(-0.5 * ((x_test[:, np.newaxis] - x_train[np.newaxis, :]) / length_scale) ** 2)
            
            # For noiseless data, the posterior mean is K_test @ K_train^{-1} @ y_train
            # Add small regularization for numerical stability
            K_train_reg = K_train + 1e-8 * np.eye(len(x_train))
            
            try:
                # Solve K_train @ alpha = y_train
                alpha = np.linalg.solve(K_train_reg, y_train)
                predictions = K_test @ alpha
                return predictions
            except np.linalg.LinAlgError:
                # Fallback to pseudo-inverse
                alpha = np.linalg.lstsq(K_train_reg, y_train, rcond=None)[0]
                predictions = K_test @ alpha
                return predictions
        return predict
    
    def make_rbf_predict(epsilon, smooth, x_train, y_train):
        """Create a predictor using RBF interpolation."""
        from scipy.interpolate import Rbf
        try:
            rbf = Rbf(x_train, y_train, function='gaussian', epsilon=epsilon, smooth=smooth)
            def predict(X):
                x_test = X[:, 0] if X.shape[1] == 1 else X.ravel()
                return rbf(x_test)
            return predict
        except Exception:
            return None
    
    def make_poly_predict(degree, x_train, y_train):
        """Create a polynomial predictor."""
        try:
            coeffs = np.polyfit(x_train, y_train, degree)
            def predict(X):
                x_test = X[:, 0] if X.shape[1] == 1 else X.ravel()
                return np.polyval(coeffs, x_test)
            return predict
        except Exception:
            return None
    
    def make_interp_predict(x_train, y_train, kind='cubic'):
        """Create an interpolation-based predictor."""
        try:
            if kind == 'cubic':
                f = interp1d(x_train, y_train, kind='cubic', fill_value='extrapolate')
            else:
                f = interp1d(x_train, y_train, kind='linear', fill_value='extrapolate')
            def predict(X):
                x_test = X[:, 0] if X.shape[1] == 1 else X.ravel()
                return f(x_test)
            return predict
        except Exception:
            return None
    
    def evaluate_predictor(predict_func, x_eval, y_eval):
        """Evaluate a predictor function."""
        try:
            pred = predict_func(np.column_stack([x_eval]))
            if not np.all(np.isfinite(pred)):
                return np.inf
            mse = np.mean((pred - y_eval) ** 2)
            var_y = np.var(y_eval)
            if var_y < 1e-15:
                return np.inf
            return mse / var_y
        except Exception:
            return np.inf
    
    # Strategy 1: Gaussian Process with optimized length scale
    # Use cross-validation to find good length scale
    x_range = x_sorted.max() - x_sorted.min()
    if x_range < 1e-10:
        x_range = 1.0
    
    # Try various length scales
    candidate_lengths = []
    for log_ls in np.arange(-1.5, 2.5, 0.25):
        ls = 10 ** log_ls * x_range
        if 0 < ls < 10 * x_range:
            candidate_lengths.append(ls)
    
    # Add some fixed reasonable values
    for factor in [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]:
        ls = factor * x_range / 10
        if ls > 0:
            candidate_lengths.append(ls)
    
    candidate_lengths = sorted(set(candidate_lengths))
    
    # Use a subset if too many
    if len(candidate_lengths) > 40:
        indices = np.linspace(0, len(candidate_lengths) - 1, 40).astype(int)
        candidate_lengths = [candidate_lengths[i] for i in indices]
    
    # Evaluate with 5-fold CV
    n_folds = 5
    folds = np.array_split(np.arange(n), n_folds)
    
    gp_results = []
    for ls in candidate_lengths:
        if time.time() > deadline - 40:
            break
        cv_errors = []
        for k in range(n_folds):
            if time.time() > deadline - 35:
                break
            val_idx = folds[k]
            train_idx = np.concatenate([folds[i] for i in range(n_folds) if i != k])
            
            x_train_cv = x_sorted[train_idx]
            y_train_cv = y_sorted[train_idx]
            x_val_cv = x_sorted[val_idx]
            y_val_cv = y_sorted[val_idx]
            
            predict_func = make_gp_predict(ls, x_train_cv, y_train_cv)
            if predict_func is not None:
                cv_err = evaluate_predictor(predict_func, x_val_cv, y_val_cv)
                if np.isfinite(cv_err):
                    cv_errors.append(cv_err)
        
        if cv_errors:
            mean_cv_err = np.mean(cv_errors)
            gp_results.append((mean_cv_err, ls))
    
    # Sort by CV error and try top few
    gp_results.sort(key=lambda x: x[0])
    
    for cv_err, ls in gp_results[:5]:
        if time.time() > deadline - 20:
            break
        predict_func = make_gp_predict(ls, x_sorted, y_sorted)
        if predict_func is not None:
            train_score = evaluate_predictor(predict_func, x_sorted, y_sorted)
            if train_score < best_score:
                best_score = train_score
                best_predict = predict_func
    
    # Strategy 2: RBF with various parameters
    if time.time() < deadline - 15:
        rbf_results = []
        for log_eps in np.arange(-2, 3, 0.25):
            if time.time() > deadline - 12:
                break
            eps = 10 ** log_eps * x_range / 10
            if eps <= 0:
                continue
            for smooth in [0.01, 0.1, 1.0]:
                if time.time() > deadline - 10:
                    break
                cv_errors = []
                for k in range(n_folds):
                    if time.time() > deadline - 8:
                        break
                    val_idx = folds[k]
                    train_idx = np.concatenate([folds[i] for i in range(n_folds) if i != k])
                    
                    x_train_cv = x_sorted[train_idx]
                    y_train_cv = y_sorted[train_idx]
                    x_val_cv = x_sorted[val_idx]
                    y_val_cv = y_sorted[val_idx]
                    
                    predict_func = make_rbf_predict(eps, smooth, x_train_cv, y_train_cv)
                    if predict_func is not None:
                        cv_err = evaluate_predictor(predict_func, x_val_cv, y_val_cv)
                        if np.isfinite(cv_err):
                            cv_errors.append(cv_err)
                
                if cv_errors:
                    mean_cv_err = np.mean(cv_errors)
                    rbf_results.append((mean_cv_err, eps, smooth))
        
        rbf_results.sort(key=lambda x: x[0])
        
        for cv_err, eps, smooth in rbf_results[:5]:
            if time.time() > deadline - 5:
                break
            predict_func = make_rbf_predict(eps, smooth, x_sorted, y_sorted)
            if predict_func is not None:
                train_score = evaluate_predictor(predict_func, x_sorted, y_sorted)
                if train_score < best_score:
                    best_score = train_score
                    best_predict = predict_func
    
    # Strategy 3: Polynomial regression
    if time.time() < deadline - 5:
        for degree in [2, 3, 4, 5, 6, 7, 8, 10, 12]:
            if time.time() > deadline - 3:
                break
            predict_func = make_poly_predict(degree, x_sorted, y_sorted)
            if predict_func is not None:
                train_score = evaluate_predictor(predict_func, x_sorted, y_sorted)
                if train_score < best_score:
                    best_score = train_score
                    best_predict = predict_func
    
    # Strategy 4: Cubic interpolation
    if time.time() < deadline - 2:
        predict_func = make_interp_predict(x_sorted, y_sorted, kind='cubic')
        if predict_func is not None:
            train_score = evaluate_predictor(predict_func, x_sorted, y_sorted)
            if train_score < best_score:
                best_score = train_score
                best_predict = predict_func
    
    # Final fallback: constant predictor (mean)
    if best_predict is None:
        mean_y = np.mean(y)
        best_predict = lambda X: np.full(X.shape[0], mean_y)
    
    return best_predict
# EVOLVE-BLOCK-END
