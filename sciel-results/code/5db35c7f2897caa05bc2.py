"""Symbolic regression over 10 hidden equations: multi-stage oracle selection."""
# EVOLVE-BLOCK-START
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def _kfold_indices(n, k=5):
    """Generate k-fold indices."""
    indices = np.arange(n)
    np.random.shuffle(indices)
    folds = [indices[i::k] for i in range(k)]
    return folds

def _cv_krr(X, y, K, alpha_grid, n_folds=5):
    """K-Fold CV for Kernel Ridge Regression.
    K is the precomputed kernel matrix.
    Returns (best_mse, best_alpha).
    """
    n = len(y)
    folds = _kfold_indices(n, n_folds)
    best_mse = np.inf
    best_alpha = None
    
    for alpha in alpha_grid:
        total_mse = 0.0
        count = 0
        for i in range(n_folds):
            test_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
            
            K_train = K[np.ix_(train_idx, train_idx)]
            K_test = K[np.ix_(test_idx, train_idx)]
            y_train = y[train_idx]
            y_test = y[test_idx]
            
            K_reg = K_train + alpha * np.eye(len(train_idx))
            try:
                w = np.linalg.solve(K_reg, y_train)
                y_pred = K_test @ w
                mse = np.mean((y_test - y_pred) ** 2)
                total_mse += mse
                count += 1
            except (np.linalg.LinAlgError, ValueError):
                continue
        
        if count == n_folds:
            avg_mse = total_mse / count
            if avg_mse < best_mse:
                best_mse = avg_mse
                best_alpha = alpha
                
    return best_mse, best_alpha

def _cv_gp(X, y, kernel_type, length_scale_grid, variance_grid, noise_grid, n_folds=5):
    """K-Fold CV for Gaussian Process.
    kernel_type: 'matern52' or 'rbf'
    Returns (best_mse, best_params).
    """
    n = len(y)
    folds = _kfold_indices(n, n_folds)
    best_mse = np.inf
    best_params = None
    
    # Precompute squared distances
    X_sq = np.sum(X ** 2, axis=1)
    sq_dists = X_sq[:, None] + X_sq[None, :] - 2 * X @ X.T
    sq_dists = np.maximum(sq_dists, 0)
    dists = np.sqrt(sq_dists)
    
    for ls in length_scale_grid:
        for var in variance_grid:
            for noise in noise_grid:
                r = dists / ls
                if kernel_type == 'matern52':
                    K_prior = var * (1 + r + r ** 2 / 3) * np.exp(-r)
                else: # rbf
                    K_prior = var * np.exp(-0.5 * (r ** 2) / (ls ** 2)) # Standard RBF: exp(-0.5 * d^2 / l^2)
                
                total_mse = 0.0
                count = 0
                for i in range(n_folds):
                    test_idx = folds[i]
                    train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
                    
                    K_train = K_prior[np.ix_(train_idx, train_idx)]
                    K_test = K_prior[np.ix_(test_idx, train_idx)]
                    y_train = y[train_idx]
                    y_test = y[test_idx]
                    
                    K_obs = K_train + noise * np.eye(len(train_idx))
                    try:
                        w = np.linalg.solve(K_obs, y_train)
                        y_pred = K_test @ w
                        mse = np.mean((y_test - y_pred) ** 2)
                        total_mse += mse
                        count += 1
                    except (np.linalg.LinAlgError, ValueError):
                        continue
                
                if count == n_folds:
                    avg_mse = total_mse / count
                    if avg_mse < best_mse:
                        best_mse = avg_mse
                        best_params = (ls, var, noise)
                        
    return best_mse, best_params

def _cv_ridge(X, y, basis_fn, alpha_grid, n_folds=5):
    """K-Fold CV for Ridge Regression with custom basis.
    Returns (best_mse, best_alpha, B_train).
    """
    n = len(y)
    folds = _kfold_indices(n, n_folds)
    B_full = basis_fn(X)
    best_mse = np.inf
    best_alpha = None
    
    for alpha in alpha_grid:
        total_mse = 0.0
        count = 0
        for i in range(n_folds):
            test_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
            
            B_train = B_full[train_idx]
            B_test = B_full[test_idx]
            y_train = y[train_idx]
            y_test = y[test_idx]
            
            m = B_train.shape[1]
            A = B_train.T @ B_train + alpha * np.eye(m)
            b = B_train.T @ y_train
            try:
                w = np.linalg.solve(A, b)
                y_pred = B_test @ w
                mse = np.mean((y_test - y_pred) ** 2)
                total_mse += mse
                count += 1
            except (np.linalg.LinAlgError, ValueError):
                continue
        
        if count == n_folds:
            avg_mse = total_mse / count
            if avg_mse < best_mse:
                best_mse = avg_mse
                best_alpha = alpha
                
    return best_mse, best_alpha, B_full


def fit(X_train, y_train):
    X = np.array(X_train, dtype=np.float64)
    y = np.array(y_train, dtype=np.float64)
    n, d = X.shape
    
    # Standardize inputs
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std < 1e-10] = 1.0
    X_norm = (X - X_mean) / X_std
    
    models = []
    cv_scores = []
    
    # Precompute kernels for KRR models
    def poly_kernel(X1, X2, deg):
        return (1 + X1 @ X2.T / d) ** deg
        
    alpha_grid_krr = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    
    # Model 1: Polynomial KRR degree 2
    K_poly2 = poly_kernel(X_norm, X_norm, 2)
    mse1, alpha1 = _cv_krr(X_norm, y, K_poly2, alpha_grid_krr)
    if mse1 < np.inf and alpha1 is not None:
        K_reg = K_poly2 + alpha1 * np.eye(n)
        w = np.linalg.solve(K_reg, y)
        def pred_krr2(X_test):
            X_test = np.array(X_test, dtype=np.float64)
            X_test_norm = (X_test - X_mean) / X_std
            K_test = poly_kernel(X_test_norm, X_norm, 2)
            return K_test @ w
        models.append(('krr_poly2', pred_krr2))
        cv_scores.append(mse1)
        
    # Model 2: Polynomial KRR degree 3
    K_poly3 = poly_kernel(X_norm, X_norm, 3)
    mse2, alpha2 = _cv_krr(X_norm, y, K_poly3, alpha_grid_krr)
    if mse2 < np.inf and alpha2 is not None:
        K_reg = K_poly3 + alpha2 * np.eye(n)
        w = np.linalg.solve(K_reg, y)
        def pred_krr3(X_test):
            X_test = np.array(X_test, dtype=np.float64)
            X_test_norm = (X_test - X_mean) / X_std
            K_test = poly_kernel(X_test_norm, X_norm, 3)
            return K_test @ w
        models.append(('krr_poly3', pred_krr3))
        cv_scores.append(mse2)
        
    # Model 3: Polynomial KRR degree 4
    K_poly4 = poly_kernel(X_norm, X_norm, 4)
    mse3, alpha3 = _cv_krr(X_norm, y, K_poly4, alpha_grid_krr)
    if mse3 < np.inf and alpha3 is not None:
        K_reg = K_poly4 + alpha3 * np.eye(n)
        w = np.linalg.solve(K_reg, y)
        def pred_krr4(X_test):
            X_test = np.array(X_test, dtype=np.float64)
            X_test_norm = (X_test - X_mean) / X_std
            K_test = poly_kernel(X_test_norm, X_norm, 4)
            return K_test @ w
        models.append(('krr_poly4', pred_krr4))
        cv_scores.append(mse3)
        
    # Model 4: Matern-5/2 Gaussian Process
    length_scale_grid = [0.2, 0.4, 0.7, 1.0, 1.5, 2.5]
    variance_grid = [0.5, 1.0, 2.0, 5.0]
    noise_grid = [0.001, 0.01, 0.1, 1.0]
    
    mse4, gp_params = _cv_gp(X_norm, y, 'matern52', length_scale_grid, variance_grid, noise_grid)
    if mse4 < np.inf and gp_params is not None:
        ls, var, noise = gp_params
        # Recompute kernel for prediction
        X_sq = np.sum(X_norm ** 2, axis=1)
        sq_dists = X_sq[:, None] + X_sq[None, :] - 2 * X_norm @ X_norm.T
        sq_dists = np.maximum(sq_dists, 0)
        dists = np.sqrt(sq_dists)
        r = dists / ls
        K_prior = var * (1 + r + r ** 2 / 3) * np.exp(-r)
        K_obs = K_prior + noise * np.eye(n)
        w_gp = np.linalg.solve(K_obs, y)
        
        def pred_gp(X_test):
            X_test = np.array(X_test, dtype=np.float64)
            X_test_norm = (X_test - X_mean) / X_std
            X_test_sq = np.sum(X_test_norm ** 2, axis=1)
            sq_dists_test = X_test_sq[:, None] + X_sq[None, :] - 2 * X_test_norm @ X_norm.T
            sq_dists_test = np.maximum(sq_dists_test, 0)
            dists_test = np.sqrt(sq_dists_test)
            r_test = dists_test / ls
            K_test = var * (1 + r_test + r_test ** 2 / 3) * np.exp(-r_test)
            return K_test @ w_gp
        models.append(('gp_matern52', pred_gp))
        cv_scores.append(mse4)
        
    # Model 5: RBF Gaussian Process
    mse5, rbf_params = _cv_gp(X_norm, y, 'rbf', length_scale_grid, variance_grid, noise_grid)
    if mse5 < np.inf and rbf_params is not None:
        ls, var, noise = rbf_params
        X_sq = np.sum(X_norm ** 2, axis=1)
        sq_dists = X_sq[:, None] + X_sq[None, :] - 2 * X_norm @ X_norm.T
        sq_dists = np.maximum(sq_dists, 0)
        dists = np.sqrt(sq_dists)
        r = dists / ls
        K_prior = var * np.exp(-0.5 * (r ** 2) / (ls ** 2))
        K_obs = K_prior + noise * np.eye(n)
        w_rbf = np.linalg.solve(K_obs, y)
        
        def pred_rbf(X_test):
            X_test = np.array(X_test, dtype=np.float64)
            X_test_norm = (X_test - X_mean) / X_std
            X_test_sq = np.sum(X_test_norm ** 2, axis=1)
            sq_dists_test = X_test_sq[:, None] + X_sq[None, :] - 2 * X_test_norm @ X_norm.T
            sq_dists_test = np.maximum(sq_dists_test, 0)
            dists_test = np.sqrt(sq_dists_test)
            r_test = dists_test / ls
            K_test = var * np.exp(-0.5 * (r_test ** 2) / (ls ** 2))
            return K_test @ w_rbf
        models.append(('gp_rbf', pred_rbf))
        cv_scores.append(mse5)
        
    # Model 6: Rational-style basis
    def build_rational_basis(X_in):
        n_in, d_in = X_in.shape
        cols = [np.ones(n_in)]
        for j in range(d_in):
            cols.append(X_in[:, j])
            cols.append(1.0 / np.maximum(X_in[:, j], 0.1))
        for i in range(d_in):
            for j in range(i, d_in):
                cols.append(X_in[:, i] * X_in[:, j])
        for j in range(d_in):
            cols.append(1.0 / np.maximum(X_in[:, j], 0.1) ** 2)
            cols.append(X_in[:, j] ** 3)
        return np.column_stack(cols)
        
    alpha_grid_rat = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    mse6, alpha6, B6_full = _cv_ridge(X, y, build_rational_basis, alpha_grid_rat)
    if mse6 < np.inf and alpha6 is not None:
        A6 = B6_full.T @ B6_full + alpha6 * np.eye(B6_full.shape[1])
        w6 = np.linalg.solve(A6, B6_full.T @ y)
        
        def pred_rat(X_test):
            X_test = np.array(X_test, dtype=np.float64)
            B_test = build_rational_basis(X_test)
            return B_test @ w6
        models.append(('rational_basis', pred_rat))
        cv_scores.append(mse6)
        
    # Model 7: Linear Ridge (simple baseline)
    def build_linear_basis(X_in):
        return np.column_stack([np.ones(len(X_in)), X_in])
        
    alpha_grid_lin = [0.001, 0.01, 0.1, 1.0, 10.0]
    mse7, alpha7, B7_full = _cv_ridge(X_norm, y, build_linear_basis, alpha_grid_lin)
    if mse7 < np.inf and alpha7 is not None:
        A7 = B7_full.T @ B7_full + alpha7 * np.eye(B7_full.shape[1])
        w7 = np.linalg.solve(A7, B7_full.T @ y)
        def pred_lin(X_test):
            X_test = np.array(X_test, dtype=np.float64)
            X_test_norm = (X_test - X_mean) / X_std
            B_test = build_linear_basis(X_test_norm)
            return B_test @ w7
        models.append(('linear_ridge', pred_lin))
        cv_scores.append(mse7)
    
    # Select best model based on CV score
    if not models:
        y_mean = np.mean(y)
        return lambda X: np.full(len(X), y_mean)
    
    cv_scores = np.array(cv_scores)
    best_idx = np.argmin(cv_scores)
    best_predict_fn = models[best_idx][1]
    
    # Simple ensemble if top 2 are very close
    sorted_indices = np.argsort(cv_scores)
    if len(sorted_indices) >= 2 and cv_scores[sorted_indices[1]] < 1.05 * cv_scores[sorted_indices[0]]:
        pred_fn1 = models[sorted_indices[0]][1]
        pred_fn2 = models[sorted_indices[1]][1]
        def predict_ensemble(X_test):
            X_test = np.array(X_test, dtype=np.float64)
            try:
                p1 = pred_fn1(X_test)
                p2 = pred_fn2(X_test)
                p1 = np.where(np.isfinite(p1), p1, np.mean(y))
                p2 = np.where(np.isfinite(p2), p2, np.mean(y))
                return 0.5 * p1 + 0.5 * p2
            except Exception:
                return np.full(len(X_test), np.mean(y))
        return predict_ensemble
    
    def predict(X_test):
        X_test = np.array(X_test, dtype=np.float64)
        try:
            preds = best_predict_fn(X_test)
            preds = np.where(np.isfinite(preds), preds, np.mean(y))
            return preds
        except Exception:
            return np.full(len(X_test), np.mean(y))
    
    return predict
# EVOLVE-BLOCK-END
