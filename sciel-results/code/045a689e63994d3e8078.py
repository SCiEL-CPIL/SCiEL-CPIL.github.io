import numpy as np
import time
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

_DEADLINE = 300.0 - 20.0  # safety margin before 320s
_START = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _START)

def fit(X_train, y_train):
    X = X_train
    y = y_train
    N, D = X.shape

    # 1. Data Preprocessing
    # Sort by X
    sort_idx = np.argsort(X[:, 0])
    X_sorted = X[sort_idx, 0].flatten()
    y_sorted = y[sort_idx].flatten()

    # Handle duplicates in X
    unique_X, inverse_indices, counts = np.unique(X_sorted, return_inverse=True, return_counts=True)
    if len(unique_X) < len(X_sorted):
        y_sum = np.zeros(len(unique_X))
        np.add.at(y_sum, inverse_indices, y_sorted)
        y_sorted = y_sum / counts
        X_sorted = unique_X

    # Check for trivial constant function
    if np.allclose(y_sorted, y_sorted[0]):
        val = y_sorted[0]
        def predict_const(X_test):
            return np.full(X_test.shape[0], val)
        return predict_const

    # 2. Candidate Models
    # Candidate A: Cubic Spline (Interpolation)
    # Since data is noiseless and dense, interpolation is often perfect for 1D.
    try:
        cs = CubicSpline(X_sorted, y_sorted)
    except Exception:
        cs = None

    # Candidate B: Gaussian Process (Kernel Regression)
    # Useful if the function is smooth but interpolation oscillates, or for extrapolation.
    # We use a Matern 5/2 kernel which is flexible for smooth functions.
    
    # Normalize inputs for GP stability
    X_mean = np.mean(X_sorted)
    X_std = np.std(X_sorted)
    if X_std == 0:
        X_std = 1.0
    X_norm = (X_sorted - X_mean) / X_std
    
    # Normalize outputs
    y_mean = np.mean(y_sorted)
    y_std = np.std(y_sorted)
    if y_std == 0:
        y_std = 1.0
    y_norm = (y_sorted - y_mean) / y_std

    noise_var = 1e-6 * np.var(y_norm) # Small noise for numerical stability

    def compute_dist_sq(X1, X2):
        # X1, X2 are 1D arrays
        X1_sq = np.sum(X1**2)[:, None]
        X2_sq = np.sum(X2**2)[None, :]
        dist_sq = X1_sq + X2_sq - 2 * X1 @ X2.T
        return np.maximum(dist_sq, 0)

    def get_kernel(dist_sq, l, sf):
        # Matern 5/2
        r = np.sqrt(dist_sq) / l
        return sf * (1 + np.sqrt(3)*r + 0.5*r**2) * np.exp(-np.sqrt(3)*r)

    def log_likelihood(params, X_tr, y_tr):
        log_l, log_sf = params
        l = np.exp(log_l)
        sf = np.exp(log_sf)
        
        dist_sq = compute_dist_sq(X_tr, X_tr)
        K = get_kernel(dist_sq, l, sf)
        K += noise_var * np.eye(len(X_tr))
        
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return 1e10
            
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_tr))
        log_det_K = 2 * np.sum(np.log(np.diag(L)))
        
        nlml = 0.5 * np.dot(y_tr, alpha) + 0.5 * log_det_K + 0.5 * len(X_tr) * np.log(2 * np.pi)
        return nlml

    # Optimize GP Hyperparameters
    # Use a few random restarts to find good length scale and signal variance
    best_nlml = np.inf
    best_params = None
    rng = np.random.RandomState(42)
    
    # Time check: GP optimization can be slow, but N=200 is small.
    # We limit the number of iterations.
    if _time_left() > 5.0:
        for _ in range(3):
            init_log_l = np.random.uniform(-1, 2)
            init_log_sf = np.random.uniform(-2, 2)
            try:
                res = minimize(log_likelihood, [init_log_l, init_log_sf], method='Nelder-Mead', 
                               args=(X_norm, y_norm),
                               options={'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-4})
                if res.fun < best_nlml:
                    best_nlml = res.fun
                    best_params = res.x
            except Exception:
                continue

    if best_params is None:
        gp_predict = None
    else:
        log_l, log_sf = best_params
        l = np.exp(log_l)
        sf = np.exp(log_sf)
        
        dist_tr = compute_dist_sq(X_norm, X_norm)
        K_train = get_kernel(dist_tr, l, sf) + noise_var * np.eye(N)
        L = np.linalg.cholesky(K_train)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))
        
        def predict_gp(X_test):
            X_t = X_test.flatten()
            X_t_norm = (X_t - X_mean) / X_std
            dist_val = compute_dist_sq(X_t_norm, X_norm)
            K_test = get_kernel(dist_val, l, sf)
            pred_norm = K_test @ alpha
            pred = pred_norm * y_std + y_mean
            return pred
        
        gp_predict = predict_gp

    # 3. Model Selection
    # Since we don't have a separate validation set, and the data is noiseless,
    # we can use a heuristic:
    # - If the function appears smooth (low curvature), GP might be better for extrapolation.
    # - If the function is complex, Spline interpolation is exact within range.
    # Given the test points are "separate" but likely in the same range (uniform inputs),
    # Spline interpolation is extremely strong for 1D noiseless data.
    # However, GP is more robust if there's slight noise or if the function is very smooth.
    
    # Heuristic: Check the "effective" number of features or complexity.
    # Or simply, try both on a hold-out set? No, we use all 200.
    # Let's use a simple CV on a subset to decide.
    
    # 5-Fold CV on the training data to estimate error.
    # Since data is noiseless, CV error is purely interpolation error (if models are interpolating)
    # or approximation error (if not).
    # Spline is interpolating. GP is not strictly interpolating due to noise/regularization.
    
    idx = np.arange(N)
    rng.shuffle(idx)
    fold_size = N // 5
    folds = [idx[i*fold_size:(i+1)*fold_size] for i in range(5)]
    
    cv_scores_spline = []
    cv_scores_gp = []
    
    for i in range(5):
        val_idx = folds[i]
        tr_idx = np.setdiff1d(idx, val_idx)
        
        X_tr = X_sorted[tr_idx]
        y_tr = y_sorted[tr_idx]
        X_val = X_sorted[val_idx]
        y_val = y_sorted[val_idx]
        
        # Spline CV
        if cs is not None:
            try:
                cs_cv = CubicSpline(X_tr, y_tr)
                pred_cv = cs_cv(X_val)
                mse_cv = np.mean((pred_cv - y_val)**2)
                cv_scores_spline.append(mse_cv)
            except Exception:
                cv_scores_spline.append(1e10)
        
        # GP CV
        if gp_predict is not None:
            # We need to refit GP on tr_idx for proper CV.
            # This is expensive. Let's approximate or skip if time is tight.
            # For N=200, refitting 5 times is ~1s. Acceptable.
            if _time_left() > 10.0:
                X_tr_norm = (X_tr - X_mean) / X_std
                y_tr_norm = (y_tr - y_mean) / y_std
                
                # Use the same hyperparameters found globally for speed?
                # Ideally, we'd optimize per fold, but that's too slow.
                # Let's assume the global hyperparameters are good enough for comparison.
                dist_tr_cv = compute_dist_sq(X_tr_norm, X_tr_norm)
                K_tr_cv = get_kernel(dist_tr_cv, l, sf) + noise_var * np.eye(len(X_tr))
                try:
                    L_cv = np.linalg.cholesky(K_tr_cv)
                    alpha_cv = np.linalg.solve(L_cv.T, np.linalg.solve(L_cv, y_tr_norm))
                    dist_val_cv = compute_dist_sq(X_val_norm, X_tr_norm) if 'X_val_norm' in locals() else None
                    # Need X_val_norm
                    X_val_norm = (X_val - X_mean) / X_std
                    dist_val_cv = compute_dist_sq(X_val_norm, X_tr_norm)
                    K_val_cv = get_kernel(dist_val_cv, l, sf)
                    pred_norm_cv = K_val_cv @ alpha_cv
                    pred_cv = pred_norm_cv * y_std + y_mean
                    mse_cv = np.mean((pred_cv - y_val)**2)
                    cv_scores_gp.append(mse_cv)
                except Exception:
                    cv_scores_gp.append(1e10)
            else:
                cv_scores_gp.append(1e10) # Skip GP CV if time is low

    mean_cv_spline = np.mean(cv_scores_spline) if cv_scores_spline else 1e10
    mean_cv_gp = np.mean(cv_scores_gp) if cv_scores_gp else 1e10

    # Select the better model
    # Spline is usually better for noiseless 1D interpolation.
    # GP is better if the function is very smooth and we want to avoid spline wiggles.
    
    if mean_cv_gp < mean_cv_spline * 0.5: # GP is significantly better
        # Use GP
        if gp_predict is not None:
            return gp_predict
        else:
            # Fallback to spline
            if cs is not None:
                def predict_spline(X_test):
                    x_test = X_test.flatten()
                    return cs(x_test)
                return predict_spline
            else:
                # Fallback to mean
                val = y_sorted[0]
                def predict_const(X_test):
                    return np.full(X_test.shape[0], val)
                return predict_const
    else:
        # Use Spline
        if cs is not None:
            def predict_spline(X_test):
                x_test = X_test.flatten()
                return cs(x_test)
            return predict_spline
        else:
            # Fallback to GP
            if gp_predict is not None:
                return gp_predict
            else:
                # Fallback to mean
                val = y_sorted[0]
                def predict_const(X_test):
                    return np.full(X_test.shape[0], val)
                return predict_const
