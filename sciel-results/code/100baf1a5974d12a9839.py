"""Symbolic regression via composite kernel ridge regression (RBF + Polynomial + Periodic) with adaptive hyperparameter search."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit(X_train, y_train):
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape

    # Deadline guard: 330s total, leave 10s margin
    deadline = time.time() + 320.0

    # Normalize inputs to [0, 1] with small padding to handle test points near boundaries
    X_min = X.min(axis=0, keepdims=True)
    X_max = X.max(axis=0, keepdims=True)
    X_range = X_max - X_min
    X_range[X_range < 1e-12] = 1.0
    padding = 0.05 * X_range
    X_min_p = X_min - padding
    X_max_p = X_max + padding
    X_range_p = X_max_p - X_min_p
    X_norm = (X - X_min_p) / X_range_p

    # Precompute pairwise squared distances for RBF and Periodic kernels
    diff = X_norm[:, None, :] - X_norm[None, :, :]
    sq_dist = np.sum(diff ** 2, axis=2)

    # Precompute polynomial kernel matrices for degrees 2 and 3
    # Degree 2: (gamma * <x, z> + coef0)^2
    # Degree 3: (gamma * <x, z> + coef0)^3
    # We'll use gamma=1.0 and coef0=1.0 for simplicity, or search a bit
    # For speed, precompute the dot product matrix
    dot_prod = X_norm @ X_norm.T

    # Candidate hyperparameters
    # RBF: length_scale (l2 = 1/(2*sigma^2) => length_scale = 1/sqrt(2*l2))
    # Let's parametrize by l2 for RBF: exp(-l2 * sq_dist)
    l2_candidates = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
    
    # Polynomial: degree, gamma, coef0
    # Keep degrees small to avoid explosion: 2, 3
    poly_degrees = [2, 3]
    poly_gammas = [0.5, 1.0, 2.0]
    poly_coef0s = [1.0]
    
    # Periodic: period, gamma
    # K_per(x,z) = exp(-gamma * sum_i sin^2(pi * (x_i - z_i) / period))
    # Precompute sin^2 terms for each dimension and period
    periods = [1.0, 2.0, 4.0]
    per_gammas = [0.5, 1.0, 2.0]
    
    # Weights for composite kernel: w_rbf, w_poly, w_per
    # To keep search space manageable, we'll fix weights or search a small grid
    # Let's try a few weight combinations
    weight_combos = [
        (1.0, 0.0, 0.0),  # RBF only
        (0.5, 0.5, 0.0),  # RBF + Poly
        (0.5, 0.0, 0.5),  # RBF + Per
        (0.33, 0.33, 0.34), # All three
        (1.0, 0.5, 0.5),
        (0.5, 1.0, 0.5),
        (0.5, 0.5, 1.0),
    ]
    
    # Regularization
    alpha_candidates = [1e-4, 1e-3, 1e-2, 1e-1]

    # Validation split
    rng = np.random.RandomState(42)
    idx = np.arange(n)
    rng.shuffle(idx)
    val_frac = 0.25
    val_size = max(1, int(n * val_frac))
    val_idx = idx[:val_size]
    train_idx = idx[val_size:]
    
    y_val = y[val_idx]
    y_train_split = y[train_idx]
    
    # Precompute kernel components for train and val splits
    # K_train_train, K_val_train, K_train_val, K_val_val
    # We need to compute kernels for different hyperparameter combos
    
    best_score = np.inf
    best_params = None
    
    # To speed up, we can precompute some parts
    # For RBF: K_rbf = exp(-l2 * sq_dist)
    # For Poly: K_poly = (gamma * dot_prod + coef0)^degree
    # For Per: K_per = exp(-gamma * sum_i sin^2(pi * diff_i / period))
    
    # Precompute sin^2 terms for each dimension and period
    # sin^2(pi * diff_i / period)
    sin_sq_cache = {}
    for period in periods:
        sin_sq_cache[period] = np.zeros_like(sq_dist)
        for i in range(d):
            diff_i = diff[:, :, i]
            sin_sq_cache[period] += np.sin(np.pi * diff_i / period) ** 2
    
    # Loop through hyperparameters
    # This might be slow if we do a full grid. Let's be selective.
    # Total combos: 8 l2 * 2 deg * 3 gamma * 1 coef0 * 3 period * 3 gamma_per * 7 weights * 4 alpha
    # = 8 * 2 * 3 * 1 * 3 * 3 * 7 * 4 = 12096 combos.
    # Each combo involves solving a 150x150 system (train split is 75% of 200 = 150).
    # 12096 * (kernel computation + solve) might be too slow.
    # Let's reduce the grid or use a smarter search.
    
    # Strategy:
    # 1. First, find the best RBF-only parameters (l2, alpha) using a coarse grid.
    # 2. Then, for each good RBF, try adding Poly and Per with a few weights.
    # 3. Or, just use a smaller grid.
    
    # Let's reduce the grid:
    l2_candidates = [1.0, 2.0, 4.0, 8.0, 16.0]  # 5
    poly_degrees = [2, 3]  # 2
    poly_gammas = [1.0]  # 1
    periods = [2.0, 4.0]  # 2
    per_gammas = [1.0]  # 1
    weight_combos = [
        (1.0, 0.0, 0.0),
        (0.5, 0.5, 0.0),
        (0.5, 0.0, 0.5),
        (0.33, 0.33, 0.34),
    ]  # 4
    alpha_candidates = [1e-4, 1e-3, 1e-2]  # 3
    
    # Total: 5 * 2 * 1 * 2 * 1 * 4 * 3 = 240 combos.
    # 240 solves of 150x150 is fast.
    
    for l2 in l2_candidates:
        if time.time() > deadline - 30:
            break
        K_rbf_train = np.exp(-l2 * sq_dist[np.ix_(train_idx, train_idx)])
        K_rbf_val_train = np.exp(-l2 * sq_dist[np.ix_(val_idx, train_idx)])
        
        for p_deg in poly_degrees:
            if time.time() > deadline - 30:
                break
            for p_gamma in poly_gammas:
                for p_coef0 in poly_coef0s:
                    K_poly_train = (p_gamma * dot_prod[np.ix_(train_idx, train_idx)] + p_coef0) ** p_deg
                    K_poly_val_train = (p_gamma * dot_prod[np.ix_(val_idx, train_idx)] + p_coef0) ** p_deg
                    
                    for period in periods:
                        if time.time() > deadline - 30:
                            break
                        for per_gamma in per_gammas:
                            K_per_train = np.exp(-per_gamma * sin_sq_cache[period][np.ix_(train_idx, train_idx)])
                            K_per_val_train = np.exp(-per_gamma * sin_sq_cache[period][np.ix_(val_idx, train_idx)])
                            
                            for w_rbf, w_poly, w_per in weight_combos:
                                if time.time() > deadline - 30:
                                    break
                                K_train_total = w_rbf * K_rbf_train + w_poly * K_poly_train + w_per * K_per_train
                                K_val_train_total = w_rbf * K_rbf_val_train + w_poly * K_poly_val_train + w_per * K_per_val_train
                                
                                for alpha in alpha_candidates:
                                    if time.time() > deadline - 30:
                                        break
                                    A = K_train_total + alpha * np.eye(len(train_idx))
                                    try:
                                        w = np.linalg.solve(A, y_train_split)
                                    except np.linalg.LinAlgError:
                                        continue
                                    
                                    pred_val = K_val_train_total @ w
                                    mse_val = np.mean((pred_val - y_val) ** 2)
                                    
                                    if mse_val < best_score:
                                        best_score = mse_val
                                        best_params = {
                                            'l2': l2,
                                            'p_deg': p_deg,
                                            'p_gamma': p_gamma,
                                            'p_coef0': p_coef0,
                                            'period': period,
                                            'per_gamma': per_gamma,
                                            'w_rbf': w_rbf,
                                            'w_poly': w_poly,
                                            'w_per': w_per,
                                            'alpha': alpha
                                        }
    
    # If no valid parameters found, use defaults
    if best_params is None:
        best_params = {
            'l2': 4.0,
            'p_deg': 2,
            'p_gamma': 1.0,
            'p_coef0': 1.0,
            'period': 2.0,
            'per_gamma': 1.0,
            'w_rbf': 1.0,
            'w_poly': 0.0,
            'w_per': 0.0,
            'alpha': 1e-3
        }
    
    # Final fit on all data
    l2 = best_params['l2']
    p_deg = best_params['p_deg']
    p_gamma = best_params['p_gamma']
    p_coef0 = best_params['p_coef0']
    period = best_params['period']
    per_gamma = best_params['per_gamma']
    w_rbf = best_params['w_rbf']
    w_poly = best_params['w_poly']
    w_per = best_params['w_per']
    alpha = best_params['alpha']
    
    K_rbf_full = np.exp(-l2 * sq_dist)
    K_poly_full = (p_gamma * dot_prod + p_coef0) ** p_deg
    K_per_full = np.exp(-per_gamma * sin_sq_cache[period])
    
    K_full = w_rbf * K_rbf_full + w_poly * K_poly_full + w_per * K_per_full
    A_full = K_full + alpha * np.eye(n)
    
    try:
        w_final = np.linalg.solve(A_full, y)
    except np.linalg.LinAlgError:
        # Fallback to RBF only if composite kernel fails
        K_rbf_only = np.exp(-4.0 * sq_dist)
        A_fallback = K_rbf_only + 1e-3 * np.eye(n)
        w_final = np.linalg.solve(A_fallback, y)
        # Update predict to use RBF only
        def predict(X_new):
            X_new = np.asarray(X_new, dtype=np.float64)
            X_new_norm = (X_new - X_min_p) / X_range_p
            diff_new = X_new_norm[:, None, :] - X_norm[None, :, :]
            sq_dist_new = np.sum(diff_new ** 2, axis=2)
            K_new = np.exp(-4.0 * sq_dist_new)
            return K_new @ w_final
        return predict

    def predict(X_new):
        X_new = np.asarray(X_new, dtype=np.float64)
        X_new_norm = (X_new - X_new.min(axis=0, keepdims=True)) / (X_new.max(axis=0, keepdims=True) - X_new.min(axis=0, keepdims=True) + 1e-12)
        # Wait, we must use the same normalization as training
        X_new_norm = (X_new - X_min_p) / X_range_p
        
        # Compute kernels between new points and training points
        diff_new = X_new_norm[:, None, :] - X_norm[None, :, :]
        sq_dist_new = np.sum(diff_new ** 2, axis=2)
        
        K_rbf_new = np.exp(-l2 * sq_dist_new)
        dot_prod_new = X_new_norm @ X_norm.T
        K_poly_new = (p_gamma * dot_prod_new + p_coef0) ** p_deg
        
        # Periodic kernel for new points
        K_per_new = np.zeros_like(K_rbf_new)
        for i in range(d):
            diff_i_new = diff_new[:, :, i]
            K_per_new += np.exp(-per_gamma * np.sin(np.pi * diff_i_new / period) ** 2)
        
        K_new = w_rbf * K_rbf_new + w_poly * K_poly_new + w_per * K_per_new
        
        return K_new @ w_final

    return predict
# EVOLVE-BLOCK-END
