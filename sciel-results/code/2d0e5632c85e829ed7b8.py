import numpy as np
import time
from itertools import product

def _rbf_kernel(X1, X2, length_scale):
    """Compute RBF kernel between two sets of points."""
    sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
    sq_dists = np.maximum(sq_dists, 0)
    return np.exp(-sq_dists / (2 * length_scale**2))

def _poly_kernel(X1, X2, degree, gamma, coef0):
    """Compute polynomial kernel between two sets of points."""
    return (gamma * X1 @ X2.T + coef0) ** degree

def _periodic_kernel(X1, X2, period, length_scale):
    """Compute periodic kernel between two sets of points."""
    # Use pairwise squared distances for the periodic kernel
    sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
    sq_dists = np.maximum(sq_dists, 0)
    return np.exp(-2 * np.pi**2 * np.sin(np.sqrt(sq_dists) / (2 * period))**2 / (2 * length_scale**2))

def _composite_kernel(X1, X2, params):
    """
    Compute composite kernel: sum of RBF, polynomial, and periodic kernels.
    params: dict with keys 'rbf_ls', 'rbf_var', 'poly_deg', 'poly_gamma', 'poly_coef0', 
             'poly_var', 'per_period', 'per_ls', 'per_var'
    """
    K = np.zeros((X1.shape[0], X2.shape[0]))
    
    if params.get('rbf_var', 0) > 1e-12:
        K += params['rbf_var'] * _rbf_kernel(X1, X2, params['rbf_ls'])
    
    if params.get('poly_var', 0) > 1e-12:
        K += params['poly_var'] * _poly_kernel(X1, X2, params['poly_deg'], params['poly_gamma'], params['poly_coef0'])
    
    if params.get('per_var', 0) > 1e-12:
        K += params['per_var'] * _periodic_kernel(X1, X2, params['per_period'], params['per_ls'])
    
    return K

def _kernel_ridge_solve(K_tr, y, alpha):
    """Solve kernel ridge regression. Returns alpha coefficients."""
    n = len(y)
    A = K_tr + alpha * np.eye(n)
    try:
        L = np.linalg.cholesky(A)
        a = np.linalg.solve(L, y)
        alpha_vec = np.linalg.solve(L.T, a)
    except np.linalg.LinAlgError:
        alpha_vec = np.linalg.solve(A + 1e-8 * np.eye(n), y)
    return alpha_vec

def _make_kernel_predictor(X_train, y_train, params, alpha):
    """Create a predictor function using kernel ridge regression."""
    K_tr = _composite_kernel(X_train, X_train, params)
    alpha_vec = _kernel_ridge_solve(K_tr, y_train, alpha)
    X_tr = X_train.copy()
    y_mean = np.mean(y_train)
    
    def predict(X):
        K_pred = _composite_kernel(X, X_tr, params)
        return y_mean + K_pred @ alpha_vec
    
    return predict

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 325.0
    
    d = X_train.shape[1]
    n = len(X_train)
    y_mean = np.mean(y_train)
    y_var = np.var(y_train)
    if y_var < 1e-12:
        return lambda X: np.full(len(X), y_mean)
    
    best_score = np.inf
    best_predictor = None
    
    def update_best(score, predictor):
        nonlocal best_score, best_predictor
        if score < best_score:
            best_score = score
            best_predictor = predictor
    
    # Grid search over kernel hyperparameters
    rbf_ls_vals = [0.3, 0.5, 1.0, 2.0, 4.0, 8.0, 15.0]
    poly_deg_vals = [1, 2, 3, 4]
    poly_gamma_vals = [0.1, 0.5, 1.0, 2.0]
    per_period_vals = [2.0, 4.0, 6.0, 10.0]
    per_ls_vals = [0.5, 1.0, 2.0, 4.0]
    alpha_vals = [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0]
    
    # Pre-compute some kernel components to save time
    # We'll do a staged search: first find good individual kernel params, then combine
    
    # Stage 1: Find best RBF parameters
    best_rbf_params = None
    best_rbf_score = np.inf
    for ls in rbf_ls_vals:
        if time.time() > deadline - 20:
            break
        for alpha in alpha_vals:
            if time.time() > deadline - 5:
                break
            try:
                params = {'rbf_ls': ls, 'rbf_var': 1.0, 'poly_var': 0, 'per_var': 0}
                K_tr = _composite_kernel(X_train, X_train, params)
                alpha_vec = _kernel_ridge_solve(K_tr, y_train, alpha)
                pred = y_mean + K_tr @ alpha_vec
                score = np.mean((pred - y_train)**2) / y_var
                if score < best_rbf_score:
                    best_rbf_score = score
                    best_rbf_params = {'rbf_ls': ls, 'rbf_var': 1.0}
            except:
                pass
    
    # Stage 2: Find best polynomial parameters
    best_poly_params = None
    best_poly_score = np.inf
    for deg in poly_deg_vals:
        if time.time() > deadline - 20:
            break
        for gamma in poly_gamma_vals:
            if time.time() > deadline - 5:
                break
            for alpha in alpha_vals:
                if time.time() > deadline - 5:
                    break
                try:
                    params = {'poly_deg': deg, 'poly_gamma': gamma, 'poly_coef0': 1.0, 'poly_var': 1.0, 'rbf_var': 0, 'per_var': 0}
                    K_tr = _composite_kernel(X_train, X_train, params)
                    alpha_vec = _kernel_ridge_solve(K_tr, y_train, alpha)
                    pred = y_mean + K_tr @ alpha_vec
                    score = np.mean((pred - y_train)**2) / y_var
                    if score < best_poly_score:
                        best_poly_score = score
                        best_poly_params = {'poly_deg': deg, 'poly_gamma': gamma, 'poly_coef0': 1.0, 'poly_var': 1.0}
                except:
                    pass
    
    # Stage 3: Find best periodic parameters
    best_per_params = None
    best_per_score = np.inf
    for period in per_period_vals:
        if time.time() > deadline - 20:
            break
        for ls in per_ls_vals:
            if time.time() > deadline - 5:
                break
            for alpha in alpha_vals:
                if time.time() > deadline - 5:
                    break
                try:
                    params = {'per_period': period, 'per_ls': ls, 'per_var': 1.0, 'rbf_var': 0, 'poly_var': 0}
                    K_tr = _composite_kernel(X_train, X_train, params)
                    alpha_vec = _kernel_ridge_solve(K_tr, y_train, alpha)
                    pred = y_mean + K_tr @ alpha_vec
                    score = np.mean((pred - y_train)**2) / y_var
                    if score < best_per_score:
                        best_per_score = score
                        best_per_params = {'per_period': period, 'per_ls': ls, 'per_var': 1.0}
                except:
                    pass
    
    # Stage 4: Composite kernel search - combine best individual params with variations
    if time.time() < deadline - 30:
        # Try composite kernels with different weight combinations
        composite_configs = []
        
        # RBF only
        if best_rbf_params:
            for ls in [best_rbf_params['rbf_ls'], 0.5, 1.0, 2.0, 4.0]:
                for alpha in [1e-3, 1e-2, 0.1, 1.0]:
                    composite_configs.append({
                        'rbf_ls': ls, 'rbf_var': 1.0,
                        'poly_var': 0, 'per_var': 0,
                        'alpha': alpha
                    })
        
        # Polynomial only
        if best_poly_params:
            for deg in [best_poly_params['poly_deg'], 2, 3, 4]:
                for gamma in [best_poly_params['poly_gamma'], 0.5, 1.0, 2.0]:
                    for alpha in [1e-3, 1e-2, 0.1, 1.0]:
                        composite_configs.append({
                            'poly_deg': deg, 'poly_gamma': gamma, 'poly_coef0': 1.0, 'poly_var': 1.0,
                            'rbf_var': 0, 'per_var': 0,
                            'alpha': alpha
                        })
        
        # Periodic only
        if best_per_params:
            for period in [best_per_params['per_period'], 2.0, 4.0, 6.0]:
                for ls in [best_per_params['per_ls'], 0.5, 1.0, 2.0]:
                    for alpha in [1e-3, 1e-2, 0.1, 1.0]:
                        composite_configs.append({
                            'per_period': period, 'per_ls': ls, 'per_var': 1.0,
                            'rbf_var': 0, 'poly_var': 0,
                            'alpha': alpha
                        })
        
        # RBF + Polynomial
        if best_rbf_params and best_poly_params:
            for rbf_ls in [best_rbf_params['rbf_ls'], 1.0, 2.0]:
                for poly_deg in [best_poly_params['poly_deg'], 2, 3]:
                    for poly_gamma in [best_poly_params['poly_gamma'], 0.5, 1.0]:
                        for rbf_w in [0.5, 1.0, 2.0]:
                            for poly_w in [0.5, 1.0, 2.0]:
                                for alpha in [1e-3, 1e-2, 0.1]:
                                    composite_configs.append({
                                        'rbf_ls': rbf_ls, 'rbf_var': rbf_w,
                                        'poly_deg': poly_deg, 'poly_gamma': poly_gamma, 'poly_coef0': 1.0, 'poly_var': poly_w,
                                        'per_var': 0,
                                        'alpha': alpha
                                    })
        
        # RBF + Periodic
        if best_rbf_params and best_per_params:
            for rbf_ls in [best_rbf_params['rbf_ls'], 1.0, 2.0]:
                for per_period in [best_per_params['per_period'], 4.0, 6.0]:
                    for per_ls in [best_per_params['per_ls'], 1.0, 2.0]:
                        for rbf_w in [0.5, 1.0]:
                            for per_w in [0.5, 1.0]:
                                for alpha in [1e-3, 1e-2, 0.1]:
                                    composite_configs.append({
                                        'rbf_ls': rbf_ls, 'rbf_var': rbf_w,
                                        'per_period': per_period, 'per_ls': per_ls, 'per_var': per_w,
                                        'poly_var': 0,
                                        'alpha': alpha
                                    })
        
        # Polynomial + Periodic
        if best_poly_params and best_per_params:
            for poly_deg in [best_poly_params['poly_deg'], 2, 3]:
                for poly_gamma in [best_poly_params['poly_gamma'], 0.5, 1.0]:
                    for per_period in [best_per_params['per_period'], 4.0, 6.0]:
                        for per_ls in [best_per_params['per_ls'], 1.0, 2.0]:
                            for poly_w in [0.5, 1.0]:
                                for per_w in [0.5, 1.0]:
                                    for alpha in [1e-3, 1e-2, 0.1]:
                                        composite_configs.append({
                                            'poly_deg': poly_deg, 'poly_gamma': poly_gamma, 'poly_coef0': 1.0, 'poly_var': poly_w,
                                            'per_period': per_period, 'per_ls': per_ls, 'per_var': per_w,
                                            'rbf_var': 0,
                                            'alpha': alpha
                                        })
        
        # Full composite: RBF + Poly + Periodic
        if best_rbf_params and best_poly_params and best_per_params:
            for rbf_ls in [best_rbf_params['rbf_ls'], 1.0, 2.0]:
                for poly_deg in [best_poly_params['poly_deg'], 2, 3]:
                    for poly_gamma in [best_poly_params['poly_gamma'], 0.5, 1.0]:
                        for per_period in [best_per_params['per_period'], 4.0, 6.0]:
                            for per_ls in [best_per_params['per_ls'], 1.0, 2.0]:
                                for rbf_w in [0.5, 1.0]:
                                    for poly_w in [0.5, 1.0]:
                                        for per_w in [0.5, 1.0]:
                                            for alpha in [1e-3, 1e-2, 0.1]:
                                                composite_configs.append({
                                                    'rbf_ls': rbf_ls, 'rbf_var': rbf_w,
                                                    'poly_deg': poly_deg, 'poly_gamma': poly_gamma, 'poly_coef0': 1.0, 'poly_var': poly_w,
                                                    'per_period': per_period, 'per_ls': per_ls, 'per_var': per_w,
                                                    'alpha': alpha
                                                })
        
        # Evaluate composite configs
        for cfg in composite_configs:
            if time.time() > deadline - 10:
                break
            try:
                alpha = cfg.pop('alpha')
                params = cfg
                K_tr = _composite_kernel(X_train, X_train, params)
                alpha_vec = _kernel_ridge_solve(K_tr, y_train, alpha)
                pred = y_mean + K_tr @ alpha_vec
                score = np.mean((pred - y_train)**2) / y_var
                if score < best_score:
                    predictor = _make_kernel_predictor(X_train, y_train, params, alpha)
                    update_best(score, predictor)
            except:
                pass
    
    # Final fallback: simple RBF with best params found
    if best_predictor is None:
        if best_rbf_params:
            try:
                params = {'rbf_ls': best_rbf_params['rbf_ls'], 'rbf_var': 1.0, 'poly_var': 0, 'per_var': 0}
                best_predictor = _make_kernel_predictor(X_train, y_train, params, 1e-2)
            except:
                pass
        if best_predictor is None:
            best_predictor = lambda X: np.full(len(X), y_mean)
    
    return best_predictor
