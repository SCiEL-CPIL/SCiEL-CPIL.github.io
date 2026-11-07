# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.interpolate import RBFInterpolator
from itertools import product

# Global time tracker
_total_start = None
_time_limit = 285.0  # Safety margin before 330s kill

def _elapsed():
    global _total_start
    if _total_start is None:
        _total_start = time.time()
    return time.time() - _total_start

def _remaining():
    return _time_limit - _elapsed()

def fit(X_train, y_train):
    global _total_start
    if _total_start is None:
        _total_start = time.time()
    
    n, d = X_train.shape
    X = X_train.astype(np.float64)
    y = y_train.astype(np.float64)
    
    # Store original stats for prediction
    y_mean = y.mean()
    y_std = y.std()
    if y_std < 1e-10:
        y_std = 1.0
    y_norm = (y - y_mean) / y_std
    
    # Normalize inputs to [0, 1] for better kernel behavior
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range < 1e-10] = 1.0
    X_norm = (X - X_min) / X_range
    
    # Strategy 1: Kernel Ridge Regression with RBF kernel and adaptive hyperparameters
    best_krr_score = np.inf
    best_krr_params = None  # (epsilon, alpha)
    best_krr_weights = None
    
    epsilon_values = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    alpha_values = [1e-6, 1e-4, 1e-3, 1e-2, 0.1, 1.0]
    
    for eps in epsilon_values:
        if _remaining() < 1.0:
            break
        # Compute RBF kernel matrix
        sq_norms = np.sum(X_norm ** 2, axis=1)
        sq_dists = sq_norms[:, None] + sq_norms[None, :] - 2 * X_norm @ X_norm.T
        np.maximum(sq_dists, 0, out=sq_dists)
        K = np.exp(-eps * sq_dists)
        
        for alpha in alpha_values:
            if _remaining() < 1.0:
                break
            try:
                w = np.linalg.solve(K + alpha * np.eye(n), y_norm)
                pred = K @ w
                mse = np.mean((pred - y_norm) ** 2)
                if mse < best_krr_score:
                    best_krr_score = mse
                    best_krr_params = (eps, alpha)
                    best_krr_weights = w
            except np.linalg.LinAlgError:
                pass
    
    # Strategy 2: Polynomial kernel ridge regression
    best_poly_score = np.inf
    best_poly_params = None  # (degree, gamma, alpha)
    best_poly_weights = None
    
    degree_values = [2, 3]
    gamma_values = [0.1, 0.5, 1.0, 2.0]
    
    for deg in degree_values:
        if _remaining() < 1.0:
            break
        for gamma in gamma_values:
            if _remaining() < 1.0:
                break
            K_poly = (gamma * X_norm @ X_norm.T + 1) ** deg
            
            for alpha in [1e-6, 1e-4, 1e-3, 1e-2, 0.1]:
                if _remaining() < 1.0:
                    break
                try:
                    w = np.linalg.solve(K_poly + alpha * np.eye(n), y_norm)
                    pred = K_poly @ w
                    mse = np.mean((pred - y_norm) ** 2)
                    if mse < best_poly_score:
                        best_poly_score = mse
                        best_poly_params = (deg, gamma, alpha)
                        best_poly_weights = w
                except np.linalg.LinAlgError:
                    pass
    
    # Strategy 3: Rich feature expansion + Ridge regression
    def build_features(X):
        n = X.shape[0]
        d = X.shape[1]
        cols = [np.ones(n)]
        
        # Linear terms
        for j in range(d):
            cols.append(X[:, j])
        
        # Quadratic and cubic terms
        for deg in range(2, 4):
            monomials = []
            def _gen(idx, remaining, current):
                if idx == d - 1:
                    monomials.append(current + [remaining])
                    return
                for k in range(remaining + 1):
                    _gen(idx + 1, remaining - k, current + [k])
            _gen(0, deg, [])
            for exps in monomials:
                term = np.ones(n)
                for j in range(d):
                    if exps[j] > 0:
                        term = term * np.power(X[:, j], exps[j])
                cols.append(term)
        
        # Trigonometric terms
        for j in range(d):
            cols.append(np.sin(X[:, j]))
            cols.append(np.cos(X[:, j]))
        
        # Sin/cos of pairs
        for i in range(d):
            for j in range(i + 1, d):
                cols.append(np.sin(X[:, i] + X[:, j]))
                cols.append(np.cos(X[:, i] + X[:, j]))
        
        # Exponential and log terms (scaled)
        for j in range(d):
            cols.append(np.exp(X[:, j] / 5.0))
            cols.append(np.log(X[:, j] + 1.0))
        
        # Products
        for i in range(d):
            for j in range(i + 1, d):
                cols.append(X[:, i] * X[:, j])
        
        # Ratios (safe)
        for i in range(d):
            for j in range(i + 1, d):
                denom = X[:, j] + 1e-10
                cols.append(X[:, i] / denom)
        
        return np.column_stack(cols)
    
    F = build_features(X_norm)
    n_features = F.shape[1]
    
    # Normalize features
    F_mean = F.mean(axis=0)
    F_std = F.std(axis=0)
    F_std[F_std < 1e-10] = 1.0
    F_norm = (F - F_mean) / F_std
    
    # Ridge regression with optimal lambda
    best_ridge_score = np.inf
    best_ridge_w = None
    best_ridge_lam = None
    
    lam_values = np.logspace(-4, 4, 9)
    
    for lam in lam_values:
        if _remaining() < 1.0:
            break
        A = F_norm.T @ F_norm + lam * np.eye(n_features)
        b = F_norm.T @ y_norm
        try:
            w = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(A, b, rcond=None)[0]
        pred = F_norm @ w
        mse = np.mean((pred - y_norm) ** 2)
        if mse < best_ridge_score:
            best_ridge_score = mse
            best_ridge_w = w
            best_ridge_lam = lam
    
    # Strategy 4: For 1D, try specific nonlinear forms
    nonlinear_score = np.inf
    nonlinear_type = None
    nonlinear_params = None
    
    if d == 1:
        x = X[:, 0]
        
        # Power law: a * x^p + b
        def obj_power(params):
            if _remaining() < 1.0:
                return np.inf
            a, p, b = params
            try:
                pred = a * np.power(x, p) + b
                return np.mean((pred - y) ** 2)
            except:
                return np.inf
        
        try:
            res = minimize(obj_power, [1.0, 1.0, 0.0], method='Nelder-Mead',
                          options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 200})
            if res.fun < nonlinear_score:
                nonlinear_score = res.fun
                nonlinear_type = 'power'
                nonlinear_params = res.x
        except:
            pass
        
        # Exponential: a * exp(p * x) + b
        def obj_exp(params):
            if _remaining() < 1.0:
                return np.inf
            a, p, b = params
            try:
                pred = a * np.exp(p * x) + b
                return np.mean((pred - y) ** 2)
            except:
                return np.inf
        
        try:
            res = minimize(obj_exp, [1.0, 0.1, 0.0], method='Nelder-Mead',
                          options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 200})
            if res.fun < nonlinear_score:
                nonlinear_score = res.fun
                nonlinear_type = 'exp'
                nonlinear_params = res.x
        except:
            pass
        
        # Sine: a * sin(p * x + b)
        def obj_sin(params):
            if _remaining() < 1.0:
                return np.inf
            a, p, b = params
            try:
                pred = a * np.sin(p * x + b)
                return np.mean((pred - y) ** 2)
            except:
                return np.inf
        
        try:
            res = minimize(obj_sin, [1.0, 1.0, 0.0], method='Nelder-Mead',
                          options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 200})
            if res.fun < nonlinear_score:
                nonlinear_score = res.fun
                nonlinear_type = 'sin'
                nonlinear_params = res.x
        except:
            pass
    
    # Strategy 5: Neural network (MLP) with Adam optimizer
    def _init_weights(fan_in, fan_out, rng):
        scale = np.sqrt(2.0 / (fan_in + fan_out))
        return (rng.standard_normal((fan_in, fan_out)) * scale)
    
    def _tanh(z):
        return np.tanh(z)
    
    def _tanh_grad(a):
        return 1.0 - a * a
    
    def _train_mlp(X, y, hidden_dims, lr=0.01, max_iter=3000, batch_size=32,
                   weight_decay=1e-5, seed=0, time_budget=None):
        rng = np.random.default_rng(seed)
        
        # Initialize weights
        layers = []
        in_dim = X.shape[1]
        for h in hidden_dims:
            W = _init_weights(in_dim, h, rng)
            b = np.zeros(h)
            layers.append((W, b))
            in_dim = h
        W_out = _init_weights(in_dim, 1, rng)
        b_out = np.zeros(1)
        layers.append((W_out, b_out))
        
        # Adam optimizer state
        m_W = [np.zeros_like(W) for W, b in layers]
        v_W = [np.zeros_like(W) for W, b in layers]
        m_b = [np.zeros_like(b) for W, b in layers]
        v_b = [np.zeros_like(b) for W, b in layers]
        beta1, beta2 = 0.9, 0.999
        eps = 1e-8
        t = 0
        
        n_samples = X.shape[0]
        
        for it in range(max_iter):
            if time_budget is not None and (time.time() - time_budget) > 20:
                break
            
            # Mini-batch
            idx = rng.choice(n_samples, size=min(batch_size, n_samples), replace=False)
            X_batch = X[idx]
            y_batch = y[idx].reshape(-1, 1)
            
            # Forward pass
            activations = [X_batch]
            zs = []
            for i, (W, b) in enumerate(layers[:-1]):
                z = activations[-1] @ W + b
                a = _tanh(z)
                zs.append(z)
                activations.append(a)
            z_out = activations[-1] @ layers[-1][0] + layers[-1][1]
            a_out = z_out  # Linear output
            
            # Compute loss and gradients
            loss = np.mean((a_out - y_batch) ** 2)
            dz_out = 2 * (a_out - y_batch) / y_batch.shape[0]
            
            grads_W = [None] * len(layers)
            grads_b = [None] * len(layers)
            
            # Output layer
            grads_W[-1] = activations[-1].T @ dz_out + weight_decay * layers[-1][0]
            grads_b[-1] = np.sum(dz_out, axis=0)
            dz = dz_out @ layers[-1][0].T
            
            # Hidden layers (reverse)
            for i in range(len(layers) - 2, -1, -1):
                da = dz * _tanh_grad(activations[i + 1])
                grads_W[i] = activations[i].T @ da + weight_decay * layers[i][0]
                grads_b[i] = np.sum(da, axis=0)
                if i > 0:
                    dz = da @ layers[i][0].T
            
            # Adam update
            t += 1
            for i in range(len(layers)):
                m_W[i] = beta1 * m_W[i] + (1 - beta1) * grads_W[i]
                v_W[i] = beta2 * v_W[i] + (1 - beta2) * grads_W[i] ** 2
                m_hat = m_W[i] / (1 - beta1 ** t)
                v_hat = v_W[i] / (1 - beta2 ** t)
                layers[i][0] -= lr * m_hat / (np.sqrt(v_hat) + eps)
                
                m_b[i] = beta1 * m_b[i] + (1 - beta1) * grads_b[i]
                v_b[i] = beta2 * v_b[i] + (1 - beta2) * grads_b[i] ** 2
                m_hat_b = m_b[i] / (1 - beta1 ** t)
                v_hat_b = v_b[i] / (1 - beta2 ** t)
                layers[i][1] -= lr * m_hat_b / (np.sqrt(v_hat_b) + eps)
        
        return layers
    
    def _predict_mlp(layers, X):
        a = X
        for i in range(len(layers) - 1):
            W, b = layers[i]
            a = _tanh(a @ W + b)
        W, b = layers[-1]
        return a @ W + b
    
    best_nn_score = np.inf
    best_nn_layers = None
    
    if _remaining() > 10:
        hidden_configs = [[16, 16], [32, 16], [16, 32, 16]]
        for hid in hidden_configs:
            if _remaining() < 5:
                break
            try:
                layers = _train_mlp(X_norm, y_norm, hid, lr=0.01, max_iter=2000,
                                   batch_size=32, time_budget=time.time())
                pred = _predict_mlp(layers, X_norm)
                mse = np.mean((pred - y_norm) ** 2)
                if mse < best_nn_score:
                    best_nn_score = mse
                    best_nn_layers = layers
            except:
                pass
    
    # Select best strategy
    strategies = [
        ('krr', best_krr_score, best_krr_params, best_krr_weights),
        ('poly', best_poly_score, best_poly_params, best_poly_weights),
        ('ridge', best_ridge_score, (best_ridge_lam,), best_ridge_w),
        ('nn', best_nn_score, None, best_nn_layers),
    ]
    
    if d == 1 and nonlinear_score < np.inf:
        strategies.append(('nonlinear', nonlinear_score, nonlinear_params, nonlinear_type))
    
    best_strategy = min(strategies, key=lambda x: x[1] if x[1] < np.inf else 1e10)
    strat_name = best_strategy[0]
    
    # Create predict function based on best strategy
    if strat_name == 'krr':
        eps, alpha = best_strategy[2]
        weights = best_strategy[3]
        
        def predict(X):
            X = np.asarray(X, dtype=np.float64)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            X_pred = (X - X_min) / X_range
            sq_norms = np.sum(X_pred ** 2, axis=1)
            sq_dists = sq_norms[:, None] + sq_norms[None, :] - 2 * X_pred @ X_norm.T
            np.maximum(sq_dists, 0, out=sq_dists)
            K = np.exp(-eps * sq_dists)
            y_pred = (K @ weights) * y_std + y_mean
            return y_pred
    
    elif strat_name == 'poly':
        deg, gamma, alpha = best_strategy[2]
        weights = best_strategy[3]
        
        def predict(X):
            X = np.asarray(X, dtype=np.float64)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            X_pred = (X - X_min) / X_range
            K_poly = (gamma * X_pred @ X_norm.T + 1) ** deg
            y_pred = (K_poly @ weights) * y_std + y_mean
            return y_pred
    
    elif strat_name == 'ridge':
        w = best_strategy[3]
        
        def predict(X):
            X = np.asarray(X, dtype=np.float64)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            X_pred = (X - X_min) / X_range
            F_pred = build_features(X_pred)
            F_pred_norm = (F_pred - F_mean) / F_std
            y_pred = (F_pred_norm @ w) * y_std + y_mean
            return y_pred
    
    elif strat_name == 'nn':
        layers = best_strategy[3]
        
        def predict(X):
            X = np.asarray(X, dtype=np.float64)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            X_pred = (X - X_min) / X_range
            y_pred = _predict_mlp(layers, X_pred) * y_std + y_mean
            return y_pred
    
    elif strat_name == 'nonlinear':
        params = best_strategy[2]
        ntype = best_strategy[3]
        
        def predict(X):
            X = np.asarray(X, dtype=np.float64)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            x = X[:, 0]
            if ntype == 'power':
                a, p, b = params
                y_pred = a * np.power(x, p) + b
            elif ntype == 'exp':
                a, p, b = params
                y_pred = a * np.exp(p * x) + b
            elif ntype == 'sin':
                a, p, b = params
                y_pred = a * np.sin(p * x + b)
            else:
                y_pred = np.mean(y) * np.ones(X.shape[0])
            return y_pred
    
    else:
        # Fallback: constant prediction
        def predict(X):
            X = np.asarray(X, dtype=np.float64)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            return np.full(X.shape[0], y_mean)
    
    return predict
# EVOLVE-BLOCK-END
