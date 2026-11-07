"""Program using multi-strategy symbolic regression with PCHIP, rational, GP, and NN."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize

# Global time tracking - we have 300s total, use 280s for safety
_global_start_time = None

def _get_remaining_time():
    """Get remaining time based on global tracking."""
    global _global_start_time
    if _global_start_time is None:
        _global_start_time = time.time()
    elapsed = time.time() - _global_start_time
    return max(0, 280.0 - elapsed)

def _is_timeup(deadline):
    return time.time() > deadline

def _try_pchip(X_train, y_train, deadline):
    """Try PCHIP interpolation - excellent for smooth 1D functions."""
    if _is_timeup(deadline):
        return None, np.inf
    
    n = X_train.shape[0]
    d = X_train.shape[1]
    
    if d == 1:
        x = X_train[:, 0]
        # Sort by x for interpolation
        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        y_sorted = y_train[sort_idx]
        
        # Check for duplicate x values
        unique_x, unique_indices = np.unique(x_sorted, return_index=True)
        if len(unique_x) < 3:
            return None, np.inf
        
        x_unique = x_sorted[unique_indices]
        y_unique = y_sorted[unique_indices]
        
        try:
            pchip = PchipInterpolator(x_unique, y_unique)
            
            def predict(X):
                try:
                    xq = X[:, 0]
                    # Handle extrapolation: clip to range with slight extension
                    x_min, x_max = x_unique[0], x_unique[-1]
                    xq_clipped = np.clip(xq, x_min - 0.1*(x_max-x_min), x_max + 0.1*(x_max-x_min))
                    return pchip(xq_clipped)
                except:
                    return np.full(X.shape[0], np.mean(y_unique))
            
            # Estimate quality on training data
            pred_train = predict(X_train)
            var_y = np.var(y_train)
            if var_y < 1e-12:
                return predict, 0.0
            nrmse = np.mean((pred_train - y_train)**2) / var_y
            return predict, nrmse
        except:
            return None, np.inf
    else:
        return None, np.inf

def _try_polynomial(X_train, y_train, deadline, max_degree=8):
    """Try polynomial regression."""
    if _is_timeup(deadline):
        return None, np.inf
    
    n, d = X_train.shape
    best_predict = None
    best_nrmse = np.inf
    
    if d == 1:
        x = X_train[:, 0]
        for degree in [1, 2, 3, 4, 5, 6, 7, 8]:
            if _is_timeup(deadline):
                break
            try:
                coeffs = np.polyfit(x, y_train, degree)
                
                def predict(X, c=coeffs):
                    try:
                        xq = X[:, 0]
                        return np.polyval(c, xq)
                    except:
                        return np.full(X.shape[0], np.mean(y_train))
                
                pred_train = predict(X_train)
                var_y = np.var(y_train)
                if var_y < 1e-12:
                    nrmse = 0.0
                else:
                    nrmse = np.mean((pred_train - y_train)**2) / var_y
                
                if nrmse < best_nrmse:
                    best_nrmse = nrmse
                    best_predict = predict
            except:
                pass
    else:
        # Multi-dimensional: use basis of monomials
        for total_degree in [1, 2, 3]:
            if _is_timeup(deadline):
                break
            try:
                cols = [np.ones(n)]
                for deg in range(1, total_degree + 1):
                    for idx in itertools.combinations_with_replacement(range(d), deg):
                        feat = np.ones(n)
                        for j in idx:
                            feat = feat * X_train[:, j]
                        cols.append(feat)
                feats = np.column_stack(cols)
                w, *_ = np.linalg.lstsq(feats, y_train, rcond=None)
                
                def predict(X, w=w, td=total_degree):
                    try:
                        nx = X.shape[0]
                        nd = X.shape[1]
                        cols = [np.ones(nx)]
                        for deg in range(1, td + 1):
                            for idx in itertools.combinations_with_replacement(range(nd), deg):
                                feat = np.ones(nx)
                                for j in idx:
                                    feat = feat * X[:, j]
                                cols.append(feat)
                        feats_x = np.column_stack(cols)
                        return feats_x @ w
                    except:
                        return np.full(X.shape[0], np.mean(y_train))
                
                pred_train = predict(X_train)
                var_y = np.var(y_train)
                if var_y < 1e-12:
                    nrmse = 0.0
                else:
                    nrmse = np.mean((pred_train - y_train)**2) / var_y
                
                if nrmse < best_nrmse:
                    best_nrmse = nrmse
                    best_predict = predict
            except:
                pass
    
    if best_predict is not None:
        return best_predict, best_nrmse
    return None, np.inf

def _try_rational(X_train, y_train, deadline):
    """Try rational function approximation y = P(x)/Q(x)."""
    if _is_timeup(deadline):
        return None, np.inf
    
    n, d = X_train.shape
    if d != 1:
        return None, np.inf
    
    x = X_train[:, 0]
    best_predict = None
    best_nrmse = np.inf
    
    for num_deg in [1, 2, 3]:
        for den_deg in [1, 2]:
            if _is_timeup(deadline):
                break
            try:
                # Build feature matrix for rational approximation
                # We'll use a linear approach: minimize ||Q(x)*y - P(x)||
                # where P has num_deg+1 coefficients, Q has den_deg+1 coefficients
                # Fix Q(0)=1 to avoid trivial solution
                
                # Basis for P: [1, x, x^2, ...]
                p_basis = np.column_stack([x**i for i in range(num_deg + 1)])
                # Basis for Q (excluding constant which is fixed to 1): [x, x^2, ...]
                q_basis = np.column_stack([x**i for i in range(1, den_deg + 1)]) if den_deg > 0 else None
                
                if q_basis is not None:
                    # A = [p_basis, -y*q_basis], b = -1
                    # Actually: P(x) + y*Q_nonconst(x) = 1
                    # [p_basis, y*q_basis] @ [p_coeffs; q_coeffs] = 1
                    A = np.column_stack([p_basis, y_train[:, None] * q_basis])
                    b = np.ones(n)
                    
                    try:
                        coeffs, *_ = np.linalg.lstsq(A, b, rcond=None)
                        p_coeffs = coeffs[:num_deg+1]
                        q_coeffs = np.concatenate([[1.0], coeffs[num_deg+1:]])
                    except:
                        continue
                else:
                    # Just polynomial in numerator
                    p_coeffs = np.polyfit(x, y_train, num_deg)
                    q_coeffs = np.array([1.0])
                
                def predict(X, pc=p_coeffs, qc=q_coeffs):
                    try:
                        xq = X[:, 0]
                        num = np.polyval(pc, xq)
                        den = np.polyval(qc, xq)
                        # Avoid division by zero
                        den_safe = np.where(np.abs(den) < 1e-10, 1e-10, den)
                        return num / den_safe
                    except:
                        return np.full(X.shape[0], np.mean(y_train))
                
                pred_train = predict(X_train)
                var_y = np.var(y_train)
                if var_y < 1e-12:
                    nrmse = 0.0
                else:
                    nrmse = np.mean((pred_train - y_train)**2) / var_y
                
                if nrmse < best_nrmse:
                    best_nrmse = nrmse
                    best_predict = predict
            except:
                pass
            if _is_timeup(deadline):
                break
        if _is_timeup(deadline):
            break
    
    if best_predict is not None:
        return best_predict, best_nrmse
    return None, np.inf

def _try_gaussian_process(X_train, y_train, deadline):
    """Try Gaussian Process regression with simple RBF kernel."""
    if _is_timeup(deadline):
        return None, np.inf
    
    n, d = X_train.shape
    if d != 1:
        return None, np.inf
    
    x = X_train[:, 0]
    
    # Standardize
    x_mean = np.mean(x)
    x_std = np.std(x) + 1e-10
    x_norm = (x - x_mean) / x_std
    
    y_mean = np.mean(y_train)
    y_std = np.std(y_train) + 1e-10
    y_norm = (y_train - y_mean) / y_std
    
    best_predict = None
    best_nrmse = np.inf
    
    # Try a few length scales
    for ls in [0.5, 1.0, 2.0, 3.0, 5.0]:
        if _is_timeup(deadline):
            break
        try:
            # RBF kernel
            def rbf(x1, x2, length_scale):
                d2 = (x1[:, None] - x2[None, :])**2
                return np.exp(-d2 / (2 * length_scale**2))
            
            K = rbf(x_norm, x_norm, ls) + 1e-8 * np.eye(n)
            
            try:
                K_inv = np.linalg.inv(K)
            except:
                K_inv = np.linalg.pinv(K)
            
            alpha = K_inv @ y_norm
            
            def predict(X, x_m=x_mean, x_s=x_std, y_m=y_mean, y_s=y_std, 
                       x_tr=x_norm, al=alpha, l=ls):
                try:
                    xq = (X[:, 0] - x_m) / x_s
                    K_star = rbf(xq, x_tr, l)
                    y_pred_norm = K_star @ al
                    return y_pred_norm * y_s + y_m
                except:
                    return np.full(X.shape[0], y_m)
            
            pred_train = predict(X_train)
            var_y = np.var(y_train)
            if var_y < 1e-12:
                nrmse = 0.0
            else:
                nrmse = np.mean((pred_train - y_train)**2) / var_y
            
            if nrmse < best_nrmse:
                best_nrmse = nrmse
                best_predict = predict
        except:
            pass
    
    if best_predict is not None:
        return best_predict, best_nrmse
    return None, np.inf

def _try_fourier(X_train, y_train, deadline, max_terms=15):
    """Try Fourier series approximation."""
    if _is_timeup(deadline):
        return None, np.inf
    
    n, d = X_train.shape
    if d != 1:
        return None, np.inf
    
    x = X_train[:, 0]
    x_min, x_max = x.min(), x.max()
    L = (x_max - x_min) / (2 * np.pi)
    
    best_predict = None
    best_nrmse = np.inf
    
    for n_terms in [3, 5, 8, 10, 15]:
        if _is_timeup(deadline):
            break
        try:
            cols = [np.ones(n)]
            for term in range(1, n_terms + 1):
                cols.append(np.sin(term * x / L))
                cols.append(np.cos(term * x / L))
            feats = np.column_stack(cols)
            w, *_ = np.linalg.lstsq(feats, y_train, rcond=None)
            
            def predict(X, w=w, nt=n_terms, L=L, x_mn=x_min):
                try:
                    xq = X[:, 0]
                    cols = [np.ones(X.shape[0])]
                    for term in range(1, nt + 1):
                        cols.append(np.sin(term * xq / L))
                        cols.append(np.cos(term * xq / L))
                    feats_x = np.column_stack(cols)
                    return feats_x @ w
                except:
                    return np.full(X.shape[0], np.mean(y_train))
            
            pred_train = predict(X_train)
            var_y = np.var(y_train)
            if var_y < 1e-12:
                nrmse = 0.0
            else:
                nrmse = np.mean((pred_train - y_train)**2) / var_y
            
            if nrmse < best_nrmse:
                best_nrmse = nrmse
                best_predict = predict
        except:
            pass
    
    if best_predict is not None:
        return best_predict, best_nrmse
    return None, np.inf

def _try_nn(X_train, y_train, deadline):
    """Try a small neural network with simple gradient descent."""
    if _is_timeup(deadline):
        return None, np.inf
    
    n, d = X_train.shape
    
    # Standardize inputs
    x_mean = np.mean(X_train, axis=0)
    x_std = np.std(X_train, axis=0) + 1e-10
    x_norm = (X_train - x_mean) / x_std
    
    y_mean = np.mean(y_train)
    y_std = np.std(y_train) + 1e-10
    y_norm = (y_train - y_mean) / y_std
    
    # Try a few architectures
    for hidden_sizes in [(10,), (20,), (30, 10)]:
        if _is_timeup(deadline):
            break
        try:
            np.random.seed(42)
            
            # Initialize weights
            layers = [d] + list(hidden_sizes) + [1]
            W = []
            b = []
            for i in range(len(layers) - 1):
                W.append(np.random.randn(layers[i], layers[i+1]) * np.sqrt(2.0 / layers[i]))
                b.append(np.zeros(layers[i+1]))
            
            # Simple gradient descent
            lr = 0.01
            n_epochs = 500
            batch_size = min(50, n)
            
            for epoch in range(n_epochs):
                if _is_timeup(deadline):
                    break
                
                # Shuffle
                idx = np.random.permutation(n)
                for start in range(0, n, batch_size):
                    batch_idx = idx[start:start+batch_size]
                    X_batch = x_norm[batch_idx]
                    y_batch = y_norm[batch_idx]
                    
                    # Forward pass
                    activations = [X_batch]
                    for i in range(len(W)):
                        # Forward pass
                        z = activations[i] @ W[i] + b[i]
                        if i < len(W) - 1:
                            a = np.tanh(z)
                        else:
                            a = z  # Linear output
                        activations.append(a)
                    if i < len(W) - 1:
                        a = np.tanh(z)
                    else:
                        a = z  # Linear output
                    activations.append(a)
                    pred = activations[-1]
                    
                    # Backward pass
                    error = pred - y_batch.reshape(-1, 1)
                    dW = [None] * len(W)
                    db = [None] * len(b)
                    
                    delta = error
                    for i in range(len(W) - 1, -1, -1):
                        if i < len(W) - 1:
                            delta = delta * (1 - activations[i+1]**2)  # tanh'
                        dW[i] = activations[i].T @ delta
                        db[i] = np.sum(delta, axis=0)
                    
                    # Update
                    for i in range(len(W)):
                        W[i] -= lr * dW[i]
                        b[i] -= lr * db[i]
            
            def predict(X, W=W, b=b, x_m=x_mean, x_s=x_std, y_m=y_mean, y_s=y_std):
                try:
                    X_norm = (X - x_m) / x_s
                    a = X_norm
                    for i in range(len(W)):
                        z = a @ W[i] + b[i]
                        if i < len(W) - 1:
                            a = np.tanh(z)
                        else:
                            a = z
                    return a.ravel() * y_s + y_m
                except:
                    return np.full(X.shape[0], y_m)
            
            pred_train = predict(X_train)
            var_y = np.var(y_train)
            if var_y < 1e-12:
                nrmse = 0.0
            else:
                nrmse = np.mean((pred_train - y_train)**2) / var_y
            
            return predict, nrmse
        except:
            continue
    
    return None, np.inf

def _try_combined_basis(X_train, y_train, deadline):
    """Try a combined basis of polynomial + Fourier + RBF."""
    if _is_timeup(deadline):
        return None, np.inf
    
    n, d = X_train.shape
    if d != 1:
        return None, np.inf
    
    x = X_train[:, 0]
    x_min, x_max = x.min(), x.max()
    L = (x_max - x_min) / (2 * np.pi) + 1e-10
    
    best_predict = None
    best_nrmse = np.inf
    
    configs = [
        (3, 5, 10),  # poly_degree, fourier_terms, rbf_centers
        (4, 8, 15),
        (5, 10, 20),
    ]
    
    for poly_deg, fourier_terms, rbf_centers in configs:
        if _is_timeup(deadline):
            break
        try:
            # Polynomial features
            poly_feats = np.column_stack([x**i for i in range(poly_deg + 1)])
            
            # Fourier features
            fourier_feats = []
            for term in range(1, fourier_terms + 1):
                fourier_feats.append(np.sin(term * x / L))
                fourier_feats.append(np.cos(term * x / L))
            fourier_feats = np.column_stack(fourier_feats)
            
            # RBF features
            centers = np.linspace(x_min, x_max, rbf_centers)
            sigma = (x_max - x_min) / rbf_centers + 1e-10
            rbf_feats = np.column_stack([np.exp(-(x - c)**2 / (2 * sigma**2)) for c in centers])
            
            # Combine
            feats = np.column_stack([np.ones(n), poly_feats[:, 1:], fourier_feats, rbf_feats])
            w, *_ = np.linalg.lstsq(feats, y_train, rcond=None)
            
            def predict(X, w=w, pd=poly_deg, ft=fourier_terms, rc=rbf_centers, 
                       xmn=x_min, xmx=x_max, L=L, s=sigma):
                try:
                    xq = X[:, 0]
                    poly = np.column_stack([xq**i for i in range(pd + 1)])
                    ff = []
                    for term in range(1, ft + 1):
                        ff.append(np.sin(term * xq / L))
                        ff.append(np.cos(term * xq / L))
                    ff = np.column_stack(ff)
                    cs = np.linspace(xmn, xmx, rc)
                    rbf = np.column_stack([np.exp(-(xq - c)**2 / (2 * s**2)) for c in cs])
                    feats_x = np.column_stack([np.ones(X.shape[0]), poly[:, 1:], ff, rbf])
                    return feats_x @ w
                except:
                    return np.full(X.shape[0], np.mean(y_train))
            
            pred_train = predict(X_train)
            var_y = np.var(y_train)
            if var_y < 1e-12:
                nrmse = 0.0
            else:
                nrmse = np.mean((pred_train - y_train)**2) / var_y
            
            if nrmse < best_nrmse:
                best_nrmse = nrmse
                best_predict = predict
        except:
            pass
    
    if best_predict is not None:
        return best_predict, best_nrmse
    return None, np.inf

def fit(X_train, y_train):
    """Fit a regression model and return a predict callable."""
    global _global_start_time
    if _global_start_time is None:
        _global_start_time = time.time()
    
    # Budget: use up to 70s per equation (4 equations * 70s = 280s < 300s)
    deadline = time.time() + 70.0
    
    n, d = X_train.shape
    y_mean = np.mean(y_train)
    
    # Initial best: constant predictor
    best_nrmse = np.inf
    best_predict = lambda X: np.full(X.shape[0], y_mean)
    
    # Try strategies in order of expected effectiveness
    strategies = [
        _try_pchip,
        _try_polynomial,
        _try_rational,
        _try_fourier,
        _try_gaussian_process,
        _try_combined_basis,
    ]
    
    for strategy in strategies:
        if _is_timeup(deadline):
            break
        try:
            predict_func, nrmse_est = strategy(X_train, y_train, deadline)
            if predict_func is not None and nrmse_est < best_nrmse:
                best_nrmse = nrmse_est
                best_predict = predict_func
                # If we found an excellent fit, stop early
                if nrmse_est < 1e-8:
                    break
        except Exception as e:
            pass
    
    # If none worked well, try NN as last resort
    if best_nrmse > 0.01 and not _is_timeup(deadline):
        try:
            predict_func, nrmse_est = _try_nn(X_train, y_train, deadline)
            if predict_func is not None and nrmse_est < best_nrmse:
                best_nrmse = nrmse_est
                best_predict = predict_func
        except:
            pass
    
    return best_predict

# EVOLVE-BLOCK-END
