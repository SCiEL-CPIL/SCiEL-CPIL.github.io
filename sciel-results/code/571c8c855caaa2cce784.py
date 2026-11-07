# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")
from scipy.interpolate import UnivariateSpline, PchipInterpolator, make_interp_spline
from scipy.linalg import cho_factor, cho_solve

_TOTAL_DEADLINE = time.time() + 320.0  # Safety margin

def _remaining():
    return _TOTAL_DEADLINE - time.time()

def _safe_predict(make_fn):
    def predict(X):
        try:
            X = np.asarray(X, dtype=np.float64)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            y = make_fn(X)
            y = np.asarray(y, dtype=np.float64)
            y = np.nan_to_num(y, nan=0.0, posinf=1e12, neginf=-1e12)
            return y
        except Exception:
            X = np.asarray(X, dtype=np.float64)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            return np.zeros(X.shape[0])
    return predict

def _fit_poly(X, y, degree):
    try:
        x = X[:, 0]
        coeffs = np.polyfit(x, y, degree)
        def predict(Xp):
            Xp = np.asarray(Xp, dtype=np.float64)
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1)
            xp = Xp[:, 0]
            return np.polyval(coeffs, xp)
        return _safe_predict(predict)
    except Exception:
        return None

def _fit_pchip(X, y):
    try:
        x = X[:, 0]
        order = np.argsort(x)
        xs, ys = x[order], y[order]
        pchip = PchipInterpolator(xs, ys)
        def predict(Xp):
            Xp = np.asarray(Xp, dtype=np.float64)
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1)
            xp = Xp[:, 0]
            yp = pchip(xp)
            return np.nan_to_num(yp, nan=0.0, posinf=1e12, neginf=-1e12)
        return _safe_predict(predict)
    except Exception:
        return None

def _fit_smooth_spline(X, y, s_factor=1.0):
    try:
        x = X[:, 0]
        order = np.argsort(x)
        xs, ys = x[order], y[order]
        n = len(xs)
        var_y = np.var(ys)
        s = s_factor * var_y * n / 100
        try:
            sp = UnivariateSpline(xs, ys, s=s, k=3)
        except:
            sp = UnivariateSpline(xs, ys, s=var_y * n, k=3)
        def predict(Xp):
            Xp = np.asarray(Xp, dtype=np.float64)
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1)
            xp = Xp[:, 0]
            yp = sp(xp)
            return np.nan_to_num(yp, nan=0.0, posinf=1e12, neginf=-1e12)
        return _safe_predict(predict)
    except Exception:
        return None

def _fit_rbf(X, y, num_centers=25, width=None):
    try:
        n, d = X.shape
        x = X[:, 0]
        x_min, x_max = x.min(), x.max()
        centers = np.linspace(x_min, x_max, num_centers)
        
        if width is None:
            best_rmse = np.inf
            best_width = 1.0
            for w in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]:
                F = np.column_stack([np.exp(-0.5 * ((x - c) / w) ** 2) for c in centers])
                try:
                    w_coeff, *_ = np.linalg.lstsq(F, y, rcond=None)
                    pred = F @ w_coeff
                    rmse = np.sqrt(np.mean((pred - y) ** 2))
                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_width = w
                except:
                    continue
            width = best_width
        
        F = np.column_stack([np.exp(-0.5 * ((x - c) / width) ** 2) for c in centers])
        w_coeff, *_ = np.linalg.lstsq(F, y, rcond=None)
        
        def predict(Xp):
            Xp = np.asarray(Xp, dtype=np.float64)
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1)
            xp = Xp[:, 0]
            Fp = np.column_stack([np.exp(-0.5 * ((xp - c) / width) ** 2) for c in centers])
            return Fp @ w_coeff
        return _safe_predict(predict)
    except Exception:
        return None

def _fit_gp(X, y, length_scale=None, noise=1e-8):
    try:
        n, d = X.shape
        mu = X.mean(axis=0)
        sigma = X.std(axis=0) + 1e-8
        Xn = (X - mu) / sigma
        
        if length_scale is None:
            dists = np.sqrt(np.sum((Xn[:, np.newaxis, :] - Xn[np.newaxis, :, :])**2, axis=2))
            median_dist = np.median(dists[dists > 0])
            length_scale = max(median_dist, 0.1)
        
        def kernel(X1, X2, ls):
            diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
            sq_dist = np.sum(diff**2, axis=2) / (ls**2)
            return np.exp(-0.5 * sq_dist)
        
        K = kernel(Xn, Xn, length_scale) + noise * np.eye(n)
        try:
            cf = cho_factor(K)
            alpha = cho_solve(cf, y)
        except:
            K += 1e-4 * np.eye(n)
            cf = cho_factor(K)
            alpha = cho_solve(cf, y)
        
        def predict(Xp):
            Xp = np.asarray(Xp, dtype=np.float64)
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1)
            Xpn = (Xp - mu) / sigma
            Ks = kernel(Xpn, Xn, length_scale)
            preds = Ks @ alpha
            return preds.flatten()
        return _safe_predict(predict)
    except Exception:
        return None

def _fit_mlp(X, y, hidden_sizes=(64, 64), lr=0.01, epochs=1000, seed=42, batch_size=32):
    try:
        d = X.shape[1]
        n = X.shape[0]
        mu = X.mean(axis=0)
        sigma = X.std(axis=0) + 1e-8
        Xn = (X - mu) / sigma
        y_mu = y.mean()
        y_sigma = y.std() + 1e-8
        yn = (y - y_mu) / y_sigma
        
        np.random.seed(seed)
        layers = [d] + list(hidden_sizes) + [1]
        W = [np.random.randn(layers[i], layers[i+1]) * np.sqrt(2.0 / layers[i]) for i in range(len(layers)-1)]
        b = [np.zeros(layers[i+1]) for i in range(len(layers)-1)]
        
        mW = [np.zeros_like(w) for w in W]
        vW = [np.zeros_like(w) for w in W]
        mb = [np.zeros_like(bb) for bb in b]
        vb = [np.zeros_like(bb) for bb in b]
        t = 0
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        
        def forward(Xb):
            a = Xb
            acts = [a]
            for i in range(len(W)):
                z = a @ W[i] + b[i]
                if i < len(W) - 1:
                    a = np.tanh(z)
                else:
                    a = z
                acts.append(a)
            return acts
        
        for epoch in range(epochs):
            if _remaining() < 5.0:
                break
            if batch_size < n:
                idx = np.random.permutation(n)
                for start in range(0, n, batch_size):
                    if _remaining() < 5.0:
                        break
                    batch_idx = idx[start:start+batch_size]
                    Xb = Xn[batch_idx]
                    yb = yn[batch_idx]
                    bs = len(batch_idx)
                    
                    t += 1
                    acts = forward(Xb)
                    pred = acts[-1]
                    dz = (pred - yb) / bs
                    
                    grads_W = [None] * len(W)
                    grads_b = [None] * len(b)
                    for i in range(len(W) - 1, -1, -1):
                        if i == len(W) - 1:
                            dz_i = dz
                        else:
                            dz_i = dz @ W[i].T * (1 - acts[i+1]**2)
                        grads_W[i] = acts[i].T @ dz_i
                        grads_b[i] = dz_i.sum(axis=0)
                    
                    for i in range(len(W)):
                        mW[i] = beta1 * mW[i] + (1 - beta1) * grads_W[i]
                        vW[i] = beta2 * vW[i] + (1 - beta2) * grads_W[i]**2
                        mW_hat = mW[i] / (1 - beta1**t)
                        vW_hat = vW[i] / (1 - beta2**t)
                        W[i] -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
                        
                        mb[i] = beta1 * mb[i] + (1 - beta1) * grads_b[i]
                        vb[i] = beta2 * vb[i] + (1 - beta2) * grads_b[i]**2
                        mb_hat = mb[i] / (1 - beta1**t)
                        vb_hat = vb[i] / (1 - beta2**t)
                        b[i] -= lr * mb_hat / (np.sqrt(vb_hat) + eps)
        
        def predict(Xp):
            Xp = np.asarray(Xp, dtype=np.float64)
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1)
            Xpn = (Xp - mu) / sigma
            a = Xpn
            for i in range(len(W)):
                z = a @ W[i] + b[i]
                a = np.tanh(z) if i < len(W) - 1 else z
            return a.flatten() * y_sigma + y_mu
        return _safe_predict(predict)
    except Exception:
        return None

def fit(X_train, y_train):
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    n, d = X_train.shape

    # Hold-out validation split
    rng = np.random.RandomState(42)
    indices = rng.permutation(n)
    val_frac = min(0.2, max(0.1, 40.0 / n))
    val_size = int(n * val_frac)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    
    X_val, y_val = X_train[val_idx], y_train[val_idx]
    X_tr, y_tr = X_train[train_idx], y_train[train_idx]
    
    var_y_val = np.var(y_val)
    if var_y_val < 1e-12:
        var_y_val = 1.0
    
    best_predict = None
    best_val_nmse = np.inf

    # Baseline: constant
    const = np.mean(y_tr)
    const_pred = np.full(X_val.shape[0], const)
    const_val_nmse = np.mean((const_pred - y_val) ** 2) / var_y_val
    best_val_nmse = const_val_nmse
    best_predict = _safe_predict(lambda Xp: np.full(np.asarray(Xp).shape[0], const))

    methods = []
    
    # Polynomial fits
    for deg in [1, 2, 3, 4, 5, 6, 8, 10]:
        methods.append(("poly_%d" % deg, lambda X, y, d=deg: _fit_poly(X, y, d)))
    
    # PCHIP
    methods.append(("pchip", lambda X, y: _fit_pchip(X, y)))
    
    # Smooth splines
    for sf in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        methods.append(("smooth_spline_%s" % sf, lambda X, y, sf=sf: _fit_smooth_spline(X, y, sf)))
    
    # RBF
    for nc in [10, 25, 50]:
        methods.append(("rbf_%d" % nc, lambda X, y, nc=nc: _fit_rbf(X, y, nc)))
    
    # GP (only if time permits, as it's O(N^3))
    if _remaining() > 50.0:
        for ls in [0.1, 0.5, 1.0, 2.0]:
            methods.append(("gp_ls_%s" % ls, lambda X, y, ls=ls: _fit_gp(X, y, length_scale=ls)))
            
    # MLP (only if time permits)
    if _remaining() > 100.0:
        for hs in [(32, 32), (64, 64)]:
            methods.append(("mlp_%s" % hs, lambda X, y, hs=hs: _fit_mlp(X, y, hs, 0.01, 1000, 42, 32)))

    for name, method in methods:
        if _remaining() < 10.0:
            break
        try:
            predict = method(X_tr, y_tr)
        except Exception:
            predict = None
        if predict is None:
            continue
        
        try:
            val_pred = predict(X_val)
            val_pred = np.asarray(val_pred, dtype=np.float64)
            if not np.all(np.isfinite(val_pred)):
                continue
            val_nmse = np.mean((val_pred - y_val) ** 2) / var_y_val
        except Exception:
            continue
        
        if not np.isfinite(val_nmse):
            continue
        
        if val_nmse < best_val_nmse:
            best_val_nmse = val_nmse
            try:
                # Refit on full data if it was the best
                full_predict = method(X_train, y_train)
                if full_predict is not None:
                    full_pred = full_predict(X_train)
                    full_pred = np.asarray(full_pred, dtype=np.float64)
                    if np.all(np.isfinite(full_pred)):
                        best_predict = full_predict
                else:
                    best_predict = predict
            except Exception:
                best_predict = predict

    if best_predict is None:
        best_predict = _safe_predict(lambda Xp: np.full(np.asarray(Xp).shape[0], const))

    return best_predict
# EVOLVE-BLOCK-END
