# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")
from scipy.interpolate import make_interp_spline, PchipInterpolator, CubicSpline
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize_scalar

_TOTAL_DEADLINE = time.time() + 320.0


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


def _fit_smooth_spline(X, y, s_factor=0.01):
    """Smooth cubic spline with optimized smoothing parameter."""
    try:
        x = X[:, 0]
        order = np.argsort(x)
        xs, ys = x[order], y[order]
        n = len(xs)
        
        # Try different smoothing factors
        best_rmse = np.inf
        best_s = s_factor
        for sf in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]:
            try:
                s = CubicSpline(xs, ys, s=sf * np.var(ys) * n)
                pred = s(xs)
                rmse = np.sqrt(np.mean((pred - ys) ** 2))
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_s = sf
            except Exception:
                continue
        
        s = CubicSpline(xs, ys, s=best_s * np.var(ys) * n)
        
        def predict(Xp):
            Xp = np.asarray(Xp, dtype=np.float64)
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1)
            xp = Xp[:, 0]
            yp = s(xp)
            return np.nan_to_num(yp, nan=0.0, posinf=1e12, neginf=-1e12)
        return predict
    except Exception:
        return None


def _fit_pchip(X, y):
    try:
        x = X[:, 0]
        order = np.argsort(x)
        xs, ys = x[order], y[order]
        s = PchipInterpolator(xs, ys)
        
        def predict(Xp):
            Xp = np.asarray(Xp, dtype=np.float64)
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1)
            xp = Xp[:, 0]
            yp = s(xp)
            return np.nan_to_num(yp, nan=0.0, posinf=1e12, neginf=-1e12)
        return predict
    except Exception:
        return None


def _fit_gp(X, y, length_scale=None, noise=1e-6):
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


def _fit_rbf(X, y, num_centers=30, width=None):
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


def _fit_fourier(X, y, num_terms=15):
    try:
        x = X[:, 0]
        x_min, x_max = x.min(), x.max()
        period = x_max - x_min + 1e-8
        
        cols = [np.ones_like(x)]
        for k in range(1, num_terms + 1):
            cols.append(np.sin(2 * np.pi * k * x / period))
            cols.append(np.cos(2 * np.pi * k * x / period))
        
        F = np.column_stack(cols)
        coeffs, *_ = np.linalg.lstsq(F, y, rcond=None)
        
        def predict(Xp):
            Xp = np.asarray(Xp, dtype=np.float64)
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1)
            xp = Xp[:, 0]
            cols = [np.ones_like(xp)]
            for k in range(1, num_terms + 1):
                cols.append(np.sin(2 * np.pi * k * xp / period))
                cols.append(np.cos(2 * np.pi * k * xp / period))
            Fp = np.column_stack(cols)
            return Fp @ coeffs
        return _safe_predict(predict)
    except Exception:
        return None


def _fit_mlp(X, y, hidden_sizes=(128, 128), lr=0.005, epochs=2000, seed=42, batch_size=32):
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
            if _remaining() < 3.0:
                break
            if batch_size < n:
                idx = np.random.permutation(n)
                for start in range(0, n, batch_size):
                    if _remaining() < 3.0:
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
            else:
                t += 1
                acts = forward(Xn)
                pred = acts[-1]
                dz = (pred - yn) / n
                
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


def _fit_neural_ode(X, y, num_steps=100, hidden=32, lr=0.01, epochs=300, seed=42):
    """
    Neural ODE-inspired approach: learn a vector field f(x, t, theta) and integrate
    from t=0 to t=1 to get predictions. This captures derivative structure.
    """
    try:
        x = X[:, 0]
        n = len(x)
        
        # Normalize x to [0, 1]
        x_min, x_max = x.min(), x.max()
        x_range = x_max - x_min + 1e-8
        xn = (x - x_min) / x_range
        
        # Normalize y
        y_mu, y_sigma = y.mean(), y.std() + 1e-8
        yn = (y - y_mu) / y_sigma
        
        np.random.seed(seed)
        
        # Neural ODE: dx/dt = f(x, t, theta)
        # We model y as a function of x via ODE: dy/dt = f(y, t, x, theta)
        # But simpler: use a residual network that learns the derivative
        # and integrate with Euler's method
        
        # Approach: Learn a function g(x, t) such that y(x) = integral_0^1 g(x, t) dt
        # This is equivalent to a neural network with t as an extra input
        
        # Even simpler: use a deep residual network with t as input
        # y(x) = x + sum of residual blocks
        
        # Let's use a different approach: learn the derivative dy/dx
        # using a neural network, then integrate
        
        # Actually, for 1D regression, a Neural ODE can be:
        # h(t) starts at h(0) = some function of x
        # dh/dt = f(h, t, x, theta)
        # y(x) = h(1)
        
        # Simplified: use a deep network with skip connections
        # that implicitly learns smooth functions
        
        # Let's implement a simple neural ODE with Euler integration
        # State: h (scalar per sample)
        # Input: x (scalar per sample), t (scalar)
        
        # Network: f(h, t, x) -> dh/dt
        # Architecture: [h, t, x] -> hidden -> hidden -> 1
        
        np.random.seed(seed)
        W1 = np.random.randn(3, hidden) * 0.1
        b1 = np.zeros(hidden)
        W2 = np.random.randn(hidden, hidden) * 0.1
        b2 = np.zeros(hidden)
        W3 = np.random.randn(hidden, 1) * 0.1
        b3 = np.zeros(1)
        
        # Adam optimizer state
        mW1 = np.zeros_like(W1); vW1 = np.zeros_like(W1)
        mb1 = np.zeros_like(b1); vb1 = np.zeros_like(b1)
        mW2 = np.zeros_like(W2); vW2 = np.zeros_like(W2)
        mb2 = np.zeros_like(b2); vb2 = np.zeros_like(b2)
        mW3 = np.zeros_like(W3); vW3 = np.zeros_like(W3)
        mb3 = np.zeros_like(b3); vb3 = np.zeros_like(b3)
        t_opt = 0
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        
        dt = 1.0 / num_steps
        t_grid = np.linspace(0, 1, num_steps + 1)
        
        def ode_step(h, t, x_vals):
            """Compute dh/dt = f(h, t, x)"""
            # Input: [h, t, x] for each sample
            inp = np.column_stack([h, np.full_like(x_vals, t), x_vals])
            z1 = inp @ W1 + b1
            a1 = np.tanh(z1)
            z2 = a1 @ W2 + b2
            a2 = np.tanh(z2)
            z3 = a2 @ W3 + b3
            return z3.flatten()
        
        def integrate_ode(x_vals):
            """Integrate ODE from t=0 to t=1 for all x_vals."""
            # Initial condition: h(0) = 0
            h = np.zeros_like(x_vals)
            for i in range(num_steps):
                t_val = t_grid[i]
                dh_dt = ode_step(h, t_val, x_vals)
                h = h + dt * dh_dt
            return h
        
        for epoch in range(epochs):
            if _remaining() < 5.0:
                break
            
            # Forward pass
            h_final = integrate_ode(xn)
            pred = h_final * y_sigma + y_mu
            target = y
            
            # MSE loss
            error = pred - target
            loss = np.mean(error ** 2)
            
            # Backprop through ODE (adjoint method simplified: just backprop through Euler steps)
            # For simplicity, we'll use a truncated backprop through the last few steps
            # or just do full backprop since num_steps is small
            
            # Actually, let's just use a simpler approach: 
            # Instead of true Neural ODE, use a deep residual network
            # that approximates the integral
            
            # Let's restart with a cleaner approach
            break
        
        # Fallback: use a deep residual network
        # y(x) = x_norm + sum_{i=1}^{L} f_i(x_norm)
        # where each f_i is a small neural network
        
        np.random.seed(seed)
        L = 10  # number of residual blocks
        hidden_size = 32
        
        W_list = [np.random.randn(1, hidden_size) * 0.1 for _ in range(L)]
        b_list = [np.zeros(hidden_size) for _ in range(L)]
        W_out_list = [np.random.randn(hidden_size, 1) * 0.1 for _ in range(L)]
        b_out_list = [np.zeros(1) for _ in range(L)]
        
        # Adam state
        mW = [np.zeros_like(w) for w in W_list]
        vW = [np.zeros_like(w) for w in W_list]
        mb = [np.zeros_like(b) for b in b_list]
        vb = [np.zeros_like(b) for b in b_list]
        mWo = [np.zeros_like(w) for w in W_out_list]
        vWo = [np.zeros_like(w) for w in W_out_list]
        mbo = [np.zeros_like(b) for b in b_out_list]
        vbo = [np.zeros_like(b) for b in b_out_list]
        t_opt = 0
        
        def forward_residual(x_vals):
            """Forward pass through residual network."""
            h = x_vals.copy().reshape(-1, 1)
            acts = [h]
            for i in range(L):
                z = h @ W_list[i] + b_list[i]
                a = np.tanh(z)
                acts.append(a)
                z_out = a @ W_out_list[i] + b_out_list[i]
                h = h + z_out
            return h.flatten(), acts
        
        for epoch in range(epochs):
            if _remaining() < 5.0:
                break
            
            h_final, acts = forward_residual(xn)
            pred = h_final * y_sigma + y_mu
            error = (pred - y) / y_sigma
            
            # Backprop
            # dLoss/dh_final = 2 * error / n
            dh = 2 * error.reshape(-1, 1) / n
            
            # Backprop through residual blocks
            dW = [np.zeros_like(w) for w in W_list]
            db = [np.zeros_like(b) for b in b_list]
            dWo = [np.zeros_like(w) for w in W_out_list]
            dbo = [np.zeros_like(b) for b in b_out_list]
            
            dh_next = dh.copy()
            for i in range(L - 1, -1, -1):
                # h_out = h_in + a @ W_out + b_out
                # dh_in = dh_out + dh_out @ W_out.T
                # da = dh_out @ W_out
                # dz = da * (1 - a^2)
                # dW_in = h_in.T @ dz
                # db_in = dz.sum(0)
                
                a = acts[i + 1]
                h_in = acts[i]
                
                dWo[i] = a.T @ dh_next
                dbo[i] = dh_next.sum(axis=0)
                
                da = dh_next @ W_out_list[i].T
                dz = da * (1 - a**2)
                dW[i] = h_in.T @ dz
                db[i] = dz.sum(axis=0)
                
                dh_next = dh_next + dh_next @ W_out_list[i]
            
            # Adam update
            t_opt += 1
            for i in range(L):
                mW[i] = beta1 * mW[i] + (1 - beta1) * dW[i]
                vW[i] = beta2 * vW[i] + (1 - beta2) * dW[i]**2
                W_list[i] -= lr * (mW[i] / (1 - beta1**t_opt)) / (np.sqrt(vW[i] / (1 - beta2**t_opt)) + eps)
                
                mb[i] = beta1 * mb[i] + (1 - beta1) * db[i]
                vb[i] = beta2 * vb[i] + (1 - beta2) * db[i]**2
                b_list[i] -= lr * (mb[i] / (1 - beta1**t_opt)) / (np.sqrt(vb[i] / (1 - beta2**t_opt)) + eps)
                
                mWo[i] = beta1 * mWo[i] + (1 - beta1) * dWo[i]
                vWo[i] = beta2 * vWo[i] + (1 - beta2) * dWo[i]**2
                W_out_list[i] -= lr * (mWo[i] / (1 - beta1**t_opt)) / (np.sqrt(vWo[i] / (1 - beta2**t_opt)) + eps)
                
                mbo[i] = beta1 * mbo[i] + (1 - beta1) * dbo[i]
                vbo[i] = beta2 * vbo[i] + (1 - beta2) * dbo[i]**2
                b_out_list[i] -= lr * (mbo[i] / (1 - beta1**t_opt)) / (np.sqrt(vbo[i] / (1 - beta2**t_opt)) + eps)
        
        def predict(Xp):
            Xp = np.asarray(Xp, dtype=np.float64)
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1)
            xp = Xp[:, 0]
            xpn = (xp - x_min) / x_range
            h_final, _ = forward_residual(xpn)
            return h_final * y_sigma + y_mu
        return _safe_predict(predict)
    except Exception:
        return None


def _try_method(method_fn, X, y):
    try:
        predict = method_fn(X, y)
        if predict is None:
            return None, np.inf
        pred = predict(X)
        pred = np.asarray(pred, dtype=np.float64)
        if not np.all(np.isfinite(pred)):
            return None, np.inf
        rmse = np.sqrt(np.mean((pred - y) ** 2))
        if not np.isfinite(rmse):
            return None, np.inf
        return predict, rmse
    except Exception:
        return None, np.inf


def _holdout_validate(method_fn, X, y, train_frac=0.8, seed=42):
    try:
        n = X.shape[0]
        np.random.seed(seed)
        idx = np.random.permutation(n)
        split = int(n * train_frac)
        X_tr, y_tr = X[idx[:split]], y[idx[:split]]
        X_va, y_va = X[idx[split:]], y[idx[split:]]
        
        predict = method_fn(X_tr, y_tr)
        if predict is None:
            return None, np.inf
        pred_va = predict(X_va)
        pred_va = np.asarray(pred_va, dtype=np.float64)
        if not np.all(np.isfinite(pred_va)):
            return None, np.inf
        rmse = np.sqrt(np.mean((pred_va - y_va) ** 2))
        if not np.isfinite(rmse):
            return None, np.inf
        return predict, rmse
    except Exception:
        return None, np.inf


def fit(X_train, y_train):
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    n, d = X_train.shape

    best_predict = None
    best_rmse = np.inf

    # Baseline: constant
    const = np.mean(y_train)
    const_rmse = np.sqrt(np.mean((y_train - const) ** 2))
    best_rmse = const_rmse
    best_predict = _safe_predict(lambda Xp: np.full(Xp.shape[0], const))

    # Try various methods
    methods = []
    
    # Polynomial fits
    for deg in [1, 2, 3, 4, 5, 6, 8, 10]:
        methods.append(("poly_%d" % deg, lambda X, y, d=deg: _fit_poly(X, y, d)))
    
    # Smooth spline fits
    for sf in [0.001, 0.01, 0.1]:
        methods.append(("smooth_spline_%s" % sf, lambda X, y, sf=sf: _fit_smooth_spline(X, y, sf)))
    
    # PCHIP
    methods.append(("pchip", lambda X, y: _fit_pchip(X, y)))
    
    # Gaussian Process with various length scales
    for ls in [0.1, 0.3, 0.5, 1.0, 2.0, 3.0]:
        methods.append(("gp_ls_%s" % ls, lambda X, y, ls=ls: _fit_gp(X, y, length_scale=ls)))
    
    # RBF with various centers and widths
    for nc in [15, 25, 40, 60]:
        for w in [0.1, 0.3, 0.5, 1.0, 2.0]:
            methods.append(("rbf_%d_w%s" % (nc, w), lambda X, y, nc=nc, w=w: _fit_rbf(X, y, nc, w)))
    
    # Fourier
    for nt in [5, 10, 15, 20]:
        methods.append(("fourier_%d" % nt, lambda X, y, nt=nt: _fit_fourier(X, y, nt)))
    
    # Neural ODE (residual network)
    for lr in [0.005, 0.01, 0.02]:
        for epochs in [200, 400]:
            methods.append(("neural_ode_lr%s_e%d" % (lr, epochs), 
                           lambda X, y, lr=lr, e=epochs: _fit_neural_ode(X, y, 100, 32, lr, e, 42)))
    
    # MLP with various architectures
    for hs in [(64, 64), (128, 128), (256, 128)]:
        for lr in [0.001, 0.005, 0.01]:
            methods.append(("mlp_%s_lr%s" % (hs, lr), lambda X, y, hs=hs, lr=lr: _fit_mlp(X, y, hs, lr, 1500, 42, 32)))
    
    # Sort methods by expected speed (fast first)
    method_order = {
        "poly": 0,
        "smooth_spline": 1,
        "pchip": 1,
        "fourier": 2,
        "rbf": 3,
        "gp": 4,
        "neural_ode": 5,
        "mlp": 6,
    }
    
    def sort_key(item):
        name = item[0]
        for prefix, priority in method_order.items():
            if name.startswith(prefix):
                return priority
        return 10
    
    methods.sort(key=sort_key)
    
    # First pass: use hold-out validation to select best method
    validated_methods = []
    for name, method in methods:
        if _remaining() < 5.0:
            break
        predict, rmse = _holdout_validate(method, X_train, y_train)
        if predict is not None and np.isfinite(rmse):
            validated_methods.append((name, method, rmse))
    
    # Sort by hold-out RMSE
    validated_methods.sort(key=lambda x: x[2])
    
    # Refit top methods on full data
    for name, method, _ in validated_methods[:10]:
        if _remaining() < 3.0:
            break
        predict, rmse = _try_method(method, X_train, y_train)
        if predict is not None and rmse < best_rmse:
            best_rmse = rmse
            best_predict = predict
    
    # If no good method found, fall back to full-data evaluation
    if best_predict is None or best_rmse >= const_rmse:
        for name, method in methods:
            if _remaining() < 3.0:
                break
            predict, rmse = _try_method(method, X_train, y_train)
            if predict is not None and rmse < best_rmse:
                best_rmse = rmse
                best_predict = predict
    
    if best_predict is None:
        best_predict = _safe_predict(lambda Xp: np.full(Xp.shape[0], const))

    return best_predict
# EVOLVE-BLOCK-END
