# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")
from scipy.interpolate import UnivariateSpline, PchipInterpolator, make_interp_spline
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve

_TOTAL_DEADLINE = time.time() + 325.0


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


def _fit_spline_interp(X, y, num_knots=None):
    try:
        x = X[:, 0]
        order = np.argsort(x)
        xs, ys = x[order], y[order]
        n = len(xs)
        if num_knots is None:
            num_knots = min(n, max(10, n // 2))
        if n < num_knots + 4:
            num_knots = max(4, n - 4)
        knot_idx = np.linspace(0, n - 1, num_knots, dtype=int)
        knot_x = xs[knot_idx]
        knot_y = ys[knot_idx]
        s = make_interp_spline(knot_x, knot_y, k=3)
        
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
        pchip = PchipInterpolator(xs, ys)
        
        def predict(Xp):
            Xp = np.asarray(Xp, dtype=np.float64)
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1)
            xp = Xp[:, 0]
            yp = pchip(xp)
            return np.nan_to_num(yp, nan=0.0, posinf=1e12, neginf=-1e12)
        return predict
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
        return predict
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


def _fit_mlp(X, y, hidden_sizes=(64, 64), lr=0.01, epochs=3000, seed=42, batch_size=32):
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


def _fit_fourier(X, y, num_terms=10):
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


def _fit_neural_ode(X, y, hidden_size=16, steps=200, lr=0.01, seed=42):
    """Neural ODE regression: learn dx/dt = f(x, t, theta) to model y as trajectory."""
    try:
        x = X[:, 0]
        n = len(x)
        
        # Normalize input to [0, 1]
        x_min, x_max = x.min(), x.max()
        x_range = x_max - x_min + 1e-8
        xn = (x - x_min) / x_range
        
        # Normalize output
        y_mu = y.mean()
        y_sigma = y.std() + 1e-8
        yn = (y - y_mu) / y_sigma
        
        # Sort by input
        order = np.argsort(xn)
        xn_sorted = xn[order]
        yn_sorted = yn[order]
        
        # Initial condition: y at t=0
        y0 = yn_sorted[0]
        
        # Neural ODE parameters
        np.random.seed(seed)
        # f(x, t) = W1 @ [x, t, 1] + b1, then tanh, then W2 @ h + b2
        W1 = np.random.randn(2, hidden_size) * 0.5
        b1 = np.zeros(hidden_size)
        W2 = np.random.randn(hidden_size, 1) * 0.5
        b2 = np.zeros(1)
        
        params = np.concatenate([W1.flatten(), b1, W2.flatten(), b2])
        
        # Euler integration
        dt = 1.0 / steps
        t_values = np.linspace(0, 1, steps + 1)
        
        def ode_step(state, t, params):
            W1 = params[:2*hidden_size].reshape(2, hidden_size)
            b1 = params[2*hidden_size:3*hidden_size]
            W2 = params[3*hidden_size:3*hidden_size+hidden_size].reshape(hidden_size, 1)
            b2 = params[3*hidden_size+hidden_size:]
            
            x_val, t_val = state
            inp = np.array([x_val, t_val, 1.0])
            h = np.tanh(W1 @ inp + b1)
            return (W2 @ h + b2).flatten()[0]
        
        def integrate(params, t_target):
            """Integrate from t=0 to t_target, return y(t_target)."""
            y_val = y0
            for i in range(steps):
                t_curr = i * dt
                if t_curr >= t_target - 1e-9:
                    break
                f_val = ode_step((y_val, t_curr), t_curr, params)
                y_val += dt * f_val
            return y_val
        
        def predict_yn(params, t_values):
            """Predict yn for all t_values via integration."""
            preds = np.zeros(len(t_values))
            y_val = y0
            for i in range(len(t_values)):
                preds[i] = y_val
                if i < len(t_values) - 1:
                    t_curr = i * dt
                    f_val = ode_step((y_val, t_curr), t_curr, params)
                    y_val += dt * f_val
            return preds
        
        # Train with Adam
        m = np.zeros_like(params)
        v = np.zeros_like(params)
        t_step = 0
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        
        # Training: match yn at sorted input points
        # We need to find which t_interval each point falls in
        # For simplicity, use all t_values and interpolate
        train_epochs = 500
        for epoch in range(train_epochs):
            if _remaining() < 10.0:
                break
            
            # Forward pass
            pred_yn = predict_yn(params, t_values)
            
            # Interpolate to match training points
            # t_values are uniform in [0,1], xn_sorted are the targets
            # Use linear interpolation
            from scipy.interpolate import interp1d
            try:
                interp_fn = interp1d(t_values, pred_yn, kind='linear', fill_value='extrapolate')
                pred_at_points = interp_fn(xn_sorted)
            except:
                pred_at_points = pred_yn
            
            # Loss
            loss = np.mean((pred_at_points - yn_sorted) ** 2)
            
            # Backprop through Euler integration (approximate)
            # This is expensive; use finite differences for gradient
            if epoch % 5 == 0:  # Compute gradient every 5 epochs
                grad = np.zeros_like(params)
                for j in range(len(params)):
                    params_p = params.copy()
                    params_p[j] += 1e-5
                    pred_p = predict_yn(params_p, t_values)
                    try:
                        interp_fn_p = interp1d(t_values, pred_p, kind='linear', fill_value='extrapolate')
                        pred_p_pts = interp_fn_p(xn_sorted)
                    except:
                        pred_p_pts = pred_p
                    loss_p = np.mean((pred_p_pts - yn_sorted) ** 2)
                    grad[j] = (loss_p - loss) / 1e-5
                
                # Adam update
                t_step += 1
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad ** 2
                m_hat = m / (1 - beta1 ** t_step)
                v_hat = v / (1 - beta2 ** t_step)
                params -= lr * m_hat / (np.sqrt(v_hat) + eps)
            
            # Check convergence
            if loss < 1e-6:
                break
        
        # Final predict
        def predict(Xp):
            Xp = np.asarray(Xp, dtype=np.float64)
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1)
            xp = Xp[:, 0]
            xpn = (xp - x_min) / x_range
            
            # Integrate for each point
            preds = np.zeros(len(xpn))
            for i, xpi in enumerate(xpn):
                # Find where xpi falls in t_values
                # t_values are uniform [0, 1], so t = xpi
                t_target = np.clip(xpi, 0, 1)
                
                y_val = y0
                for step_i in range(steps):
                    t_curr = step_i * dt
                    if t_curr >= t_target - 1e-9:
                        break
                    f_val = ode_step((y_val, t_curr), t_curr, params)
                    y_val += dt * f_val
                preds[i] = y_val * y_sigma + y_mu
            
            return np.nan_to_num(preds, nan=0.0, posinf=1e12, neginf=-1e12)
        
        return _safe_predict(predict)
    except Exception:
        return None


def _fit_mixture_of_splines(X, y, num_components=3):
    """Fit a mixture of smooth splines with different smoothing parameters."""
    try:
        x = X[:, 0]
        order = np.argsort(x)
        xs, ys = x[order], y[order]
        n = len(xs)
        
        # Try different smoothing factors
        s_factors = [0.01, 0.1, 1.0, 10.0]
        splines = []
        for sf in s_factors:
            var_y = np.var(ys)
            s = sf * var_y * n / 100
            try:
                sp = UnivariateSpline(xs, ys, s=s, k=3)
                splines.append(sp)
            except:
                splines.append(None)
        
        # Weighted average
        def predict(Xp):
            Xp = np.asarray(Xp, dtype=np.float64)
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1)
            xp = Xp[:, 0]
            
            preds = []
            weights = []
            for sp in splines:
                if sp is not None:
                    yp = sp(xp)
                    if np.all(np.isfinite(yp)):
                        preds.append(yp)
                        weights.append(1.0)
            
            if not preds:
                return np.zeros(len(xp))
            
            # Simple average
            avg_pred = np.mean(preds, axis=0)
            return np.nan_to_num(avg_pred, nan=0.0, posinf=1e12, neginf=-1e12)
        
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
    for deg in [1, 2, 3, 4, 5, 6, 8]:
        methods.append(("poly_%d" % deg, lambda X, y, d=deg: _fit_poly(X, y, d)))
    
    # PCHIP (preserves shape)
    methods.append(("pchip", lambda X, y: _fit_pchip(X, y)))
    
    # Smooth splines
    for sf in [0.001, 0.01, 0.1, 1.0, 10.0]:
        methods.append(("smooth_spline_%s" % sf, lambda X, y, sf=sf: _fit_smooth_spline(X, y, sf)))
    
    # Interpolation splines
    for nk in [10, 15, 25, 40, 60, 80]:
        methods.append(("spline_interp_%d" % nk, lambda X, y, nk=nk: _fit_spline_interp(X, y, nk)))
    
    # Fourier
    for nt in [3, 5, 8, 12, 15]:
        methods.append(("fourier_%d" % nt, lambda X, y, nt=nt: _fit_fourier(X, y, nt)))
    
    # RBF
    for nc in [10, 15, 25, 40, 60]:
        for w in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]:
            methods.append(("rbf_%d_w%s" % (nc, w), lambda X, y, nc=nc, w=w: _fit_rbf(X, y, nc, w)))
    
    # GP
    for ls in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]:
        methods.append(("gp_ls_%s" % ls, lambda X, y, ls=ls: _fit_gp(X, y, length_scale=ls)))
    
    # Neural ODE (new structural approach)
    for hs in [8, 16, 32]:
        for steps in [100, 200, 400]:
            methods.append(("neural_ode_hs%d_s%d" % (hs, steps), 
                           lambda X, y, hs=hs, steps=steps: _fit_neural_ode(X, y, hs, steps, 0.01, 42)))
    
    # Mixture of splines
    methods.append(("mixture_splines", lambda X, y: _fit_mixture_of_splines(X, y, 3)))
    
    # MLP
    for hs in [(32, 32), (64, 64), (128, 64)]:
        for lr in [0.005, 0.01, 0.02]:
            methods.append(("mlp_%s_lr%s" % (hs, lr), lambda X, y, hs=hs, lr=lr: _fit_mlp(X, y, hs, lr, 2000, 42, 32)))
    
    # Sort by expected speed
    method_order = {
        "poly": 0,
        "pchip": 1,
        "smooth_spline": 2,
        "spline_interp": 3,
        "fourier": 4,
        "rbf_": 5,
        "gp": 6,
        "neural_ode": 7,
        "mixture": 8,
        "mlp": 9,
    }
    
    def sort_key(item):
        name = item[0]
        for prefix, priority in method_order.items():
            if name.startswith(prefix):
                return priority
        return 10
    
    methods.sort(key=sort_key)
    
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
