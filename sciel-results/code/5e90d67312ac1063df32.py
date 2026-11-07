# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")
from scipy.optimize import minimize
from scipy.interpolate import make_interp_spline

_TOTAL_DEADLINE = time.time() + 315.0


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


def _nrmse(y_pred, y_true):
    v = float(np.var(y_true))
    if v < 1e-12:
        v = 1.0
    d = y_pred - y_true
    return float(np.mean(d * d)) / v


def _fit_ols(F, y, ridge_alpha=1e-8):
    try:
        n, p = F.shape
        F_reg = F.copy()
        F_reg[:, 0] += ridge_alpha * np.sqrt(n)
        w, *_ = np.linalg.lstsq(F_reg, y, rcond=None)
        if not np.all(np.isfinite(w)):
            return None
        return w
    except Exception:
        return None


def _poly_features(X, degree):
    d = X.shape[1]
    cols = [np.ones(X.shape[0])]
    for j in range(d):
        for p in range(1, degree + 1):
            cols.append(X[:, j] ** p)
    if d >= 2:
        for i in range(d):
            for j in range(i, d):
                for p in range(1, degree + 1):
                    for q in range(1, degree + 1 - p + 1):
                        if p + q <= degree:
                            cols.append((X[:, i] ** p) * (X[:, j] ** q))
    return np.column_stack(cols)


def _trig_features(X, max_freq=8):
    d = X.shape[1]
    cols = [np.ones(X.shape[0])]
    for j in range(d):
        x = X[:, j]
        for k in range(1, max_freq + 1):
            cols.append(np.sin(k * x))
            cols.append(np.cos(k * x))
        cols.append(np.tanh(x))
        cols.append(np.arctan(x))
    return np.column_stack(cols)


def _exp_features(X):
    d = X.shape[1]
    cols = [np.ones(X.shape[0])]
    for j in range(d):
        x = X[:, j]
        cols.append(np.exp(x))
        cols.append(np.exp(-x))
        cols.append(np.sqrt(np.abs(x) + 1e-9))
        cols.append(np.log(np.abs(x) + 1e-9))
        cols.append(1.0 / (np.abs(x) + 1e-6))
        cols.append(x * np.exp(-x))
        cols.append(x * np.exp(x))
        cols.append(np.exp(-x ** 2))
        cols.append(np.exp(x ** 2 / 2))
    return np.column_stack(cols)


def _mixed_features(X):
    d = X.shape[1]
    cols = [np.ones(X.shape[0])]
    for j in range(d):
        x = X[:, j]
        for deg in range(1, 7):
            cols.append(x ** deg)
        for f in range(1, 5):
            cols.append(np.sin(f * x))
            cols.append(np.cos(f * x))
        cols.append(np.tanh(x))
        cols.append(np.arctan(x))
        cols.append(np.exp(-x ** 2))
        cols.append(np.exp(x))
        cols.append(np.exp(-x))
        cols.append(np.log(np.abs(x) + 1e-8))
        cols.append(1.0 / (np.abs(x) + 1e-4))
        cols.append(np.sqrt(np.abs(x) + 1e-8))
    return np.column_stack(cols)


def _gaussian_rbf_features(X, centers, widths):
    d = X.shape[1]
    cols = [np.ones(X.shape[0])]
    for c in centers:
        for w in widths:
            dist = np.sqrt(np.sum((X - c) ** 2, axis=1))
            cols.append(np.exp(-0.5 * (dist / w) ** 2))
    return np.column_stack(cols)


def _fit_spline(X, y, k=5):
    X = np.asarray(X, dtype=np.float64)
    if X.shape[1] == 1:
        x = X[:, 0]
        order = np.argsort(x)
        xs, ys = x[order], y[order]
        keep = np.concatenate(([True], np.diff(xs) > 1e-12))
        xs, ys = xs[keep], ys[keep]
        if len(xs) < k + 1:
            k = max(1, len(xs) - 1)
        try:
            sp = make_interp_spline(xs, ys, k=k)
            def predict(Xq):
                Xq = np.asarray(Xq, dtype=np.float64)
                if Xq.ndim == 1:
                    Xq = Xq.reshape(1, -1)
                return sp(Xq[:, 0])
            return predict
        except Exception:
            return None
    return None


def _fit_mlp(X, y, hidden_sizes=(32, 32), lr=0.005, epochs=400, seed=42):
    try:
        d = X.shape[1]
        n = X.shape[0]
        mu = X.mean(axis=0)
        sigma = X.std(axis=0) + 1e-8
        Xn = (X - mu) / sigma
        y_mu = y.mean()
        y_sigma = y.std() + 1e-8
        yn = (y - y_mu) / y_sigma

        rng = np.random.default_rng(seed)
        layers = [d] + list(hidden_sizes) + [1]
        W = [rng.randn(layers[i], layers[i+1]) * np.sqrt(2.0 / layers[i]) for i in range(len(layers)-1)]
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
        return predict
    except Exception:
        return None


def _gaussian_process_fit(X, y):
    try:
        n, d = X.shape
        x_mean = X.mean(axis=0)
        x_std = X.std(axis=0) + 1e-8
        Xn = (X - x_mean) / x_std
        y_mean = y.mean()
        y_std = y.std() + 1e-8
        yn = (y - y_mean) / y_std

        def rbf(A, B, ls):
            sq = np.sum(A ** 2, axis=1)[:, None] + np.sum(B ** 2, axis=1)[None, :] - 2 * A @ B.T
            sq = np.maximum(sq, 0)
            return np.exp(-sq / (2 * ls ** 2))

        def log_marginal(params):
            ls, sigma = params
            K = rbf(Xn, Xn, ls) + sigma ** 2 * np.eye(n)
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, yn))
                lml = -0.5 * yn @ alpha - np.sum(np.log(np.diag(L)))
                return -float(lml)
            except Exception:
                return 1e6

        best_ls, best_sigma = 1.0, 0.1
        best_lml = log_marginal([1.0, 0.1])
        try:
            res = minimize(log_marginal, [1.0, 0.1], method='Nelder-Mead',
                           options={'maxiter': 150, 'xatol': 1e-3, 'fatol': 1e-3})
            if res.fun < best_lml:
                best_lml = float(res.fun)
                best_ls, best_sigma = float(res.x[0]), float(res.x[1])
        except Exception:
            pass

        K = rbf(Xn, Xn, best_ls) + best_sigma ** 2 * np.eye(n)
        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, yn))
        except Exception:
            return None

        def predict(Xq):
            Xq = np.asarray(Xq, dtype=np.float64)
            if Xq.ndim == 1:
                Xq = Xq.reshape(1, -1)
            Xqn = (Xq - x_mean) / x_std
            Ks = rbf(Xqn, Xn, best_ls)
            out = (Ks @ alpha) * y_std + y_mean
            return out
        return predict
    except Exception:
        return None


def fit(X_train, y_train):
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    n, d = X_train.shape

    call_deadline = time.time() + 70.0

    candidates = []

    def try_linear_model(feat_fn, name=""):
        if time.time() > call_deadline:
            return
        try:
            F = feat_fn(X_train)
            if not np.all(np.isfinite(F)):
                F = np.nan_to_num(F, nan=0.0, posinf=1e6, neginf=-1e6)
            w = _fit_ols(F, y_train)
            if w is None:
                return
            y_pred = F @ w
            if not np.all(np.isfinite(y_pred)):
                return
            score = _nrmse(y_pred, y_train)
            make_fn = lambda Xq, F_fn=feat_fn, w=w: F_fn(Xq) @ w
            candidates.append((score, _safe_predict(make_fn)))
        except Exception:
            pass

    # Polynomial degrees
    for deg in [2, 3, 4, 5, 6, 8, 10]:
        if time.time() > call_deadline:
            break
        try_linear_model(lambda X, d=deg: _poly_features(X, d), f"poly{deg}")

    # Trig
    for freq in [4, 6, 8]:
        if time.time() > call_deadline:
            break
        try_linear_model(lambda X, f=freq: _trig_features(X, f), f"trig{freq}")

    # Exponential
    if time.time() < call_deadline:
        try_linear_model(_exp_features, "exp")

    # Mixed
    if time.time() < call_deadline:
        try_linear_model(_mixed_features, "mixed")

    # RBF with various centers/widths
    if d == 1 and time.time() < call_deadline - 5:
        x = X_train[:, 0]
        x_min, x_max = x.min(), x.max()
        centers = np.linspace(x_min, x_max, 8)
        widths = [0.3, 0.5, 1.0, 1.5]
        try_linear_model(lambda X: _gaussian_rbf_features(X, centers, widths), "rbf")

    # Spline fits
    if d == 1:
        for k in [3, 5, 7]:
            if time.time() > call_deadline - 3:
                break
            try:
                sp_pred = _fit_spline(X_train, y_train, k=k)
                if sp_pred is not None:
                    y_pred = sp_pred(X_train)
                    if np.all(np.isfinite(y_pred)):
                        score = _nrmse(y_pred, y_train)
                        candidates.append((score, _safe_predict(sp_pred)))
            except Exception:
                pass

    # MLP
    if time.time() < call_deadline - 10:
        try:
            mlp_pred = _fit_mlp(X_train, y_train, hidden_sizes=(32, 32), epochs=400)
            if mlp_pred is not None:
                y_pred = mlp_pred(X_train)
                if np.all(np.isfinite(y_pred)):
                    score = _nrmse(y_pred, y_train)
                    candidates.append((score, _safe_predict(mlp_pred)))
        except Exception:
            pass

    # Gaussian Process
    if time.time() < call_deadline - 5:
        try:
            gp_pred = _gaussian_process_fit(X_train, y_train)
            if gp_pred is not None:
                y_pred = gp_pred(X_train)
                if np.all(np.isfinite(y_pred)):
                    score = _nrmse(y_pred, y_train)
                    candidates.append((score, _safe_predict(gp_pred)))
        except Exception:
            pass

    if not candidates:
        c = float(np.mean(y_train))
        return lambda X: np.full(np.asarray(X).shape[0], c)

    candidates.sort(key=lambda t: t[0])
    best_score, best_predict = candidates[0]
    return best_predict
# EVOLVE-BLOCK-END
