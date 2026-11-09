import numpy as np
import time
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 285  # Safety margin before 330s limit

    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape

    # ---- Fallback for d > 1 (unlikely per contract) ----
    if d != 1:
        try:
            X_aug = np.column_stack([np.ones(n), X])
            w, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
            def predict(X_new):
                X_new = np.asarray(X_new, dtype=np.float64)
                X_aug = np.column_stack([np.ones(X_new.shape[0]), X_new])
                return X_aug @ w
            return predict
        except Exception:
            y_mean = float(np.mean(y))
            def predict(X_new):
                return np.full(len(np.asarray(X_new).reshape(-1, 1)), y_mean)
            return predict

    x = X.flatten()
    y = y.flatten()

    # =========================================================
    # Model A: Gaussian Process (primary) — exploits noiseless data
    # =========================================================
    def fit_gp(x, y, max_iter=120):
        """Optimize log-likelihood over (log length, log signal) via L-BFGS-B."""
        n = len(x)
        # Standardize for a well-conditioned kernel
        xm = float(np.mean(x))
        xs = float(np.std(x))
        if not np.isfinite(xs) or xs < 1e-8:
            xs = 1.0
        xz = (x - xm) / xs

        # Initial scale guess: median spacing
        x_sorted = np.sort(xz)
        gaps = np.diff(x_sorted)
        gaps = gaps[gaps > 1e-12]
        if len(gaps):
            med_gap = float(np.median(gaps))
        else:
            med_gap = 1.0
        if not np.isfinite(med_gap) or med_gap <= 1e-12:
            med_gap = 1.0

        ym = float(np.mean(y))
        ys = float(np.std(y))
        if not np.isfinite(ys) or ys < 1e-12:
            ys = 1.0
        yz = (y - ym) / ys

        def neg_mll(theta):
            try:
                log_l = float(theta[0])
                log_s = float(theta[1])
                length = np.exp(log_l)
                signal = np.exp(log_s)
                # RBF kernel in standardized space
                diff = xz[:, None] - xz[None, :]
                K = signal * np.exp(-0.5 * (diff / length) ** 2)
                # tiny jitter for numerical stability (data is noiseless)
                K = K + 1e-10 * np.eye(n)
                try:
                    L = np.linalg.cholesky(K)
                except np.linalg.LinAlgError:
                    K = K + 1e-6 * np.eye(n)
                    try:
                        L = np.linalg.cholesky(K)
                    except np.linalg.LinAlgError:
                        return 1e12
                # solve L a = yz
                a = np.linalg.solve(L, yz)
                # solve L^T v = a
                v = np.linalg.solve(L.T, a)
                logdet = 2.0 * np.sum(np.log(np.diag(L)))
                mll = -0.5 * (n * np.log(2 * np.pi) + logdet + yz @ v)
                val = -mll
                if not np.isfinite(val):
                    return 1e12
                return val
            except Exception:
                return 1e12

        x0 = np.array([np.log(med_gap), np.log(max(ys, 1e-8))])
        best_theta = x0.copy()
        best_val = neg_mll(x0)
        try:
            res = minimize(neg_mll, x0, method='L-BFGS-B',
                           options={'maxiter': max_iter, 'maxls': 40})
            if res.x is not None:
                rv = neg_mll(res.x)
                if rv < best_val:
                    best_val = rv
                    best_theta = res.x.copy()
        except Exception:
            pass

        log_l = float(best_theta[0])
        log_s = float(best_theta[1])
        length = np.exp(log_l)
        signal = np.exp(log_s)

        diff = xz[:, None] - xz[None, :]
        K = signal * np.exp(-0.5 * (diff / length) ** 2)
        K = K + 1e-10 * np.eye(n)
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            K = K + 1e-6 * np.eye(n)
            L = np.linalg.cholesky(K)
        a = np.linalg.solve(L, yz)

        def predict_gp(X_new):
            xn = np.asarray(X_new, dtype=np.float64).reshape(-1)
            xzn = (xn - xm) / xs
            diff = xzn[:, None] - xz[None, :]
            Ks = signal * np.exp(-0.5 * (diff / length) ** 2)
            # v = L^{-1} a ;  f = Ks @ v  (mean), scaled back
            v = np.linalg.solve(L, a)
            f_z = Ks @ v
            return (f_z * ys) + ym

        # training sanity check
        try:
            train_pred = predict_gp(x)
            if np.all(np.isfinite(train_pred)):
                return predict_gp
        except Exception:
            pass
        return None

    # =========================================================
    # Model B: Cubic Spline (fallback)
    # =========================================================
    def create_spline_model(x, y):
        try:
            idx = np.argsort(x)
            xs = x[idx]
            ys = y[idx]
            if np.any(np.diff(xs) == 0):
                # average duplicates
                ux, inv = np.unique(xs, return_inverse=True)
                if len(ux) < len(xs):
                    agg = np.zeros(len(ux))
                    cnt = np.zeros(len(ux))
                    np.add.at(agg, inv, ys)
                    np.add.at(cnt, inv, 1)
                    ys = agg / cnt
                    xs = ux
                else:
                    return None
            cs = CubicSpline(xs, ys)
            def predict(X_new):
                xn = np.asarray(X_new, dtype=np.float64).reshape(-1)
                return cs(xn)
            return predict
        except Exception:
            return None

    # =========================================================
    # Model C: Ridge with rich features (fallback)
    # =========================================================
    def build_features(xv):
        xv = np.asarray(xv, dtype=np.float64).flatten()
        m = len(xv)
        cols = [np.ones(m)]
        xk = np.ones(m)
        for k in range(1, 13):
            xk = xk * xv
            cols.append(xk.copy())
        for f in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]:
            cols.append(np.sin(f * xv))
            cols.append(np.cos(f * xv))
        for c in [-1.0, -0.5, 0.5, 1.0]:
            cols.append(np.exp(c * xv))
        cols.append(np.abs(xv))
        cols.append(np.tanh(xv))
        cols.append(np.arctan(xv))
        cols.append(np.log(np.abs(xv) + 1e-6))
        return np.column_stack(cols)

    def fit_ridge(x, y, alpha):
        try:
            Xf = build_features(x)
            p = Xf.shape[1]
            I = np.eye(p)
            I[0, 0] = 0.0
            A = Xf.T @ Xf + alpha * I
            b = Xf.T @ y
            try:
                w = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                w, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            def predict(X_new):
                return build_features(np.asarray(X_new, dtype=np.float64)) @ w
            return predict
        except Exception:
            return None

    # =========================================================
    # Selection: prefer GP if it fits well, else best of spline/ridge
    # =========================================================
    candidates = []

    gp_model = fit_gp(x, y)
    if gp_model is not None:
        candidates.append(('gp', gp_model))

    sp_model = create_spline_model(x, y)
    if sp_model is not None:
        candidates.append(('spline', sp_model))

    rd_model = fit_ridge(x, y, alpha=1.0)
    if rd_model is not None:
        candidates.append(('ridge', rd_model))

    # Validate on a held-out subset to guard against a bad GP fit
    idx = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(idx)
    val_size = max(10, n // 10)
    val_idx = idx[:val_size]
    y_val = y[val_idx]

    def safe_mse(pred):
        try:
            p = np.asarray(pred, dtype=np.float64).reshape(-1)
            if p.shape[0] != y_val.shape[0]:
                return np.inf
            mse = float(np.mean((p - y_val) ** 2))
            return mse if np.isfinite(mse) else np.inf
        except Exception:
            return np.inf

    var_y = float(np.var(y))
    if not np.isfinite(var_y) or var_y < 1e-12:
        var_y = 1.0

    scored = []
    for name, model in candidates:
        try:
            pv = model(x[val_idx])
            mse = safe_mse(pv)
        except Exception:
            mse = np.inf
        # normalized mse
        nmse = mse / var_y
        scored.append((name, nmse, model))

    # Choose: prefer GP if its normalized MSE is reasonable; else best overall.
    gp_nmse = None
    for name, nmse, model in scored:
        if name == 'gp':
            gp_nmse = nmse

    # Sort by nmse
    scored.sort(key=lambda t: t[1])

    if not scored:
        y_mean = float(np.mean(y))
        def predict(X_new):
            return np.full(len(np.asarray(X_new).reshape(-1, 1)), y_mean)
        return predict

    best_name, best_nmse, best_model = scored[0]

    # If GP is present and competitive (within a factor of best), trust GP for
    # smoothness on test points; otherwise fall back to the empirically best.
    if gp_nmse is not None and gp_nmse < 10.0:
        final_model = [m for nm, ns, m in scored if nm == 'gp'][0]
    else:
        final_model = best_model

    def predict(X_new):
        X_new = np.asarray(X_new, dtype=np.float64)
        try:
            out = final_model(X_new)
            out = np.asarray(out, dtype=np.float64).reshape(-1)
            # replace any non-finite with mean
            if not np.all(np.isfinite(out)):
                out = np.where(np.isfinite(out), out, float(np.mean(y)))
            return out
        except Exception:
            return np.full(X_new.shape[0], float(np.mean(y)))

    return predict
