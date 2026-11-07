# EVOLVE-BLOCK-START
import numpy as np
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.linear_model import LinearRegression


def _safe_predict(pred, X):
    """Ensure a finite prediction array of shape (n,)."""
    pred = np.asarray(pred, dtype=np.float64)
    if pred.ndim != 1:
        pred = pred.reshape(-1)
    if pred.shape[0] != X.shape[0]:
        # pad or truncate defensively
        if pred.shape[0] < X.shape[0]:
            pad = np.full(X.shape[0] - pred.shape[0], np.nan)
            pred = np.concatenate([pred, pad])
        else:
            pred = pred[:X.shape[0]]
    if not np.all(np.isfinite(pred)):
        finite = pred[np.isfinite(pred)]
        fill = float(np.mean(finite)) if finite.size else 0.0
        pred = np.where(np.isfinite(pred), pred, fill)
    return pred


def fit(X_train, y_train):
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64).reshape(-1)
    n, d = X.shape

    start = time.time()
    deadline = start + 270.0  # safety margin within 300s

    def elapsed():
        return time.time() - start

    # ---- 1. Polynomial least-squares (degree 3) ----
    def poly_features(X, deg):
        cols = [np.ones(X.shape[0])]
        for k in range(1, deg + 1):
            for i in range(d):
                for j in range(i, d):
                    term = np.ones(X.shape[0])
                    for _ in range(k):
                        pass
                    # build monomial x_i^a * x_j^b with a+b=k, a>=b (i>=j handled)
                    for a in range(k, -1, -1):
                        b = k - a
                        if a < b:
                            continue
                        if a > i and b > 0:
                            continue
                        if b > j and a > 0:
                            continue
                        term = X[:, i] ** a * X[:, j] ** b
                        cols.append(term)
        # The above is messy; use a clean recursive monomial builder instead.
        return np.column_stack(cols)

    def monomials(X, deg):
        cols = [np.ones(X.shape[0])]
        # generate all exponent tuples (e1..ed) with sum <= deg
        def gen(idx, remaining):
            if idx == d:
                yield ()
            else:
                for e in range(remaining + 1):
                    for rest in gen(idx + 1, remaining - e):
                        yield (e,) + rest
        seen = set()
        for exps in gen(0, deg):
            if sum(exps) == 0:
                continue
            if exps in seen:
                continue
            seen.add(exps)
            term = np.ones(X.shape[0])
            for i in range(d):
                if exps[i] > 0:
                    term = term * (X[:, i] ** exps[i])
            cols.append(term)
        return np.column_stack(cols)

    def fit_poly(deg):
        F = monomials(X, deg)
        w, *_ = np.linalg.lstsq(F, y, rcond=None)
        def predict(Xt):
            Ft = monomials(Xt, deg)
            return Ft @ w
        return predict

    models = []
    try:
        models.append(fit_poly(2))
    except Exception:
        pass
    try:
        models.append(fit_poly(3))
    except Exception:
        pass

    # ---- 2. Gaussian Process RBF with several lengthscales ----
    def fit_gp(lengthscale):
        kernel = ConstantKernel(1.0) * RBF(lengthscale=lengthscale, length_scale_bounds=("fixed", "fixed"))
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, n_restarts_optimizer=0, normalize_y=True)
        gp.fit(X, y)
        def predict(Xt):
            return gp.predict(Xt)
        return predict

    for ls in [0.5, 1.0, 2.0, 5.0]:
        if elapsed() > deadline:
            break
        try:
            models.append(fit_gp(ls))
        except Exception:
            pass

    # ---- 3. Random Forest ----
    if elapsed() < deadline - 5.0:
        try:
            rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=1, random_state=0, n_jobs=-1)
            rf.fit(X, y)
            models.append(lambda Xt: rf.predict(Xt))
        except Exception:
            pass

    # Fallback: constant model
    if not models:
        c = float(np.mean(y)) if y.size else 0.0
        return lambda Xt: np.full(Xt.shape[0], c)

    # ---- 4. Stacking: build meta-features from train predictions, fit linear meta ----
    meta_cols = []
    for m in models:
        try:
            p = _safe_predict(m(X), X)
            meta_cols.append(p)
        except Exception:
            meta_cols.append(np.zeros(n))
    meta_X = np.column_stack(meta_cols)
    try:
        meta = LinearRegression()
        meta.fit(meta_X, y)
        use_stack = True
    except Exception:
        use_stack = False

    def predict(Xt):
        Xt = np.asarray(Xt, dtype=np.float64)
        if use_stack:
            cols = []
            for m in models:
                try:
                    cols.append(_safe_predict(m(Xt), Xt))
                except Exception:
                    cols.append(np.zeros(Xt.shape[0]))
            MX = np.column_stack(cols)
            out = _safe_predict(meta.predict(MX), Xt)
        else:
            # average of models
            acc = np.zeros(Xt.shape[0])
            cnt = 0
            for m in models:
                try:
                    acc += _safe_predict(m(Xt), Xt)
                    cnt += 1
                except Exception:
                    pass
            out = acc / max(cnt, 1)
        return out

    return predict
# EVOLVE-BLOCK-END
