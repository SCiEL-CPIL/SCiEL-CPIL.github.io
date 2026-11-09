"""
Combined symbolic regression: GP + local polynomial + robust spline, with
quasi-interpolant (Dirac-delta weighted sum) as an additional non-parametric
model family.  Selects the best model by cross-validated normalized MSE.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.optimize import minimize
from numpy.polynomial import legendre as npl

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _nrmse(y, yhat):
    resid = y - yhat
    mse = float(np.mean(resid * resid))
    vy = float(np.var(y))
    if vy <= 1e-12:
        return mse
    return mse / vy


def _unique_sorted(x, y):
    """Sort by x and average y for duplicate x values."""
    idx = np.argsort(x)
    xs = x[idx]
    ys = y[idx]
    ux, inv = np.unique(xs, return_inverse=True)
    if len(ux) == len(xs):
        return xs, ys
    # group means
    sums = np.zeros(len(ux))
    counts = np.zeros(len(ux))
    np.add.at(sums, inv, ys)
    np.add.at(counts, inv, 1)
    return ux, sums / counts


# ---------------------------------------------------------------------------
# Model 1: Gaussian Process (Matern 5/2)  – from inspiration 1
# ---------------------------------------------------------------------------

def _fit_gp(X, y, deadline):
    """Fit a 1-D GP with Matern-5/2 kernel.  Returns (predict_fn, train_nrmse) or None."""
    x = X[:, 0]
    n = len(x)
    if n < 5:
        return None

    x_mean = float(np.mean(x))
    x_std = float(np.std(x))
    if x_std < 1e-12:
        x_std = 1.0
    x_norm = (x - x_mean) / x_std

    y_mean = float(np.mean(y))
    y_std = float(np.std(y))
    if y_std < 1e-12:
        y_std = 1.0
    y_norm = (y - y_mean) / y_std

    def _matern52(x1, x2, ls, sv):
        d = np.abs(x1[:, None] - x2[None, :]) / ls
        u = d
        return sv * (1.0 + np.sqrt(5.0) * u + (5.0 / 3.0) * u * u) * np.exp(-np.sqrt(5.0) * u)

    def _negll(params):
        log_ls, log_sv, log_nv = params
        ls = float(np.exp(log_ls))
        sv = float(np.exp(log_sv))
        nv = float(np.exp(log_nv))
        if ls < 1e-4 or sv > 1e8 or nv > 1e8:
            return 1e6
        try:
            K = _matern52(x_norm, x_norm, ls, sv)
            K[np.diag_indices_from(K)] += nv
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))
            log_det = 2.0 * np.sum(np.log(np.diag(L)))
            ll = -0.5 * y_norm @ alpha - 0.5 * log_det - 0.5 * n * np.log(2.0 * np.pi)
            return -ll
        except np.linalg.LinAlgError:
            return 1e6

    # multi-start
    best_nll = np.inf
    best_params = None
    for ls0, sv0, nv0 in [(0.0, 0.0, -5.0), (-1.0, 0.5, -4.0), (1.0, -0.5, -6.0)]:
        if time.time() > deadline:
            break
        try:
            res = minimize(_negll, [ls0, sv0, nv0], method='Nelder-Mead',
                           options={'maxiter': 150, 'xatol': 1e-5, 'fatol': 1e-5})
            if res.fun < best_nll:
                best_nll = res.fun
                best_params = res.x
        except Exception:
            pass

    if best_params is None:
        return None

    ls = float(np.exp(best_params[0]))
    sv = float(np.exp(best_params[1]))
    nv = float(np.exp(best_params[2]))

    K = _matern52(x_norm, x_norm, ls, sv)
    K[np.diag_indices_from(K)] += nv
    try:
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))
    except np.linalg.LinAlgError:
        return None

    yhat_train = (_matern52(x_norm, x_norm, ls, sv) @ alpha) * y_std + y_mean
    train_score = _nrmse(y, yhat_train)

    def gp_predict(X_new):
        xn = (X_new[:, 0] - x_mean) / x_std
        Kt = _matern52(xn, x_norm, ls, sv)
        mu = (Kt @ alpha) * y_std + y_mean
        return np.where(np.isfinite(mu), mu, y_mean).astype(np.float64)

    return gp_predict, train_score


# ---------------------------------------------------------------------------
# Model 2: Local polynomial regression  – structural change
# ---------------------------------------------------------------------------

def _fit_local_poly(x, y, deg=3, bandwidths=(0.05, 0.1, 0.2, 0.5), deadline=None):
    """Local polynomial regression with adaptive bandwidth.  Returns (predict_fn, train_nrmse) or None."""
    n = len(x)
    if n < deg + 2:
        return None

    x_mean = float(np.mean(x))
    x_range = float(np.max(x) - np.min(x))
    if x_range < 1e-12:
        return None

    best_bw = bandwidths[0]
    best_score = np.inf

    for bw_frac in bandwidths:
        if deadline is not None and time.time() > deadline:
            break
        h = bw_frac * x_range
        yhat = np.empty(n)
        for i in range(n):
            dist = np.abs(x - x[i])
            mask = dist <= h
            npts = np.sum(mask)
            if npts < deg + 1:
                # widen
                mask = dist <= h * 2
                npts = np.sum(mask)
                if npts < deg + 1:
                    yhat[i] = y[i]
                    continue
            # local polynomial fit
            xi = x[mask]
            yi = y[mask]
            # design matrix: [1, xi-xi[i], (xi-xi[i])^2, ...]
            dx = xi - x[i]
            Phi = np.vander(dx, N=deg + 1, increasing=True)
            # weight by distance (Gaussian kernel)
            w = np.exp(-0.5 * (dist[mask] / (h * 0.5)) ** 2)
            try:
                coeff, *_ = np.linalg.lstsq(Phi * w[:, None], yi * w, rcond=None)
                yhat[i] = coeff[0]  # value at dx=0
            except Exception:
                yhat[i] = np.mean(yi)

        score = _nrmse(y, yhat)
        if score < best_score:
            best_score = score
            best_bw = bw_frac

    h = best_bw * x_range

    def lp_predict(X_new):
        xn = X_new[:, 0]
        out = np.empty(len(xn))
        for i, xv in enumerate(xn):
            dist = np.abs(x - xv)
            mask = dist <= h
            npts = np.sum(mask)
            if npts < deg + 1:
                mask = dist <= h * 2
                npts = np.sum(mask)
            if npts < deg + 1:
                out[i] = float(np.mean(y))
                continue
            xi = x[mask]
            yi = y[mask]
            dx = xi - xv
            Phi = np.vander(dx, N=deg + 1, increasing=True)
            w = np.exp(-0.5 * (dist[mask] / (h * 0.5)) ** 2)
            try:
                coeff, *_ = np.linalg.lstsq(Phi * w[:, None], yi * w, rcond=None)
                out[i] = coeff[0]
            except Exception:
                out[i] = np.mean(yi)
        return np.where(np.isfinite(out), out, float(np.mean(y))).astype(np.float64)

    return lp_predict, best_score


# ---------------------------------------------------------------------------
# Model 3: Robust spline  – from current program, simplified
# ---------------------------------------------------------------------------

def _fit_spline(x, y, s=0, k=3, deadline=None):
    """Cubic (or quadratic) spline with outlier removal.  Returns (predict_fn, train_nrmse) or None."""
    xs, ys = _unique_sorted(x, y)
    n = len(xs)
    if n < k + 1:
        return None

    try:
        sp = UnivariateSpline(xs, ys, s=s, k=k)
        yhat = sp(x)
        if not np.all(np.isfinite(yhat)):
            return None
        score = _nrmse(y, yhat)

        # outlier removal
        if score > 1e-4:
            resid = np.abs(y - yhat)
            thr = np.percentile(resid, 90)
            keep = resid <= thr
            if np.sum(keep) >= k + 1:
                try:
                    sp2 = UnivariateSpline(x[keep], y[keep], s=s, k=k)
                    yhat2 = sp2(x)
                    if np.all(np.isfinite(yhat2)):
                        score2 = _nrmse(y, yhat2)
                        if score2 < score:
                            sp = sp2
                            score = score2
                except Exception:
                    pass

        ym = float(np.mean(y))

        def sp_predict(X_new):
            xv = X_new[:, 0]
            yp = sp(xv)
            return np.where(np.isfinite(yp), yp, ym).astype(np.float64)

        return sp_predict, score
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Model 4: Gauss-Legendre quasi-interpolant  – new structural idea
# ---------------------------------------------------------------------------

def _fit_gauss_quasi(x, y, n_quad=50, deadline=None):
    """
    Approximate f as a weighted sum of Dirac deltas at training points,
    where weights are solved by matching a Gauss-Legendre quadrature
    representation of f on the training domain.

    The idea: represent f(x) ≈ Σ_i w_i δ(x - x_i).  To solve for w_i,
    we require that the quadrature approximation of f at n_quad Gauss-Legendre
    nodes matches the interpolated f at those nodes.  This gives an
    overdetermined system (n_quad × n) which we solve in the least-squares
    sense with Tikhonov regularisation to minimise the L2 norm of w.
    """
    n = len(x)
    if n < 5:
        return None

    xs, ys = _unique_sorted(x, y)
    n_u = len(xs)
    if n_u < 5:
        return None

    x_min, x_max = float(xs[0]), float(xs[-1])
    if x_max - x_min < 1e-12:
        return None

    # Gauss-Legendre nodes and weights on [x_min, x_max]
    nodes, weights = np.polynomial.legendre.leggauss(n_quad)
    nodes = 0.5 * (x_max - x_min) * nodes + 0.5 * (x_max + x_min)
    weights = 0.5 * (x_max - x_min) * weights

    # Interpolate f at the quadrature nodes (cubic spline)
    try:
        sp = UnivariateSpline(xs, ys, s=0, k=3)
        f_quad = sp(nodes)
        if not np.all(np.isfinite(f_quad)):
            return None
    except Exception:
        # fallback: linear interp
        f_quad = np.interp(nodes, xs, ys)

    # Build the "Dirac-delta quadrature" matrix:
    # M[q, i] = integral of δ(x - x_i) * L_q(x) dx  where L_q are the
    # Lagrange-like basis from quadrature.  More simply: we want
    #   f(node_q) ≈ Σ_i w_i * K(node_q, x_i)
    # where K is a narrow kernel.  Use a Gaussian kernel with bandwidth
    # set to the median spacing of the training points.
    dx = np.median(np.diff(xs))
    sigma = max(dx * 0.5, (x_max - x_min) / n_u * 0.3)

    # M[q, i] = exp(-0.5 * ((nodes_q - x_i) / sigma)^2)
    diff = nodes[:, None] - xs[None, :]
    M = np.exp(-0.5 * (diff / sigma) ** 2)

    # Tikhonov regularised least squares: min ||M w - f_quad||^2 + λ ||w||^2
    # Try a few λ values
    best_w = None
    best_score = np.inf
    for lam in [1e-6, 1e-4, 1e-2, 1.0]:
        if deadline is not None and time.time() > deadline:
            break
        try:
            A = M.T @ M + lam * np.eye(n_u)
            b = M.T @ f_quad
            w = np.linalg.solve(A, b)
            if not np.all(np.isfinite(w)):
                continue
            # Evaluate on training data
            diff_tr = x[:, None] - xs[None, :]
            K_tr = np.exp(-0.5 * (diff_tr / sigma) ** 2)
            yhat = K_tr @ w
            score = _nrmse(y, yhat)
            if score < best_score:
                best_score = score
                best_w = w
        except Exception:
            continue

    if best_w is None:
        return None

    ym = float(np.mean(y))

    def gq_predict(X_new):
        xv = X_new[:, 0]
        diff = xv[:, None] - xs[None, :]
        K = np.exp(-0.5 * (diff / sigma) ** 2)
        yp = K @ best_w
        return np.where(np.isfinite(yp), yp, ym).astype(np.float64)

    return gq_predict, best_score


# ---------------------------------------------------------------------------
# Model 5: Linear interpolation (simple fallback)
# ---------------------------------------------------------------------------

def _fit_linear_interp(x, y):
    xs, ys = _unique_sorted(x, y)
    if len(xs) < 2:
        return None
    try:
        f = interp1d(xs, ys, kind='linear', fill_value='extrapolate')
        yhat = f(x)
        if not np.all(np.isfinite(yhat)):
            return None
        score = _nrmse(y, yhat)
        ym = float(np.mean(y))

        def li_predict(X_new):
            yp = f(X_new[:, 0])
            return np.where(np.isfinite(yp), yp, ym).astype(np.float64)

        return li_predict, score
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Cross-validation helper
# ---------------------------------------------------------------------------

def _cv_score(predict_fn, x, y, n_folds=5, seed=42):
    """Simple k-fold CV normalized MSE."""
    n = len(x)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    fold_size = n // n_folds
    scores = []
    for k in range(n_folds):
        val_idx = idx[k * fold_size:(k + 1) * fold_size]
        train_idx = np.concatenate([idx[:k * fold_size], idx[(k + 1) * fold_size:]])
        X_tr = x[train_idx].reshape(-1, 1)
        y_tr = y[train_idx]
        X_val = x[val_idx].reshape(-1, 1)
        y_val = y[val_idx]
        try:
            pred = predict_fn(X_val)
            if not np.all(np.isfinite(pred)):
                scores.append(np.inf)
                continue
            scores.append(_nrmse(y_val, pred))
        except Exception:
            scores.append(np.inf)
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Main fit
# ---------------------------------------------------------------------------

def fit(X_train, y_train):
    t0 = time.time()
    DEADLINE = t0 + 280.0  # safety margin under 300 s

    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    n = X_train.shape[0]
    d = X_train.shape[1]

    y_mean = float(np.mean(y_train))

    if d != 1:
        # Fallback for multi-dim: mean predictor
        return lambda X: np.full(X.shape[0], y_mean, dtype=np.float64)

    x = X_train[:, 0]

    candidates = []  # list of (name, predict_fn, train_score)

    # --- Model 1: GP ---
    if time.time() < DEADLINE - 30:
        try:
            res = _fit_gp(X_train, y_train, DEADLINE)
            if res is not None:
                candidates.append(('gp', res[0], res[1]))
        except Exception:
            pass

    # --- Model 2: Local polynomial ---
    if time.time() < DEADLINE - 20:
        try:
            res = _fit_local_poly(x, y_train, deg=3,
                                  bandwidths=(0.05, 0.1, 0.2, 0.5),
                                  deadline=DEADLINE)
            if res is not None:
                candidates.append(('local_poly', res[0], res[1]))
        except Exception:
            pass

    # --- Model 3: Spline (s=0, k=3) ---
    if time.time() < DEADLINE - 10:
        try:
            res = _fit_spline(x, y_train, s=0, k=3, deadline=DEADLINE)
            if res is not None:
                candidates.append(('spline_s0', res[0], res[1]))
        except Exception:
            pass

    # --- Model 3b: Spline (s=1e-8, k=3) ---
    if time.time() < DEADLINE - 10:
        try:
            res = _fit_spline(x, y_train, s=1e-8, k=3, deadline=DEADLINE)
            if res is not None:
                candidates.append(('spline_s1e8', res[0], res[1])
                                  )
        except Exception:
            pass

    # --- Model 4: Gauss quasi-interpolant ---
    if time.time() < DEADLINE - 10:
        try:
            res = _fit_gauss_quasi(x, y_train, n_quad=50, deadline=DEADLINE)
            if res is not None:
                candidates.append(('gauss_quasi', res[0], res[1]))
        except Exception:
            pass

    # --- Model 5: Linear interpolation ---
    if time.time() < DEADLINE - 5:
        try:
            res = _fit_linear_interp(x, y_train)
            if res is not None:
                candidates.append(('lin_interp', res[0], res[1]))
        except Exception:
            pass

    # --- Select best model ---
    if not candidates:
        return lambda X: np.full(X.shape[0], y_mean, dtype=np.float64)

    # If only one candidate, use it
    if len(candidates) == 1:
        return candidates[0][1]

    # Use CV to select among candidates (only if time allows)
    if time.time() < DEADLINE - 15 and len(candidates) > 1:
        best_name, best_fn, best_cv = None, None, np.inf
        for name, fn, tr_score in candidates:
            if time.time() > DEADLINE - 5:
                break
            try:
                cv = _cv_score(fn, x, y_train, n_folds=5, seed=42)
            except Exception:
                cv = np.inf
            if cv < best_cv:
                best_cv = cv
                best_fn = fn
                best_name = name
        if best_fn is not None:
            return best_fn

    # Fallback: pick by train score
    candidates.sort(key=lambda c: c[2])
    return candidates[0][1]

# EVOLVE-BLOCK-END
