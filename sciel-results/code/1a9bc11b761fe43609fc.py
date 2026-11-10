"""Rewritten solver: shared degree<=5 polynomial for unresolved tendency,
robust repeated-CV model selection + extrapolation guard for unseen regime."""
import numpy as np
import time
from math import comb


def _scaled_to_monomial(c_scaled, x_mean, x_std, degree):
    """Convert coefficients from scaled basis z=(x-mean)/std to monomial in x."""
    coeff = np.zeros(degree + 1, dtype=np.float64)
    for j in range(degree + 1):
        cj = c_scaled[j]
        if cj == 0.0:
            continue
        sf = cj / (x_std ** j)
        for k in range(j + 1):
            coeff[k] += sf * comb(j, k) * ((-x_mean) ** (j - k))
    return coeff


def _fit_ridge(Vd, y, lam):
    """Ridge fit in the (pre-scaled) feature matrix Vd. Returns coefficients."""
    d = Vd.shape[1] - 1
    A = Vd.T @ Vd
    b = Vd.T @ y
    A = A + (lam * np.eye(d + 1))
    try:
        c = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        c = np.linalg.lstsq(Vd, y, rcond=None)[0]
    return c


def _cv_mse(Vd, y, n, lam, seeds, k):
    """Repeated k-fold CV MSE for a fixed feature matrix and ridge lambda."""
    total = 0.0
    for seed in seeds:
        idx = np.arange(n)
        np.random.seed(seed)
        np.random.shuffle(idx)
        folds = np.array_split(idx, k)
        for i in range(k):
            va = folds[i]
            tr = np.concatenate([folds[j] for j in range(k) if j != i])
            Vtr = Vd[tr]
            Vva = Vd[va]
            try:
                c = _fit_ridge(Vtr, y[tr], lam)
                pred = Vva @ c
                total += float(np.sum((pred - y[va]) ** 2))
            except Exception:
                return np.inf
    return total / (n * len(seeds))


def construct(training_x, training_subgrid_tendency):
    start = time.time()
    deadline = start + 130.0  # safety margin well inside the 150 s wall-clock kill

    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = x.size
    if n == 0:
        return np.zeros(6, dtype=np.float64)

    y_mean = float(np.mean(y))

    # Robust, minimal-impact clip: remove only pathological spikes (>~18 sigma).
    if n >= 20:
        med = float(np.median(y))
        mad = float(np.median(np.abs(y - med)))
        scale = 1.4826 * mad
        if scale > 1e-12:
            y = np.clip(y, med - 12.0 * scale, med + 12.0 * scale)

    x_mean = float(np.mean(x))
    x_std = float(np.std(x))
    if x_std < 1e-12:
        x_std = 1.0
    x_scaled = (x - x_mean) / x_std
    V_full = np.vander(x_scaled, N=6, increasing=True)  # (n, 6)

    y_std = float(np.std(y))

    def fallback(degree=1):
        try:
            Vd = V_full[:, :degree + 1]
            c = _fit_ridge(Vd, y, 1e-8)
            m = _scaled_to_monomial(c, x_mean, x_std, degree)
            out = np.zeros(6, dtype=np.float64)
            out[:degree + 1] = m
            if not np.all(np.isfinite(out)):
                out = np.array([y_mean, 0, 0, 0, 0, 0], dtype=np.float64)
            return out
        except Exception:
            return np.array([y_mean, 0, 0, 0, 0, 0], dtype=np.float64)

    if n < 8:
        return fallback(min(1, n - 1))

    # Extrapolation guard: keep the monomial polynomial bounded over a range
    # wider than the training x-span (proxy for the unseen F=10 regime).
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    x_range = max(x_max - x_min, 1e-9)
    guard_limit = 50.0 * max(y_std, 1e-3)
    guard_grid = np.linspace(x_min - 0.5 * x_range, x_max + 0.5 * x_range, 200)

    def poly_guard(m):
        try:
            vals = np.full_like(guard_grid, m[-1])
            for i in range(len(m) - 2, -1, -1):
                vals = vals * guard_grid + m[i]
            return float(np.max(np.abs(vals)))
        except Exception:
            return np.inf

    def make_out(d, lam):
        Vd = V_full[:, :d + 1]
        c = _fit_ridge(Vd, y, lam)
        m = _scaled_to_monomial(c, x_mean, x_std, d)
        out = np.zeros(6, dtype=np.float64)
        out[:d + 1] = m
        return out

    degrees = [0, 1, 2, 3, 4, 5]
    lambdas = np.concatenate(([1e-6], np.logspace(-4, 2, 20)))
    seeds = [0, 1, 2]
    k = 5

    candidates = []
    for d in degrees:
        if time.time() > deadline - 20.0:
            break
        Vd = V_full[:, :d + 1]
        for lam in lambdas:
            if time.time() > deadline - 20.0:
                break
            cv = _cv_mse(Vd, y, n, float(lam), seeds, k)
            if not np.isfinite(cv):
                continue
            prior = 1.0 + 0.02 * (d - 1)  # gentle smoothness prior
            candidates.append((cv * prior, d, float(lam)))

    if not candidates:
        return fallback(1)

    candidates.sort(key=lambda t: t[0])

    # Local lambda polish around the CV winner.
    base_cv, base_d, base_lam = candidates[0]
    if time.time() < deadline - 20.0:
        Vd = V_full[:, :base_d + 1]
        polish_lams = [base_lam * f for f in (0.4, 0.7, 1.0, 1.5, 2.5)]
        best_pol_cv = base_cv
        best_pol_lam = base_lam
        for pl in polish_lams:
            if pl <= 0 or not np.isfinite(pl):
                continue
            cv = _cv_mse(Vd, y, n, float(pl), seeds, k)
            if np.isfinite(cv) and cv < best_pol_cv:
                best_pol_cv = cv
                best_pol_lam = float(pl)
        base_lam = best_pol_lam

    # Accept the first (best-CV) candidate that passes the extrapolation guard.
    best = None
    for _, d, lam in candidates:
        if time.time() > deadline:
            break
        try:
            out = make_out(d, lam)
        except Exception:
            continue
        if not np.all(np.isfinite(out)):
            continue
        if poly_guard(out) <= guard_limit:
            best = out
            break

    if best is None:
        # Fallback: best-CV candidate regardless of guard, else a safe low-degree fit.
        _, d, lam = candidates[0]
        try:
            best = make_out(d, lam)
            if not np.all(np.isfinite(best)):
                best = None
        except Exception:
            best = None

    if best is None or not np.all(np.isfinite(best)):
        best = fallback(1)

    return best
