"""Combined closure: centered/standardized ridge polynomial fit with CV model selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _fit_centered(x, y, degree, alpha):
    """Fit a degree-`degree` polynomial in centered-standardized x, with ridge regularization.
    Returns (coefficients_in_centered_basis, mu, sigma)."""
    n = len(x)
    mu = float(np.mean(x))
    sigma = float(np.std(x))
    if sigma < 1e-12:
        sigma = 1.0
    xc = (x - mu) / sigma
    V = np.column_stack([xc**k for k in range(degree + 1)])
    if alpha > 0:
        A = V.T @ V
        b = V.T @ y
        reg = alpha * np.eye(A.shape[0])
        reg[0, 0] += 1e-12
        try:
            c = np.linalg.solve(A + reg, b)
        except np.linalg.LinAlgError:
            c = np.zeros(degree + 1)
    else:
        try:
            c = np.linalg.lstsq(V, y, rcond=None)[0]
        except Exception:
            c = np.zeros(degree + 1)
    return c, mu, sigma


def _to_original_basis(c, mu, sigma, degree):
    """Convert centered-standardized coefficients to original x basis.
    p(x) = sum_i c_i * ((x-mu)/sigma)^i ; return coefficients for sum_k C_k * x^k."""
    C = np.zeros(degree + 1)
    for i in range(degree + 1):
        ci = c[i]
        if ci == 0.0:
            continue
        inv_si = (1.0 / sigma) ** i
        for k in range(i + 1):
            # binom(i, k)
            if k == 0 or k == i:
                Ck = 1.0
            else:
                mm = min(k, i - k)
                num = 1
                den = 1
                for t in range(1, mm + 1):
                    num *= (i - mm + t)
                    den *= t
                Ck = num / den
            C[k] += ci * Ck * ((-mu) ** (i - k)) * inv_si
    return C


def _cv_mse(x, y, degree, alpha, K=5, seed=0):
    """Estimate prediction MSE via K-fold CV on the (centered-standardized) polynomial."""
    n = len(x)
    if n < degree + 2:
        return np.inf
    if n < 5000:
        rng = np.random.RandomState(seed)
        idx = rng.permutation(n)
        fold_size = n // K
        mse_sum = 0.0
        cnt = 0
        for k in range(K):
            lo = k * fold_size
            hi = lo + fold_size
            if k == K - 1:
                hi = n
            val_idx = idx[lo:hi]
            train_idx = np.concatenate([idx[:lo], idx[hi:]])
            if len(train_idx) < degree + 2:
                continue
            c, mu, sigma = _fit_centered(x[train_idx], y[train_idx], degree, alpha)
            pred = np.polynomial.polynomial.polyval((x[val_idx] - mu) / sigma, c)
            mse_sum += np.mean((pred - y[val_idx]) ** 2)
            cnt += 1
        return mse_sum / max(cnt, 1)
    else:
        # large n: random holdout (cheaper)
        rng = np.random.RandomState(seed)
        perm = rng.permutation(n)
        n_val = min(1000, n // 10)
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]
        c, mu, sigma = _fit_centered(x[train_idx], y[train_idx], degree, alpha)
        pred = np.polynomial.polynomial.polyval((x[val_idx] - mu) / sigma, c)
        return np.mean((pred - y[val_idx]) ** 2)


def construct(training_x, training_subgrid_tendency):
    t_start = time.time()
    deadline = t_start + 130.0  # safety margin before the 150s kill

    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    n = len(x)
    if n < 6:
        return np.array([float(np.mean(y)) if n else 0.0, 0, 0, 0, 0, 0], dtype=np.float64)

    y_var = float(np.var(y))
    if y_var < 1e-15:
        y_var = 1e-15

    best_score = np.inf
    best_coeffs = np.array([float(np.mean(y)), 0, 0, 0, 0, 0], dtype=np.float64)

    degrees = [1, 2, 3, 4, 5]
    alphas = [0.0, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]

    def _consider(deg, alpha):
        nonlocal best_score, best_coeffs
        cv = _cv_mse(x, y, deg, alpha)
        if not np.isfinite(cv):
            return
        score = cv / y_var
        if score < best_score:
            best_score = score
            c_centered, mu, sigma = _fit_centered(x, y, deg, alpha)
            coeffs_orig = _to_original_basis(c_centered, mu, sigma, deg)
            bc = np.zeros(6)
            bc[:deg + 1] = coeffs_orig
            if np.all(np.isfinite(bc)):
                best_coeffs = bc

    # Coarse pass
    for deg in degrees:
        if time.time() > deadline:
            break
        for alpha in alphas:
            if time.time() > deadline:
                break
            try:
                _consider(deg, float(alpha))
            except Exception:
                continue

    # Refine: finer alpha grid around the whole range if time remains
    elapsed = time.time() - t_start
    remaining = deadline - elapsed
    if remaining > 8.0:
        refined_alphas = np.logspace(-5, 3, 16)
        for deg in degrees:
            if time.time() > deadline:
                break
            for alpha in refined_alphas:
                if time.time() > deadline:
                    break
                try:
                    _consider(deg, float(alpha))
                except Exception:
                    continue

    if not np.all(np.isfinite(best_coeffs)):
        best_coeffs = np.array([float(np.mean(y)), 0, 0, 0, 0, 0], dtype=np.float64)

    return best_coeffs
# EVOLVE-BLOCK-END
