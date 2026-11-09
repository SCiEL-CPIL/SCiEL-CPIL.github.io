# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

_DEADLINE = 600.0  # safety margin before 630s kill
_START = time.time()

def _elapsed():
    return time.time() - _START

def _deadline_reached():
    return _elapsed() > _DEADLINE


def _shannon_bits(probs):
    """Shannon entropy in bits, ignoring zero probabilities."""
    probs = np.asarray(probs, dtype=np.float64)
    nz = probs > 0.0
    if not np.any(nz):
        return 0.0
    p = probs[nz]
    p = p / p.sum()
    return float(-np.sum(p * np.log2(p)))


def _projection_probs(P, r):
    """
    P has shape (n, n) with P[y, x].
    Returns the pooled distribution of value = x + r*y.
    """
    n = P.shape[0]
    xs = np.arange(n, dtype=np.float64)
    ys = np.arange(n, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys, indexing='xy')  # X[i,j]=x, Y[i,j]=y
    # P[y, x] -> index [y, x] = [j, i]; but we built meshgrid with indexing='xy'
    # so X[i,j]=x_j, Y[i,j]=y_i. We need P[j, i] where j=y, i=x.
    # Transpose: P_t[i, j] = P[j, i] = P[y, x] with x=j, y=i
    P_t = P.T  # P_t[i, j] = P[j, i]
    vals = X + r * Y
    # Pool by value
    order = np.argsort(vals.ravel())
    vals_flat = vals.ravel()[order]
    p_flat = P_t.ravel()[order]
    # Find unique values and sum probabilities
    uniq, idx = np.unique(vals_flat, return_index=True)
    # Build sum per unique
    sums = np.zeros(len(uniq), dtype=np.float64)
    # Use np.add.at for stable accumulation
    np.add.at(sums, np.searchsorted(uniq, vals_flat), p_flat)
    total = sums.sum()
    if total <= 0:
        return np.zeros(len(uniq))
    return sums / total


def _objective_from_P(P):
    """
    Return D / M where D = H_{-1}, M = max(H_0, H_1, H_2).
    Returns -inf if invalid (M <= 1e-9).
    """
    P = np.asarray(P, dtype=np.float64)
    P = np.clip(P, 0.0, None)
    s = P.sum()
    if s <= 1e-9:
        return -np.inf
    P = P / s
    Hm1 = _shannon_bits(_projection_probs(P, -1.0))
    H0 = _shannon_bits(_projection_probs(P, 0.0))
    H1 = _shannon_bits(_projection_probs(P, 1.0))
    H2 = _shannon_bits(_projection_probs(P, 2.0))
    M = max(H0, H1, H2)
    if M <= 1e-9:
        return -np.inf
    return Hm1 / M


def _parametric_P(alpha, beta, gamma, n=10):
    """
    P[y, x] ∝ exp(-alpha*x^2 - beta*y^2 - gamma*x*y)
    x, y in 0..n-1
    """
    xs = np.arange(n, dtype=np.float64)
    ys = np.arange(n, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys, indexing='xy')  # X[i,j]=x, Y[i,j]=y
    logw = -alpha * X**2 - beta * Y**2 - gamma * X * Y
    # subtract max for numerical stability
    logw = logw - logw.max()
    w = np.exp(logw)
    w_t = w.T  # w_t[j, i] = w[i, j] = weight for (x=i, y=j) -> P[y=j, x=i]
    s = w_t.sum()
    if s <= 0:
        return np.ones((n, n)) / (n * n)
    return w_t / s


def _neg_obj_params(params):
    """Negative of the objective for the parametric model."""
    alpha, beta, gamma = params
    P = _parametric_P(alpha, beta, gamma)
    val = _objective_from_P(P)
    if not np.isfinite(val):
        return 1e9
    return -val


def _direct_neg_obj(params):
    """Negative objective for direct 100-param optimization (simplex-friendly)."""
    n = 10
    P = np.asarray(params, dtype=np.float64).reshape(n, n)
    P = np.clip(P, 0.0, None)
    s = P.sum()
    if s <= 1e-9:
        return 1e9
    P = P / s
    val = _objective_from_P(P)
    if not np.isfinite(val):
        return 1e9
    return -val


def _refine_with_bfgs(x0):
    """Local refinement using L-BFGS-B on the parametric model."""
    try:
        res = minimize(
            _neg_obj_params,
            x0,
            method='L-BFGS-B',
            options={'maxiter': 2000, 'maxfun': 4000},
        )
        if res.success or res.fun < 1e9:
            return res.x
    except Exception:
        pass
    return x0


def _try_direct_optimization(n=10):
    """
    Directly optimize the 100-cell distribution using scipy differential evolution
    or Nelder-Mead on a transformed parameterization.
    """
    # Use logit-free approach: optimize on P directly with bounds [0, 1]
    # but that's 100 dims; instead use a lower-dimensional ansatz first,
    # then do a final direct polish on the best P found.
    pass


def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Returns a 10x10 array P of finite weights, P[y, x] for integers x, y in 0..9.
    """
    n = 10

    # Baseline: uniform
    P_uniform = np.ones((n, n), dtype=np.float64) / (n * n)
    best_P = P_uniform.copy()
    best_val = _objective_from_P(best_P)

    # --- Phase 1: Parametric ansatz optimization ---
    # P[y, x] ∝ exp(-alpha*x^2 - beta*y^2 - gamma*x*y)
    # We want to maximize H_{-1} / max(H_0, H_1, H_2)
    # H_{-1} is entropy of x-y distribution. To maximize it, we want x-y to be
    # as uniform as possible over its range [-9, 9].
    # H_0 is entropy of x (marginal), H_1 of x+y, H_2 of x+2y.
    # We want to keep these as small as possible relative to H_{-1}.

    # Try a grid of (alpha, beta, gamma)
    # alpha controls x spread, beta controls y spread, gamma controls correlation
    # If gamma is negative, x and y are negatively correlated, which makes x-y
    # have larger variance and potentially higher entropy.

    alphas = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    betas = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    gammas = [-0.5, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]

    candidates = []
    for a in alphas:
        for b in betas:
            for g in gammas:
                P = _parametric_P(a, b, g)
                val = _objective_from_P(P)
                if np.isfinite(val) and val > best_val:
                    best_val = val
                    best_P = P.copy()
                candidates.append((a, b, g, val))

    if _deadline_reached():
        return np.clip(best_P, 0.0, None) / np.clip(best_P.sum(), 1e-9, None)

    # Refine the best few candidates with L-BFGS-B
    candidates.sort(key=lambda t: t[3], reverse=True)
    for (a, b, g, v) in candidates[:5]:
        if _deadline_reached():
            break
        x0 = np.array([a, b, g], dtype=np.float64)
        xopt = _refine_with_bfgs(x0)
        P = _parametric_P(*xopt)
        val = _objective_from_P(P)
        if np.isfinite(val) and val > best_val:
            best_val = val
            best_P = P.copy()

    # --- Phase 2: Direct optimization polish ---
    # Use scipy.optimize.minimize on the 100-dim P with bounds, starting from best_P
    # We'll use Nelder-Mead or Powell on the clipped, normalized P
    x0 = best_P.ravel().copy()

    def _direct_obj(x):
        P = np.clip(x, 0.0, None).reshape(n, n)
        s = P.sum()
        if s <= 1e-9:
            return 1e9
        P = P / s
        val = _objective_from_P(P)
        if not np.isfinite(val):
            return 1e9
        return -val

    try:
        res = minimize(
            _direct_obj,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10},
        )
        if res.fun < 1e9:
            P = np.clip(res.x, 0.0, None).reshape(n, n)
            s = P.sum()
            if s > 1e-9:
                P = P / s
                val = _objective_from_P(P)
                if np.isfinite(val) and val > best_val:
                    best_val = val
                    best_P = P.copy()
    except Exception:
        pass

    if _deadline_reached():
        P = np.clip(best_P, 0.0, None)
        s = P.sum()
        if s > 1e-9:
            P = P / s
        return P

    # --- Phase 3: Try a few more structured ansatze ---
    # Ansatz: P[y, x] ∝ (1 + c * (x - y)^2) for various c
    # This directly weights cells by |x-y| to flatten the x-y distribution
    for c in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
        if _deadline_reached():
            break
        xs = np.arange(n, dtype=np.float64)
        ys = np.arange(n, dtype=np.float64)
        X, Y = np.meshgrid(xs, ys, indexing='xy')
        w = 1.0 + c * (X - Y)**2
        P = w.T / w.T.sum()
        val = _objective_from_P(P)
        if np.isfinite(val) and val > best_val:
            best_val = val
            best_P = P.copy()

    # Ansatz: P[y, x] ∝ (1 + c * |x - y|)
    for c in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
        if _deadline_reached():
            break
        xs = np.arange(n, dtype=np.float64)
        ys = np.arange(n, dtype=np.float64)
        X, Y = np.meshgrid(xs, ys, indexing='xy')
        w = 1.0 + c * np.abs(X - Y)
        P = w.T / w.T.sum()
        val = _objective_from_P(P)
        if np.isfinite(val) and val > best_val:
            best_val = val
            best_P = P.copy()

    # Ansatz: P[y, x] ∝ exp(c * (x - y))  -- tilts the distribution
    for c in [-0.3, -0.2, -0.1, 0.1, 0.2, 0.3]:
        if _deadline_reached():
            break
        xs = np.arange(n, dtype=np.float64)
        ys = np.arange(n, dtype=np.float64)
        X, Y = np.meshgrid(xs, ys, indexing='xy')
        w = np.exp(c * (X - Y))
        P = w.T / w.T.sum()
        val = _objective_from_P(P)
        if np.isfinite(val) and val > best_val:
            best_val = val
            best_P = P.copy()

    # Final direct polish from the best P found
    x0 = best_P.ravel().copy()
    try:
        res = minimize(
            _direct_obj,
            x0,
            method='Powell',
            options={'maxiter': 3000, 'ftol': 1e-10, 'xtol': 1e-8},
        )
        if res.fun < 1e9:
            P = np.clip(res.x, 0.0, None).reshape(n, n)
            s = P.sum()
            if s > 1e-9:
                P = P / s
                val = _objective_from_P(P)
                if np.isfinite(val) and val > best_val:
                    best_val = val
                    best_P = P.copy()
    except Exception:
        pass

    # Ensure validity
    P = np.clip(best_P, 0.0, None)
    s = P.sum()
    if s <= 1e-9:
        P = np.ones((n, n), dtype=np.float64) / (n * n)
    else:
        P = P / s

    return P


# EVOLVE-BLOCK-END
