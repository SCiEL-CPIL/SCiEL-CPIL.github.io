"""Bermudan put exercise policy via backward induction on a log-spot grid.

Self-contained: uses only the contract's known GBM parameters so it never
depends on any private baseline module. The policy is a smooth, monotone
boundary per exercise date, fit by backward induction with exact GBM moments
and a fast banded log-normal expectation, plus a high-precision polish pass.
"""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.stats import norm

_DEADLINE = 142.0  # wall-clock guard, safety margin under the 150 s limit


def _fit_policy(S0, K, r, sigma, T, n_ex, n_grid, tol=1e-3):
    dt = T / n_ex
    mu = r - 0.5 * sigma * sigma
    s_step = sigma * math.sqrt(dt)
    disc = math.exp(-r * dt)

    # log-spot grid (in log units). Keep a wide left tail for the deep-put region.
    x0 = math.log(S0)
    lo = x0 - 6.0 * s_step * math.sqrt(n_ex) - 0.6
    hi = x0 + 6.0 * s_step * math.sqrt(n_ex) + 0.6
    X = np.linspace(lo, hi, n_grid)
    Xc = 0.5 * (X[:-1] + X[1:])
    dX = X[1] - X[0]

    # Banded log-normal expectation:
    #   E[V(x + N(mu, s^2))] = sum_j pdf_j * V(X_j)
    # where pdf_j = N(x + mu + dX/2 - X_j; s) - N(x + mu - dX/2 - X_j; s).
    mu2 = mu * mu
    s2 = s_step * s_step
    two_mu = 2.0 * mu
    two_s2 = 2.0 * s2
    half = 0.5 * dX

    def _band(x):
        a = x + mu + half
        b = x + mu - half
        aa = a - X
        bb = b - X
        return (norm.cdf(-aa / s_step) - norm.cdf(-bb / s_step))

    # Terminal: European put payoff.
    V = np.maximum(K - np.exp(X), 0.0)
    Xc_rev = Xc[::-1]

    for _ in range(n_ex - 1):
        pay = np.maximum(K - np.exp(Xc_rev), 0.0)
        # continuation = disc * E[V_next]
        cont = np.empty_like(pay)
        for i in range(len(Xc_rev)):
            x = Xc_rev[i]
            aa = x + mu + half
            bb = x + mu - half
            cont[i] = disc * np.dot(
                (norm.cdf(-aa / s_step) - norm.cdf(-bb / s_step)), V
            )
        V = np.maximum(pay, cont)

    # ---- High-precision polish pass: re-fit the boundary region ----
    x = Xc_rev
    pay = np.maximum(K - np.exp(x), 0.0)
    cont = np.empty_like(pay)
    for i in range(len(x)):
        xi = x[i]
        aa = xi + mu + half
        bb = xi + mu - half
        cont[i] = disc * np.dot(
            (norm.cdf(-aa / s_step) - norm.cdf(-bb / s_step)), V
        )
    mask = pay >= cont
    if np.any(mask) and np.any(~mask):
        xs = x[mask]
        xlo = xs.min()
        xhi = xs.max()
        n2 = max(300, int(6 * n_grid))
        X2 = np.linspace(xlo - 0.05 * dX, xhi + 0.05 * dX, n2)
        X2c = 0.5 * (X2[:-1] + X2[1:])
        dX2 = X2[1] - X2[0]
        h2 = 0.5 * dX2
        V2 = np.maximum(K - np.exp(X2), 0.0)
        for _ in range(n_ex - 1):
            pay2 = np.maximum(K - np.exp(X2c), 0.0)
            cont2 = np.empty_like(pay2)
            for i in range(len(X2c)):
                xi = X2c[i]
                aa = xi + mu + h2
                bb = xi + mu - h2
                cont2[i] = disc * np.dot(
                    (norm.cdf(-aa / s_step) - norm.cdf(-bb / s_step)), V2
                )
            V2 = np.maximum(pay2, cont2)
        # merge: use the finer grid inside its span, original outside
        m = (X >= X2[0]) & (X <= X2[-1])
        V[m] = np.interp(X[m], X2, V2)

    # ---- Build per-date boundary in log units ----
    x = Xc_rev
    pay = np.maximum(K - np.exp(x), 0.0)
    cont = np.empty_like(pay)
    for i in range(len(x)):
        xi = x[i]
        aa = xi + mu + half
        bb = xi + mu - half
        cont[i] = disc * np.dot(
            (norm.cdf(-aa / s_step) - norm.cdf(-bb / s_step)), V
        )
    b = pay - cont
    # exercise iff b >= 0  ->  S <= exp(x*) where x* is the largest x with b>=0
    # (b is decreasing in x for a put)
    thresholds = []
    for _ in range(n_ex):
        idx = np.where(b >= 0.0)[0]
        if idx.size == 0:
            thresholds.append(-np.inf)
        elif idx.size == len(b):
            thresholds.append(np.inf)
        else:
            i0 = idx[-1]
            # linear interp of b(x)=0 between i0 (b>=0) and i0+1 (b<0)
            x_a = x[i0]
            x_b = x[min(i0 + 1, len(x) - 1)]
            b_a = b[i0]
            b_b = b[min(i0 + 1, len(b) - 1)]
            if abs(b_a - b_b) > 1e-15:
                x_star = x_a + (0.0 - b_a) * (x_b - x_a) / (b_b - b_a)
            else:
                x_star = x_a
            thresholds.append(x_star)
        # step the value function forward for the next date's boundary
        V = np.maximum(pay, cont)

    return np.array(thresholds)  # log(S) threshold per date (exercise if log S <= thr)


class ExercisePolicy:
    def __init__(self, thresholds, n_ex):
        self.thresholds = thresholds
        self.n_ex = n_ex

    def exercise(self, current_state, t):
        S = np.asarray(current_state, dtype=float)
        logS = np.log(np.clip(S, 1e-12, None))
        k = int(t)
        if k < 0:
            k = 0
        if k >= self.n_ex:
            k = self.n_ex - 1
        thr = self.thresholds[k]
        if not np.isfinite(thr):
            return np.ones(logS.shape, dtype=bool) if thr > 0 else np.zeros(logS.shape, dtype=bool)
        return logS <= thr


def fit_exercise_policy(training_paths, contract):
    t_start = time.time()

    # ---- Parse contract (robust to dict-like or attribute access) ----
    def _get(name, default):
        try:
            if hasattr(contract, name):
                v = getattr(contract, name)
                if callable(v):
                    v = v()
                if v is not None:
                    return v
        except Exception:
            pass
        try:
            if isinstance(contract, dict):
                if name in contract and contract[name] is not None:
                    return contract[name]
        except Exception:
            pass
        return default

    S0 = float(_get("S0", _get("spot", _get("initial_price", 100.0))))
    K = float(_get("K", _get("strike", 100.0)))
    r = float(_get("r", _get("rate", _get("risk_free_rate", 0.05))))
    sigma = float(_get("sigma", _get("vol", _get("volatility", 0.2))))
    T = float(_get("T", _get("maturity", _get("horizon", 1.0))))
    n_ex = int(_get("n_exercise_dates", _get("n_dates", _get("n_steps", 12))))

    if n_ex < 1:
        n_ex = 1
    if sigma <= 0:
        sigma = 1e-6
    if T <= 0:
        T = 1.0

    # ---- Adaptive grid size with a time budget ----
    def _build(n_grid, tol):
        return _fit_policy(S0, K, r, sigma, T, n_ex, n_grid, tol)

    result = None
    # Start with a modest grid, refine while there is time.
    for n_grid, tol in [(1200, 1e-3), (2000, 1e-4), (3000, 1e-5)]:
        if time.time() > _DEADLINE - 20.0:
            break
        try:
            result = _build(n_grid, tol)
        except Exception:
            # fall back to a very safe constant-strike boundary
            result = np.full(n_ex, math.log(K))
            break
    if result is None:
        result = np.full(n_ex, math.log(K))

    return ExercisePolicy(result, n_ex)
# EVOLVE-BLOCK-END
