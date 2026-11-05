"""Rewritten program: structured convex-QP search over autocorrelation lag peaks."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import comb

N = 400
DEADLINE = 150.0
EPS = 1e-4
TOL = 1e-8
MAX_IT = 100000
SEED = 20240607


def _value(h):
    s = float(np.sum(h))
    if s == 0.0:
        return float("inf")
    c = np.convolve(h, h)
    return abs(2.0 * N * np.max(c)) / (s * s)


def _grad_c(h, k):
    """Gradient of c_k = sum_i h[i] h[i+k] w.r.t. h (full, signed)."""
    g = np.zeros(N)
    for i in range(N):
        j = i + k
        if j < N:
            g[i] += h[j]
            g[j] += h[i]
    return g


def _solve_qp(k, h0, t0):
    """Minimize c_k s.t. sum h = N and c_k >= c_0 + EPS via projected gradient."""
    h = np.asarray(h0, dtype=float).copy()
    t = float(t0)
    for _ in range(MAX_IT):
        c0 = float(np.dot(h, h))
        ck = float(np.dot(h[:N - k], h[k:]))
        gk = _grad_c(h, k)
        g0 = 2.0 * h
        # gradient of objective c_k
        go = gk
        # equality multiplier: sum h = N
        lam = (N - float(np.sum(h))) / float(N)
        go = go - lam
        # inequality (active) c_k - c_0 >= EPS:  mu*(gk - g0) - mu
        if ck - c0 <= EPS + TOL:
            mu = max(0.0, (EPS - (ck - c0)) / float(np.dot(gk, gk) + 1e-12))
            go = go + mu * (gk - g0) - mu
        # projected gradient step onto sum h = N
        step = go - np.mean(go)
        h = h - step
        t = max(0.0, t - EPS)
        c0 = float(np.dot(h, h))
        ck = float(np.dot(h[:N - k], h[k:]))
        if ck - c0 > EPS and abs(step).max() < 1e-9:
            break
    return h, t


def _refine(h, k):
    """Final polish: minimize c_k with equality + active inequality via L-BFGS-B."""
    def obj(v):
        x = v[:N]
        c0 = float(np.dot(x, x))
        ck = float(np.dot(x[:N - k], x[k:]))
        return ck + 1e6 * max(0.0, (c0 + EPS) - ck) ** 2

    def jac(v):
        x = v[:N]
        gk = _grad_c(x, k)
        g0 = 2.0 * x
        c0 = float(np.dot(x, x))
        ck = float(np.dot(x[:N - k], x[k:]))
        slack = (c0 + EPS) - ck
        j = gk.copy()
        if slack > 0.0:
            j = j + 2e6 * slack * (gk - g0)
        return j

    v0 = np.concatenate([h, [max(0.0, 1e-6)]])
    cons = [{"type": "eq", "fun": lambda v: float(np.sum(v[:N])) - N,
             "jac": lambda v: np.concatenate([np.ones(N), [0.0]])}]
    try:
        res = minimize(obj, v0, jac=jac, method="L-BFGS-B",
                       constraints=cons, options={"maxiter": 5000, "ftol": 1e-14})
        x = res.x[:N]
    except Exception:
        x = h
    return x


def _sine_h(k, phase):
    h = np.sin(2.0 * np.pi * k / N * np.arange(N) + phase)
    s = float(np.sum(h))
    if abs(s) > 1e-12:
        h = h * (N / s)
    return h


def _gauss_h(k, phase):
    x = np.arange(N)
    h = np.cos(2.0 * np.pi * k / N * x + phase) + 0.25 * np.cos(4.0 * np.pi * k / N * x)
    s = float(np.sum(h))
    if abs(s) > 1e-12:
        h = h * (N / s)
    return h


def _block_h(k, phase):
    # two-lobe square wave with period ~ N/k
    x = np.arange(N)
    h = np.sign(np.sin(2.0 * np.pi * k / N * x + phase))
    s = float(np.sum(h))
    if abs(s) > 1e-12:
        h = h * (N / s)
    return h


def construct():
    import time
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    best_h = np.ones(N)
    best_val = _value(best_h)  # 2.0

    # candidate lags: small (many terms, large potential peak) and a few mid
    lags = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 25, 32, 40, 50, 64, 80, 100, 128]

    for k in lags:
        if time.time() - t0 > DEADLINE - 25.0:
            break
        inits = [np.ones(N),
                 _sine_h(k, 0.0), _sine_h(k, np.pi / 2), _sine_h(k, np.pi),
                 _gauss_h(k, 0.0), _gauss_h(k, np.pi),
                 _block_h(k, 0.0), _block_h(k, np.pi / 2)]
        # a couple of random inits
        for _ in range(2):
            inits.append(rng.standard_normal(N))
        for h0 in inits:
            if time.time() - t0 > DEADLINE - 25.0:
                break
            try:
                h, t = _solve_qp(k, h0, 1.0)
                h = _refine(h, k)
                v = _value(h)
                if np.all(np.isfinite(h)) and abs(np.sum(h)) > 1e-8 and v < best_val:
                    best_val = v
                    best_h = h
            except Exception:
                continue

    return best_h


# ---- quick self-test when run directly (not used by evaluator) ----
if __name__ == "__main__":
    import time
    t0 = time.time()
    h = construct()
    print("value:", _value(h), "time:", time.time() - t0)
