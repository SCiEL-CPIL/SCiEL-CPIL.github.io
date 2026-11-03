"""Signed autocorrelation max-abs minimizer via smooth continuous relaxation + local search."""
import time

N = 400
DEADLINE = time.time() + 140.0  # safety margin under the 150s kill

import numpy as np
from scipy.optimize import minimize


def _score(s):
    s = np.asarray(s)
    ac = np.correlate(s, s, mode="full")
    integral = s.sum()
    if integral == 0:
        return float("inf")
    return float(np.max(np.abs(ac))) / (integral * integral)


def _smooth_obj(x):
    """Epigraph-style smooth surrogate: softmax over |ac| scaled by squared integral."""
    ac = np.correlate(x, x, mode="full")
    integral = x.sum()
    if abs(integral) < 1e-8:
        return 1e12
    a = np.abs(ac) / (integral * integral)
    tau = 5.0
    m = a.max()
    return float(m + np.log(np.sum(np.exp((a - m) / tau))) * tau)


def _smooth_grad(x):
    ac = np.correlate(x, x, mode="full")
    integral = x.sum()
    if abs(integral) < 1e-8:
        return np.zeros_like(x)
    a = np.abs(ac) / (integral * integral)
    tau = 5.0
    m = a.max()
    p = np.exp((a - m) / tau)
    p = p / p.sum()
    # d a_i / d x_j  (ac = x * x, full)
    # ac_k = sum_j x_j x_{j+k}; sign handling for abs
    n = len(x)
    L = 2 * n - 1
    jac = np.zeros((L, n))
    for k in range(L):
        lo = max(0, k - n + 1)
        hi = min(n - 1, k)
        # ac_k = sum_{j=lo..hi} x_j x_{j+k}
        idx = np.arange(lo, hi + 1)
        jac[k, idx] += 2.0 * x[idx + k]
        jac[k, idx + k] += 2.0 * x[idx]
    da = np.sign(ac) * jac / (integral * integral)
    # d/integral^2 factor: a = |ac|/I^2, da = da_ac/I^2 - 2|ac|*I'/I^3
    g = da.T @ p
    g -= 2.0 * (np.abs(ac) * p).sum() * np.ones(n) / integral
    return g


def _local_search(s, budget_iters=4000):
    """Flip individual bits to reduce peak autoconvolution ratio."""
    s = s.copy()
    ac = np.correlate(s, s, mode="full")
    integral = s.sum()
    best = _score(s)
    improved = True
    it = 0
    while improved and it < budget_iters and time.time() < DEADLINE:
        improved = False
        it += 1
        # candidate: flip bit i changes ac by -2*s_i*(s convolved)
        # new_ac_k = ac_k - 2*s_i*(s_{k-i} + s_{k-i-?}) ... use direct delta
        # delta to ac from flipping i: ac' = ac - 2*s_i*(s shifted)
        # ac_k = sum_j s_j s_{j+k}; removing 2*s_i term where j=i or j+k=i
        # For each k: terms involving i: j=i -> s_i s_{i+k}; j+k=i -> s_{i-k} s_i
        # delta_k = -2*s_i*(s_{i+k}+s_{i-k}) (with bounds)
        n = len(s)
        L = 2 * n - 1
        for i in range(n):
            si = s[i]
            # contributions
            # s_{i+k} for k such that 0<=i+k<n
            # s_{i-k} for k such that 0<=i-k<n
            delta = np.zeros(L)
            # j=i: s_i * s_{i+k}, k in 0..n-1-i
            k1 = np.arange(0, n - i)
            delta[k1] -= 2.0 * si * s[i + k1]
            # j+k=i: k in i-n+1..i, i.e. k such that 0<=i-k<n
            k2 = np.arange(i - (n - 1), i + 1)
            # valid k2: 0<=i-k<n -> k in i-n+1..i
            valid = (k2 >= 0) & (k2 < L)
            k2 = k2[valid]
            idx = i - k2
            valid2 = (idx >= 0) & (idx < n)
            k2 = k2[valid2]
            idx = i - k2
            delta[k2] -= 2.0 * si * s[idx]
            new_ac = ac + delta
            new_int = integral - 2.0 * si
            if new_int == 0:
                continue
            cand = float(np.max(np.abs(new_ac))) / (new_int * new_int)
            if cand < best - 1e-12:
                s[i] = -si
                ac = new_ac
                integral = new_int
                best = cand
                improved = True
                if time.time() > DEADLINE:
                    return s, best
    return s, best


def construct():
    start = time.time()
    rng = np.random.default_rng(12345)

    # Start from a smooth low-pass filtered random impulse train
    # Build a base random sequence, low-pass filter it, then optimize coefficients
    base = rng.choice([-1.0, 1.0], size=N)
    # Low-pass via moving average to get a smooth surrogate
    kernel = np.ones(8) / 8.0
    smooth = np.convolve(base, kernel, mode="same")

    # Optimize the continuous waveform to minimize peak autoconvolution ratio
    x0 = smooth.copy()
    res = minimize(
        _smooth_obj, x0, jac=_smooth_grad, method="L-BFGS-B",
        options={"maxiter": 400, "gtol": 1e-6, "ftol": 1e-10},
    )
    x_opt = res.x

    # Sign-round to discrete ±1
    s = np.sign(x_opt)
    s[s == 0] = 1.0
    best_s = s.copy()
    best_val = _score(best_s)

    # Local search polish (flip-based) with deadline guard
    if time.time() < DEADLINE - 5:
        best_s, best_val = _local_search(best_s, budget_iters=6000)

    # Extra random restarts if time remains
    restarts = 0
    while time.time() < DEADLINE - 10 and restarts < 3:
        restarts += 1
        base = rng.choice([-1.0, 1.0], size=N)
        smooth = np.convolve(base, kernel, mode="same")
        r2 = minimize(
            _smooth_obj, smooth, jac=_smooth_grad, method="L-BFGS-B",
            options={"maxiter": 200, "gtol": 1e-5, "ftol": 1e-9},
        )
        cand = np.sign(r2.x)
        cand[cand == 0] = 1.0
        if time.time() < DEADLINE - 5:
            cand, cv = _local_search(cand, budget_iters=2000)
        else:
            cv = _score(cand)
        if cv < best_val:
            best_s, best_val = cand, cv

    return best_s


# Expose score for the evaluator interface (kept as attribute per seed convention)
def _score_fn(s):
    return _score(s)

construct.score = _score_fn
