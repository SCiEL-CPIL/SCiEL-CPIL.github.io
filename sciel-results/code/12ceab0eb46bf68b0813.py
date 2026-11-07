"""Erdos minimum overlap: Fourier-parametrized search + smooth L-BFGS-B + direct polish."""
import numpy as np
import time

N = 100
TARGET = N / 2.0
DEADLINE = 140.0          # safety margin below the 150s wall-clock kill
START = time.time()


def remaining():
    return DEADLINE - (time.time() - START)


def objective(h):
    """Raw evaluator value (lower is better)."""
    c = np.correlate(h, 1.0 - h, mode="full")
    return float(np.max(c)) / N * 2.0


def objective_smooth(h, beta=60.0):
    """Smooth surrogate of max(c) via log-sum-exp; differentiable for L-BFGS-B."""
    c = np.correlate(h, 1.0 - h, mode="full")
    c_max = np.max(c)
    # log-sum-exp with shift for stability
    s = np.sum(np.exp(beta * (c - c_max)))
    soft = c_max + (1.0 / beta) * np.log(s)
    return soft / N * 2.0


def project(h):
    """Clip to [0,1] and fix the sum to N/2 (alternating clip + shift)."""
    h = np.clip(h, 0.0, 1.0)
    for _ in range(200):
        s = h.sum()
        if abs(s - TARGET) < 1e-13:
            break
        h = h + (TARGET - s) / N
        h = np.clip(h, 0.0, 1.0)
    # final nudge to meet the 1e-9 * N tolerance
    s = h.sum()
    if abs(s - TARGET) > 1e-12:
        h = h + (TARGET - s) / N
        h = np.clip(h, 0.0, 1.0)
    return h


def fourier_h(theta):
    """theta = [a0, (f1,p1,a1), (f2,p2,a2), ...] -> h vector in R^N."""
    i = np.arange(N)
    h = 0.5 + theta[0]
    idx = 1
    while idx + 3 <= len(theta):
        f, p, a = theta[idx], theta[idx + 1], theta[idx + 2]
        h = h + a * np.sin(2.0 * np.pi * f * i / N + p)
        idx += 3
    return h


def optimize_fourier(theta0, max_iter=4000, beta=60.0):
    from scipy.optimize import minimize

    def obj(x):
        h = project(fourier_h(x))
        return objective_smooth(h, beta=beta)

    try:
        res = minimize(
            obj, theta0, method="L-BFGS-B",
            options={"maxiter": max_iter, "ftol": 1e-13, "gtol": 1e-9, "maxls": 50},
        )
        return res.x
    except Exception:
        return theta0


def main_search():
    best_h = np.full(N, 0.5)
    best_val = objective(best_h)  # 1.0

    rng = np.random.default_rng(0)
    n_harm = 12
    dim = 1 + 3 * n_harm
    i = np.arange(N)

    for trial in range(120):
        if remaining() < 12.0:
            break

        # build a random Fourier seed
        theta0 = np.zeros(dim)
        theta0[0] = rng.uniform(-0.05, 0.05)
        for k in range(n_harm):
            theta0[1 + 3 * k] = float(rng.integers(1, N // 2 + 1))   # frequency
            theta0[2 + 3 * k] = rng.uniform(0.0, 2.0 * np.pi)        # phase
            theta0[3 + 3 * k] = rng.uniform(-0.4, 0.4)               # amplitude

        theta_opt = optimize_fourier(theta0, max_iter=3000, beta=60.0)
        h = project(fourier_h(theta_opt))
        v = objective(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()

        # occasionally re-run the same optimum with a sharper beta to refine
        if v < best_val + 1e-6 and remaining() > 20.0:
            theta_opt2 = optimize_fourier(theta_opt, max_iter=1500, beta=150.0)
            h2 = project(fourier_h(theta_opt2))
            v2 = objective(h2)
            if v2 < best_val:
                best_val = v2
                best_h = h2.copy()

    # Direct-space local polish with decaying Gaussian steps
    step = 0.05
    for it in range(3000):
        if remaining() < 5.0:
            break
        step = max(0.001, 0.05 * (1.0 - it / 6000.0))
        cand = best_h + rng.normal(0.0, step, N)
        cand = project(cand)
        v = objective(cand)
        if v < best_val:
            best_val = v
            best_h = cand

    return best_h, best_val


# EVOLVE-BLOCK-START
def construct():
    global _CACHED
    try:
        if '_CACHED' in globals():
            return _CACHED
        h, _ = main_search()
        h = project(h)
        # enforce sum within tolerance
        s = h.sum()
        h = h + (TARGET - s) / N
        h = np.clip(h, 0.0, 1.0)
        assert np.all(np.isfinite(h))
        _CACHED = h
        return h
    except Exception:
        return np.full(N, 0.5)
# EVOLVE-BLOCK-END
