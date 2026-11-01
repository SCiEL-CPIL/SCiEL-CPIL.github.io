"""Fresh design: tanh-relaxation (L-BFGS-B) + discrete sign-flip hill-climbing,
with an explicit wall-clock deadline guard. Metric: max|conv|/(sum|h|)^2 (lower=better)."""

import time
import numpy as np

N = 400
# deadline guard: return best found with a safety margin before the 150 s kill
DEADLINE = time.time() + 138.0


def _conv(h):
    """Full linear convolution (length 2N-1) of the step heights with themselves."""
    return np.convolve(h, h, mode="full")


def _metric(h):
    """C3 upper bound: max|autoconvolution| / (sum|h|)^2. Lower is better."""
    s = float(np.sum(np.abs(h)))
    if s <= 0.0:
        return np.inf
    return float(np.max(np.abs(_conv(h)))) / (s * s)


def _objective(u):
    """Smooth surrogate objective over the continuous relaxation u -> h = tanh(u)."""
    h = np.tanh(u)
    s = float(np.sum(np.abs(h)))
    if s <= 1e-12:
        return 1e12
    c = _conv(h)
    return float(np.max(np.abs(c))) / (s * s)


def _signed(h):
    """Round the continuous relaxation to a signed step function (each height +/-1)."""
    return np.where(h >= 0.0, 1.0, -1.0)


def _hill_climb(h, time_budget=1.0):
    """Discrete sign-flip hill-climbing to escape local minima of the metric."""
    h = h.copy()
    best = _metric(h)
    order = np.random.permutation(N)
    start = time.time()
    improved = True
    while improved and (time.time() - start) < time_budget:
        improved = False
        for i in order:
            h[i] = -h[i]
            v = _metric(h)
            if v < best - 1e-15:
                best = v
                improved = True
            else:
                h[i] = -h[i]  # revert
    return h, best


def construct():
    best_h = np.ones(N)
    best_score = _metric(best_h)

    rng = np.random.default_rng(12345)

    # ---- Phase 1: continuous tanh-relaxation from diverse structured starts ----
    starts = [
        np.ones(N),                                   # all positive
        np.ones(N) * np.sign(np.arange(N) - N / 2),   # smooth sign ramp
    ]
    for k in range(4):
        # random block pattern
        blk = rng.integers(4, 16, size=N)
        starts.append(np.where(blk % 2 == 0, 1.0, -1.0))
    # alternating signs
    starts.append(np.where(np.arange(N) % 2 == 0, 1.0, -1.0))

    for base in starts:
        if time.time() > DEADLINE:
            break
        u0 = np.arctanh(np.clip(base, -0.99, 0.99))
        from scipy.optimize import minimize
        try:
            res = minimize(_objective, u0, method="L-BFGS-B",
                           options={"maxiter": 150, "maxls": 40, "maxfun": 20000})
        except Exception:
            continue
        h = _signed(res.x)
        v = _metric(h)
        if v < best_score:
            best_score = v
            best_h = h.copy()

    # ---- Phase 2: discrete sign-flip hill-climbing from best + a few random seeds ----
    candidates = [best_h.copy()]
    for _ in range(4):
        candidates.append(_signed(rng.choice([-1.0, 1.0], size=N)))
    for cand in candidates:
        if time.time() > DEADLINE:
            break
        h2, v2 = _hill_climb(cand, time_budget=1.5)
        if v2 < best_score:
            best_score = v2
            best_h = h2.copy()

    return best_h


if __name__ == "__main__":
    t0 = time.time()
    r = construct()
    print("metric:", _metric(r), "elapsed:", time.time() - t0)
