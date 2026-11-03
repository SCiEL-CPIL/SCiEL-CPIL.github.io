"""Rewritten program: min-max overlap via smooth-max optimization + coordinate descent."""
import time
import numpy as np

N = 100

_DEADLINE = 145.0  # safety margin before 150s kill
_START = None


def _overlap_max(h):
    """Return the maximum over shifts of sum_i h_i*(1-h_{i-k}) for valid i."""
    n = len(h)
    best = 0.0
    for k in range(1, n):
        # overlap for shift k: sum over i where both i and i-k in range
        lo = k
        hi = n
        a = h[lo:hi]
        b = h[0:n - k]
        val = np.sum(a * (1.0 - b))
        if val > best:
            best = val
    return best


def _smooth_max(h, t=0.05):
    """Smooth approximation of max overlap, differentiable."""
    n = len(h)
    vals = []
    for k in range(1, n):
        a = h[k:n]
        b = h[0:n - k]
        vals.append(np.sum(a * (1.0 - b)))
    vals = np.array(vals)
    m = vals.max()
    return m + t * np.log(np.sum(np.exp((vals - m) / t)))


def _project(h):
    """Project onto [0,1] box and fix sum to N/2."""
    h = np.clip(h, 0.0, 1.0)
    s = h.sum()
    target = N / 2.0
    if s > 0:
        h = h + (target - s) / N
    return np.clip(h, 0.0, 1.0)


def _score(h):
    """Upper bound on C5 = max overlap value (lower is better)."""
    return float(_overlap_max(h))


def _initial_pattern():
    """Structured starting point: alternating high/low to decorrelate shifts."""
    h = np.empty(N)
    # try a few structured patterns and pick best
    candidates = []
    # alternating 0.9/0.1
    a = np.where(np.arange(N) % 2 == 0, 0.9, 0.1)
    candidates.append(a.copy())
    # period-3 pattern
    b = np.tile([0.9, 0.1, 0.5], N // 3 + 1)[:N]
    candidates.append(b.copy())
    # period-4
    c = np.tile([1.0, 0.0, 0.0, 1.0], N // 4 + 1)[:N]
    candidates.append(c.copy())
    # random
    rng = np.random.default_rng(0)
    for _ in range(20):
        r = rng.random(N) * 1.0
        candidates.append(r)
    best = None
    bestv = 1e9
    for cand in candidates:
        cand = _project(cand)
        v = _score(cand)
        if v < bestv:
            bestv = v
            best = cand
    return best, bestv


def construct():
    global _START
    if _START is None:
        _START = time.time()

    h, v = _initial_pattern()

    # Coordinate descent with random restarts under deadline
    rng = np.random.default_rng(1)
    best_h = h.copy()
    best_v = v

    def try_coord(h0, iters):
        h = h0.copy()
        v = _score(h)
        step = 0.1
        for it in range(iters):
            if time.time() > _DEADLINE:
                break
            # pick random coordinate
            i = rng.integers(0, N)
            # try a few perturbations
            improved = False
            for delta in (step, -step, step * 0.5, -step * 0.5, step * 2.0, -step * 2.0):
                h_new = h.copy()
                h_new[i] = np.clip(h[i] + delta, 0.0, 1.0)
                # fix sum
                s = h_new.sum()
                h_new = h_new + (N / 2.0 - s) / N
                h_new = np.clip(h_new, 0.0, 1.0)
                vn = _score(h_new)
                if vn < v - 1e-12:
                    h = h_new
                    v = vn
                    improved = True
                    break
            if not improved:
                step *= 0.7
                if step < 1e-4:
                    break
        return h, v

    # main descent
    h, v = try_coord(h, 2000)
    if v < best_v:
        best_h, best_v = h, v

    # several random restarts
    restarts = 0
    while time.time() < _DEADLINE - 5.0:
        restarts += 1
        r = rng.random(N)
        r = _project(r)
        # blend with best to keep near good region sometimes
        if restarts % 3 == 0:
            r = 0.5 * r + 0.5 * best_h
            r = _project(r)
        h, v = try_coord(r, 800)
        if v < best_v:
            best_h, best_v = h, v
            step = 0.1

    # final polish on best
    h, v = try_coord(best_h, 1500)
    if v < best_v:
        best_h, best_v = h, v

    # ensure exact sum
    best_h = np.clip(best_h, 0.0, 1.0)
    s = best_h.sum()
    best_h = best_h + (N / 2.0 - s) / N
    best_h = np.clip(best_h, 0.0, 1.0)

    return best_h
