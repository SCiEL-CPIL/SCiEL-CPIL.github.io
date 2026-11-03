"""Rewritten program: smooth local search for the second autocorrelation inequality.

Maximizes  C2 = ||h||_2^2 / (||h||_1 * ||h*h||_inf)
for 50 nonnegative step heights.

Strategy:
  * Scale invariance -> fix ||h||_2 = 1 and minimize  F = ||h||_1 * ||h*h||_inf.
  * The autoconvolution peak is largest for concentrated (Gaussian-like) shapes,
    which also minimize ||h||_1 for fixed ||h||_2.  Start from a Gaussian.
  * Refine with a projected gradient / subgradient method on the simplex of
    nonnegative vectors, using the subgradients of ||h||_1 and of max(h*h).
"""
import time
import numpy as np

N = 50

# ---- fixed geometry -------------------------------------------------------
# equal intervals of [-1/4, 1/4]  ->  50 cells of width 1/50
_dx = 1.0 / N
# convolution index offset: (h*h)[k] = sum_i h_i h_{i+k}
# k ranges from -(N-1) .. (N-1)
_offsets = np.arange(-(N - 1), N)


def _autoconv_peak(h):
    """Return (peak_value, argmax_index) of the linear autoconvolution h*h."""
    # full linear convolution of h with itself
    c = np.convolve(h, h, mode="full")          # length 2N-1, index k+N-1
    j = int(np.argmax(c))
    return float(c[j]), j


def _score(h):
    """C2 = ||h||_2^2 / (||h||_1 * ||h*h||_inf)."""
    h = np.asarray(h, dtype=float)
    n2 = float(np.dot(h, h))
    if n2 <= 1e-300:
        return 0.0
    n1 = float(np.sum(h))
    peak, _ = _autoconv_peak(h)
    if peak <= 1e-300 or n1 <= 1e-300:
        return 0.0
    return n2 / (n1 * peak)


def _gaussian(alpha):
    """Gaussian-like nonnegative vector on the grid (not normalized)."""
    x = (np.arange(N) - (N - 1) / 2.0) * _dx          # centered on grid
    v = np.exp(-alpha * x * x)
    return v


def _project_nonneg(h):
    return np.maximum(h, 0.0)


def _local_search(h0, deadline, iters):
    """Projected subgradient refinement on nonnegative vectors.

    Minimizes F(h) = ||h||_1 * ||h*h||_inf  (with h>=0), scale-invariant.
    Uses a simple adaptive step-size projected (sub)gradient method.
    """
    h = np.array(h0, dtype=float)
    if np.any(h < 0):
        h = _project_nonneg(h)
    best = h.copy()
    best_F = float("inf")

    # initial peak subgradient
    peak, j = _autoconv_peak(h)
    k = j - (N - 1)
    # subgradient of ||h*h||_inf w.r.t. h:  g_i = h_{i-k} + h_{i+k} (valid indices)
    def peak_subgrad(kk):
        g = np.zeros(N)
        for i in range(N):
            a = i - kk
            b = i + kk
            if 0 <= a < N:
                g[i] += h[a]
            if 0 <= b < N:
                g[i] += h[b]
        return g

    g_peak = peak_subgrad(k)
    g_l1 = np.sign(h)
    g_l1[h == 0.0] = 0.0
    F = float(np.sum(h) * peak)
    best_F = F

    step = 1e-3
    min_step = 1e-7
    failed = 0
    t0 = time.time()

    for it in range(iters):
        if time.time() > deadline:
            break
        # gradient of F = n1 * peak:  grad = peak * g_l1 + n1 * g_peak
        n1 = float(np.sum(h))
        grad = peak * g_l1 + n1 * g_peak

        # project gradient step
        cand = _project_nonneg(h - step * grad)
        # renormalize to keep ||.||_2 ~ 1 (scale invariance; keeps numbers stable)
        n2 = float(np.dot(cand, cand))
        if n2 <= 1e-300:
            break
        cand = cand / np.sqrt(n2)

        cand_peak, cand_j = _autoconv_peak(cand)
        cand_F = float(np.sum(cand) * cand_peak)

        if cand_F < F - 1e-12:
            # accept
            h = cand
            F = cand_F
            failed = 0
            # recompute subgradients
            peak, j = cand_peak, cand_j
            k = j - (N - 1)
            g_peak = peak_subgrad(k)
            g_l1 = np.sign(h)
            g_l1[h == 0.0] = 0.0
            step = min(step * 1.05, 1e-2)
            if F < best_F:
                best_F = F
                best = h.copy()
        else:
            failed += 1
            step = max(step * 0.8, min_step)
            if failed > 200:
                # restart with a smaller random perturbation to escape
                failed = 0
                step = 1e-3
                h = _project_nonneg(h + 1e-4 * np.random.randn(N))
                n2 = float(np.dot(h, h))
                if n2 <= 1e-300:
                    break
                h = h / np.sqrt(n2)
                peak, j = _autoconv_peak(h)
                k = j - (N - 1)
                g_peak = peak_subgrad(k)
                g_l1 = np.sign(h)
                g_l1[h == 0.0] = 0.0
                F = float(np.sum(h) * peak)

    return best, best_F


def construct():
    """Return 50 nonnegative step heights maximizing C2.

    Runs a time-budgeted multi-restart smooth local search from Gaussian-like
    seeds (the shape family that minimizes the autoconvolution peak for a given
    spread), then returns the best vector found.
    """
    deadline = time.time() + 145.0     # safety margin under 150s wall-clock
    rng = np.random.default_rng(12345)

    best_h = None
    best_val = -1.0

    # candidate seeds: a family of Gaussian widths + a few flat/plateau shapes
    alphas = [20.0, 30.0, 40.0, 60.0, 80.0, 120.0, 200.0]
    # also a couple of slightly randomized starts
    starts = []
    for a in alphas:
        v = _gaussian(a)
        starts.append(v)
    for _ in range(4):
        a = rng.uniform(15.0, 150.0)
        v = _gaussian(a)
        v = v * (1.0 + 0.05 * rng.standard_normal(N))
        v = _project_nonneg(v)
        starts.append(v)

    # normalize each start to ||.||_2 = 1
    for idx, s in enumerate(starts):
        if time.time() > deadline:
            break
        n2 = float(np.dot(s, s))
        if n2 <= 1e-300:
            continue
        s = s / np.sqrt(n2)
        # per-restart time budget: spend more on promising (low F) starts
        _, F0 = _autoconv_peak(s)
        F0 = float(np.sum(s) * F0)
        # give each restart a slice of the remaining time
        remaining = max(0.0, deadline - time.time())
        # first few restarts get more time
        per = max(1.0, remaining / max(1, len(starts) - idx))
        dl = time.time() + min(per, remaining)
        iters = int(max(50, min(4000, per * 2000)))
        h, F = _local_search(s, dl, iters)
        val = _score(h)
        if val > best_val:
            best_val = val
            best_h = h.copy()

    if best_h is None:
        # fallback: uniform
        best_h = np.ones(N)
        best_val = _score(best_h)

    return [float(x) for x in best_h]


if __name__ == "__main__":
    t = time.time()
    res = construct()
    print("N =", len(res))
    print("score C2 =", _score(res))
    print("elapsed =", time.time() - t)
