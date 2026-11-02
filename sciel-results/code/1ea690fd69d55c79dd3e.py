"""Parametric Gaussian-basis design for the second autocorrelation inequality.

Model the 50 nonnegative step heights as the pointwise maximum of a small set
of shifted, scaled Gaussian bumps. Centers, variances and amplitudes are
continuous parameters, which enforces non-negativity and smoothness and lets a
gradient-free optimizer navigate the shape space. We then run a local
amplitude-allocation refinement on the resulting support.
"""
import time
import math
import random
import numpy as np

N = 50
# equal-interval grid on [-1/4, 1/4]
X = np.linspace(-0.25, 0.25, N)
DX = X[1] - X[0]

# deadline guard: evaluator kills at ~150s wall-clock; keep a safety margin
_START = time.time()
DEADLINE = 150.0
MARGIN = 15.0  # seconds to stop before the hard limit


def _elapsed():
    return time.time() - _START


def _budget_left():
    return DEADLINE - _MARGIN - _elapsed()


def _ratio(h):
    """score = ||h||_2^2 / (||h||_1 * ||h*h||_inf)  (step heights, unit interval)."""
    h = np.asarray(h, dtype=float)
    l1 = h.sum()
    if l1 <= 0:
        return 0.0
    l2sq = float((h * h).sum())
    # autoconvolution on the grid (step heights on unit intervals)
    conv = np.convolve(h, h, mode='full')
    sup = conv.max()
    if sup <= 0:
        return 0.0
    return l2sq / (l1 * sup)


def _gaussian_bump(center, width, amp, x):
    """Single Gaussian bump evaluated on the grid."""
    if width <= 1e-8:
        width = 1e-8
    return amp * np.exp(-0.5 * ((x - center) / width) ** 2)


def _params_to_heights(params):
    """params = [c0,w0,a0, c1,w1,a1, ...] -> pointwise-max nonnegative heights."""
    k = len(params) // 3
    h = np.zeros(N)
    for i in range(k):
        c = params[3 * i]
        w = params[3 * i + 1]
        a = params[3 * i + 2]
        if a < 0:
            a = 0.0
        h = np.maximum(h, _gaussian_bump(c, w, a, X))
    return h


def _random_params(k, rng):
    """Random Gaussian parameters within the grid domain."""
    p = []
    for _ in range(k):
        c = rng.uniform(-0.25, 0.25)
        w = rng.uniform(0.01, 0.12)
        a = rng.uniform(0.3, 1.5)
        p.extend([c, w, a])
    return p


def _nelder_mead_simplex(f, x0, max_iter, tol):
    """Minimal Nelder-Mead implementation (no scipy dependency required)."""
    n = len(x0)
    # initial simplex
    simplex = [np.array(x0, dtype=float)]
    for i in range(n):
        pt = np.array(x0, dtype=float)
        step = 0.05 if x0[i] == 0 else 0.05 * abs(x0[i])
        pt[i] += step
        simplex.append(pt)

    fvals = [f(p) for p in simplex]
    order = np.argsort(fvals)
    simplex = [simplex[i] for i in order]
    fvals = [fvals[i] for i in order]

    alpha, gamma, rho = 1.0, 2.0, 0.5

    for _ in range(max_iter):
        centroid = np.mean(simplex[:-1], axis=0)
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = f(xr)
        if fr < fvals[0]:
            xe = centroid + gamma * (xr - centroid)
            fe = f(xe)
            if fe < fr:
                simplex[-1] = xe
                fvals[-1] = fe
            else:
                simplex[-1] = xr
                fvals[-1] = fr
        elif fr < fvals[-2]:
            simplex[-1] = xr
            fvals[-1] = fr
        else:
            xc = centroid + rho * (simplex[-1] - centroid)
            fc = f(xc)
            if fc < fvals[-1]:
                simplex[-1] = xc
                fvals[-1] = fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = 0.5 * (simplex[i] + simplex[0])
                    fvals[i] = f(simplex[i])
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        if abs(fvals[-1] - fvals[0]) < tol:
            break
    return simplex[0], fvals[0]


def _optimize_gaussian(k, x0, max_iter, tol):
    """Optimize Gaussian params to maximize the ratio."""
    def neg_ratio(p):
        h = _params_to_heights(p)
        return -_ratio(h)
    p_opt, _ = _nelder_mead_simplex(neg_ratio, x0, max_iter, tol)
    return _params_to_heights(p_opt)


def _refine_amplitudes(h0, iters=400, lr=0.02):
    """Local amplitude allocation refinement (nonnegative) on a fixed support."""
    h = np.array(h0, dtype=float)
    # only refine where there is support
    support = h > 1e-6
    best = h.copy()
    best_score = _ratio(best)
    for _ in range(iters):
        if _budget_left() < 5.0:
            break
        s = _ratio(best)
        # gradient-free coordinate / subgradient step on amplitudes
        # use a simple projected gradient on the ratio
        l1 = best.sum()
        conv = np.convolve(best, best, mode='full')
        sup = conv.max()
        idx = np.argmax(conv)
        # move mass off the peak of the convolution
        grad = np.zeros(N)
        # d/dh of ||h||_2^2/(||h||_1*sup): approximate via finite difference
        # use a random perturbation search (cheap)
        for _ in range(8):
            cand = best.copy()
            m = rng_support.choice(np.where(support)[0])
            delta = lr * (0.5 - random.random())
            cand[m] = max(0.0, cand[m] + delta)
            sc = _ratio(cand)
            if sc > best_score:
                best = cand
                best_score = sc
                break
        else:
            # no improvement this round; shrink step
            lr = max(0.001, lr * 0.85)
    return best, best_score


rng_support = np.random.default_rng(0)
random.seed(0)


def construct():
    global _START
    _START = time.time()

    best_h = np.ones(N)
    best_score = _ratio(best_h)

    # diverse initializations: equal, geometric, random, alternating
    inits = []
    inits.append(np.ones(N))
    geo = np.geomspace(0.2, 1.0, N)
    inits.append(geo)
    inits.append(geo[::-1])
    alt = np.where(np.arange(N) % 2 == 0, 1.0, 0.3)
    inits.append(alt)
    inits.append(alt[::-1])
    rng = np.random.default_rng(1)
    inits.append(rng.uniform(0.2, 1.0, N))

    # Parametric Gaussian search: try a few bump counts
    for k in (1, 2, 3):
        if _budget_left() < 10.0:
            break
        for start in range(3):
            if _budget_left() < 10.0:
                break
            p0 = _random_params(k, np.random.default_rng(100 + k * 10 + start))
            h = _optimize_gaussian(k, p0, max_iter=200, tol=1e-6)
            sc = _ratio(h)
            if sc > best_score:
                best_score = sc
                best_h = h.copy()

    # Also evaluate the discrete initializations directly
    for h0 in inits:
        if _budget_left() < 5.0:
            break
        sc = _ratio(h0)
        if sc > best_score:
            best_score = sc
            best_h = h0.copy()

    # Final local amplitude-allocation refinement on the best support
    if _budget_left() > 10.0:
        refined, sc = _refine_amplitudes(best_h, iters=300, lr=0.02)
        if sc > best_score:
            best_score = sc
            best_h = refined

    # Scale to minimum amplitude = 1 (exploits 1/alpha scaling of the ratio).
    # The ratio is scale-invariant: scaling all heights by alpha leaves the
    # ratio unchanged. We normalize so the smallest positive height is 1.0.
    pos = best_h[best_h > 1e-9]
    if len(pos) > 0:
        scale = 1.0 / pos.min()
        best_h = best_h * scale

    return list(best_h)


# EVOLVE-BLOCK-START
def construct():
    global _START
    _START = time.time()

    best_h = np.ones(N)
    best_score = _ratio(best_h)

    # diverse initializations: equal, geometric, random, alternating
    inits = []
    inits.append(np.ones(N))
    geo = np.geomspace(0.2, 1.0, N)
    inits.append(geo)
    inits.append(geo[::-1])
    alt = np.where(np.arange(N) % 2 == 0, 1.0, 0.3)
    inits.append(alt)
    inits.append(alt[::-1])
    rng = np.random.default_rng(1)
    inits.append(rng.uniform(0.2, 1.0, N))

    # Parametric Gaussian search: try a few bump counts
    for k in (1, 2, 3):
        if _budget_left() < 10.0:
            break
        for start in range(3):
            if _budget_left() < 10.0:
                break
            p0 = _random_params(k, np.random.default_rng(100 + k * 10 + start))
            h = _optimize_gaussian(k, p0, max_iter=200, tol=1e-6)
            sc = _ratio(h)
            if sc > best_score:
                best_score = sc
                best_h = h.copy()

    # Also evaluate the discrete initializations directly
    for h0 in inits:
        if _budget_left() < 5.0:
            break
        sc = _ratio(h0)
        if sc > best_score:
            best_score = sc
            best_h = h0.copy()

    # Final local amplitude-allocation refinement on the best support
    if _budget_left() > 10.0:
        refined, sc = _refine_amplitudes(best_h, iters=300, lr=0.02)
        if sc > best_score:
            best_score = sc
            best_h = refined

    # Scale to minimum amplitude = 1 (exploits 1/alpha scaling of the ratio).
    pos = best_h[best_h > 1e-9]
    if len(pos) > 0:
        scale = 1.0 / pos.min()
        best_h = best_h * scale

    return list(best_h)
# EVOLVE-BLOCK-END


if __name__ == "__main__":
    h = construct()
    print("score:", _ratio(h))
    print("len:", len(h))
    print("min:", min(h), "max:", max(h))
