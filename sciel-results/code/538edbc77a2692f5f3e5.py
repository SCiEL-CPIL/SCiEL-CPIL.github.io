"""
Strategy: Coordinate Descent on the 400-element vector, guided by a smooth
max-plus surrogate (log-sum-exp) to make the objective differentiable and
stable, plus periodic small random perturbations to escape local minima.

Key idea:
- The objective is  f(h) = |2*N*max(conv(h))/sum(h)^2|
- max(conv) is non-differentiable and has many flat regions, causing Powell
  to stall early. Instead, we use coordinate descent with a smooth surrogate
  for the max:  S = logsumexp(conv / T) * T  (temperature T), and
  minimize the surrogate while periodically recomputing the true f to
  track the best.
- We also exploit scale invariance: the objective is invariant under h -> a*h,
  so we can periodically rescale h to keep the sum near 1, improving
  numerical stability.
- Coordinate updates: for each index i, we perform a 1-D golden-section
  search on h[i] while keeping the rest fixed. This is fast (O(N) per eval)
  and avoids the overhead of full-vector Powell iterations.
- To escape local minima, every K iterations we add a small Gaussian
  perturbation to a random subset of coordinates.

This is a structural change from the current program's Powell-on-full-vector
approach: we replace it with an explicit coordinate-descent loop that
interleaves surrogate-guided 1-D minimization and periodic random kicks.
"""

import numpy as np
import time

N = 400
_DEADLINE = 145.0

def _evaluate(h):
    """Compute the true task value."""
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return np.inf
    conv = np.convolve(h, h)
    m = np.max(conv)
    return abs(2.0 * N * m / (s * s))

def _surrogate_eval(h, T):
    """Compute a smooth surrogate for the objective using log-sum-exp."""
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return 1e10
    conv = np.convolve(h, h)
    # log-sum-exp for numerical stability
    cmax = np.max(conv)
    if not np.isfinite(cmax):
        return 1e10
    logsumexp_val = cmax + T * np.log(np.sum(np.exp((conv - cmax) / T)))
    surrogate_peak = logsumexp_val / T  # approximates max(conv)
    return abs(2.0 * N * surrogate_peak / (s * s))

def _golden_section_1d(f, a, b, xtol=1e-6, max_iter=50):
    """Golden-section search for 1-D minimization."""
    gr = (np.sqrt(5) - 1) / 2  # golden ratio ~0.618
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc = f(c)
    fd = f(d)
    for _ in range(max_iter):
        if b - a < xtol:
            break
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - gr * (b - a)
            fc = f(c)
        else:
            a = c
            c = d
            fc = fd
            d = a + gr * (b - a)
            fd = f(d)
    return (a + b) / 2.0

def construct():
    t0 = time.time()
    deadline = t0 + _DEADLINE

    # Initialize with a good starting point: a raised-cosine windowed chirp
    n = np.arange(N)
    center = (N - 1) / 2.0
    # Chirp with quadratic phase
    phase = np.pi * n * (n + 1) / 9.0
    base = np.cos(phase)
    # Raised-cosine window
    w = 0.5 * (1 + np.cos(2 * np.pi * (n - center) / N))
    h = 1.0 + w * base  # bias of 1 to keep sum positive
    h = h - np.mean(h) + 1.0  # ensure positive mean

    best_h = h.copy()
    best_val = _evaluate(best_h)

    # Temperature for surrogate
    T = 1.0
    # Scale h so sum is ~1 for stability
    h = h / np.sum(h) if np.sum(h) != 0 else h

    rng = np.random.default_rng(42)

    total_iters = 0
    max_total_iters = 50000
    kick_every = 200

    while time.time() < deadline - 5 and total_iters < max_total_iters:
        total_iters += 1

        # Periodically kick a random subset of coordinates
        if total_iters % kick_every == 0:
            kick_frac = 0.1
            n_kick = max(1, int(kick_frac * N))
            idx = rng.choice(N, size=n_kick, replace=False)
            h[idx] += 0.1 * rng.standard_normal(n_kick)
            # Rescale to keep sum ~1
            s = np.sum(h)
            if s != 0 and np.isfinite(s):
                h = h / s

        # Coordinate descent: one pass over all 400 coordinates
        for i in range(N):
            if time.time() > deadline - 5:
                break

            # Save old value
            old = h[i]

            # Define 1-D function
            def _f1d(x):
                h[i] = x
                return _surrogate_eval(h, T)

            # Search range around current value
            # Estimate scale
            scale = max(1e-3, 0.1 * abs(old) + 0.01)
            a = old - 5 * scale
            b = old + 5 * scale

            # Golden section search
            new_val = _golden_section_1d(_f1d, a, b, xtol=1e-5, max_iter=30)
            h[i] = new_val

        # After a full pass, check true objective
        val = _evaluate(h)
        if np.isfinite(val) and val < best_val:
            best_val = val
            best_h = h.copy()

        # Gradually reduce temperature
        T = max(0.1, T * 0.995)

    # Final check
    final_val = _evaluate(best_h)
    if not np.isfinite(final_val) or final_val >= 2.0:
        return np.ones(N)

    return best_h
