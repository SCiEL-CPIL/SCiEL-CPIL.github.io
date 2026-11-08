"""Second autocorrelation inequality – optimized solver.

Maximizes R = ||h||_2^2 / (||h||_1 * ||h*h||_inf) over 50 nonnegative step heights.
"""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 50
DT = 0.5 / N
TIME_LIMIT = 115.0  # safety margin below 120s / 150s


def compute_ratio(h):
    """Compute R = ||h||_2^2 / (||h||_1 * ||h*h||_inf)."""
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0.0:
        return 0.0
    l2_sq = np.sum(h * h)

    # Discrete autocorrelation via FFT
    size = 1 << (2 * N - 1).bit_length()
    H = np.fft.rfft(h, n=size)
    ac = np.fft.irfft(H * np.conj(H), n=size)[:2 * N - 1]
    # ac[k] = sum_i h[i] h[i-k+N-1] for k=0..2N-2  (circular, but zero-padded so linear)
    # The true linear autocorrelation lags: for lag m = k-(N-1), sum_i h[i]h[i-m]
    # Max over all lags is just max(ac) since it covers all linear lags.
    max_auto = np.max(ac)
    if max_auto <= 0.0:
        return 0.0
    conv_max = DT * max_auto
    return l2_sq / (l1 * conv_max)


def neg_ratio(h):
    return -compute_ratio(h)


def _ratio_safe(h):
    """Safe wrapper for compute_ratio used in optimization."""
    h = np.asarray(h, dtype=np.float64)
    if np.any(h < -1e-12) or np.any(np.isnan(h)):
        return 0.0
    return compute_ratio(np.clip(h, 0, None))


def _try_candidate(h, best_h, best_score):
    """Evaluate a candidate and update best if improved."""
    h = np.clip(np.asarray(h, dtype=np.float64), 0, None)
    if np.all(h == 0):
        return best_h, best_score
    s = compute_ratio(h)
    if s > best_score:
        return h.copy(), s
    return best_h, best_score


def construct():
    start_time = time.time()
    best_h = np.ones(N)
    best_score = compute_ratio(best_h)

    # ---- Phase 1: Analytical / structured initial candidates ----
    candidates = []

    # Single spike at various positions
    for pos in [0, N // 2, N - 1]:
        v = np.zeros(N)
        v[pos] = 1.0
        candidates.append(v)

    # Two spikes: far apart (minimizes overlap in autocorrelation)
    for gap in [N // 2, N // 4, 3 * N // 4]:
        v = np.zeros(N)
        v[0] = 1.0
        v[gap % N] = 1.0
        candidates.append(v)

    # Ramp: h[i] = i+1
    candidates.append(np.arange(1, N + 1, dtype=np.float64))

    # Cosine bump (concentrated in middle)
    idx = np.arange(N)
    candidates.append(1.0 + np.cos(2 * np.pi * idx / N))

    # Exponential decay
    candidates.append(np.exp(-np.arange(N) / 10.0))

    # Block: first half ones, second half zeros (and vice versa)
    v1 = np.zeros(N); v1[:N // 2] = 1.0; candidates.append(v1)
    v2 = np.zeros(N); v2[N // 2:] = 1.0; candidates.append(v2)

    # Sparse random: pick k random positions, set to 1
    for k in [2, 3, 5, 10]:
        pos = np.random.choice(N, size=k, replace=False)
        v = np.zeros(N); v[pos] = 1.0
        candidates.append(v)

    for c in candidates:
        best_h, best_score = _try_candidate(c, best_h, best_score)
        if time.time() - start_time > TIME_LIMIT - 30:
            break

    # ---- Phase 2: Differential evolution (global search) ----
    if time.time() - start_time < TIME_LIMIT - 40:
        try:
            bounds = [(0.0, 1.0)] * N
            de_res = differential_evolution(
                neg_ratio, bounds,
                maxiter=30, popsize=15, tol=1e-6,
                seed=42, mutation=(0.3, 1.0), recombination=0.8,
                polishing=False, update_interval=1,
                callback=lambda xk, convergence: (time.time() - start_time > TIME_LIMIT - 35)
            )
            if hasattr(de_res, 'x') and de_res.x is not None:
                best_h, best_score = _try_candidate(de_res.x, best_h, best_score)
        except Exception:
            pass

    # ---- Phase 3: Local refinement from multiple starts ----
    # Gather diverse starting points
    starts = [best_h.copy()]
    # Add structured starts
    starts.append(np.ones(N) * 0.5)
    starts.append(np.random.RandomState(1).random(N))
    starts.append(np.random.RandomState(2).random(N))
    # Sparse starts
    for seed in range(3):
        rs = np.random.RandomState(seed)
        v = np.zeros(N)
        v[rs.choice(N, size=3, replace=False)] = 1.0
        starts.append(v)

    # Add DE result if different
    try:
        de_x = np.clip(de_res.x, 0, 1)
        if np.max(np.abs(de_x - best_h)) > 1e-6:
            starts.append(de_x)
    except Exception:
        pass

    for i, h0 in enumerate(starts):
        if time.time() - start_time > TIME_LIMIT - 10:
            break
        h0 = np.clip(h0, 0, None)
        if np.all(h0 == 0):
            h0 = np.ones(N) * 0.1
        try:
            res = minimize(
                neg_ratio, h0, method='L-BFGS-B',
                bounds=[(0.0, None)] * N,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            best_h, best_score = _try_candidate(res.x, best_h, best_score)

            # Second pass: Nelder-Mead from L-BFGS-B result
            if time.time() - start_time < TIME_LIMIT - 5:
                res2 = minimize(
                    neg_ratio, np.clip(res.x, 0, None),
                    method='Nelder-Mead',
                    options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-8,
                             'adaptive': True, 'initial_simplex': None}
                )
                best_h, best_score = _try_candidate(res2.x, best_h, best_score)
        except Exception:
            pass

    # ---- Phase 4: Coordinate-descent polish ----
    if time.time() - start_time < TIME_LIMIT - 5:
        h = np.clip(best_h, 0, None).copy()
        improved = True
        max_cd_iters = 50
        for _ in range(max_cd_iters):
            if time.time() - start_time > TIME_LIMIT - 3:
                break
            if not improved:
                break
            improved = False
            for i in range(N):
                old = h[i]
                # Try a few perturbations
                for scale in [0.5, 0.8, 1.2, 1.5, 2.0, 0.0, 1.0]:
                    h[i] = old * scale
                    s = compute_ratio(h)
                    if s > best_score + 1e-12:
                        best_score = s
                        best_h = h.copy()
                        improved = True
                h[i] = old  # restore if not improved at this coordinate
        # Final check
        best_h, best_score = _try_candidate(best_h, best_h, best_score)

    return [float(x) for x in best_h]
