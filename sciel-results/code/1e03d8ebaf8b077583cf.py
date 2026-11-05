"""
Third autocorrelation inequality - structural redesign.

Key insight: The constant sequence gives value = 2.0 (proven lower bound).
However, the formula uses np.max(conv) (signed max, NOT absolute max).
If we can make ALL entries of the autoconvolution negative (or at least
make the maximum significantly less than sum(h^2)), we can potentially
get value < 2.0.

Wait - actually c[N-1] = sum(h[i]^2) is always non-negative, so max(conv) >= 0.
And by Cauchy-Schwarz, sum(h^2) >= (sum(h))^2 / N, so value >= 2.0.

So the minimum possible value is exactly 2.0, achieved by constant sequences.
The current program already achieves 1.0747 which means it's finding value ≈ 1.86.

Wait, let me re-read: score = seed/value. Seed value = 2.0. So score 1.0747 means
value = 2.0/1.0747 ≈ 1.861. That's below 2.0!

How is that possible? Let me re-examine. The proof says value >= 2.0 only if
max(conv) >= c[N-1]. But max(conv) is the SIGNED maximum. c[N-1] = sum(h^2) >= 0.
So max(conv) >= c[N-1] >= S^2/N. Thus value = 2*N*max(conv)/S^2 >= 2*N*(S^2/N)/S^2 = 2.

Unless... S is negative? No, S^2 is the same. Unless max(conv) is negative?
But c[N-1] = sum(h[i]^2) >= 0 always. So max(conv) >= 0.

Hmm, but the score is 1.0747 > 1.0, meaning value < 2.0. Let me check the formula again.

value = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h) ** 2)

For h = ones(400): conv has entries [1,2,3,...,400,399,...,1]. max = 400.
value = 2*400*400/400^2 = 2.0. Correct.

For the score to be 1.0747, we need value = 2/1.0747 ≈ 1.861.

This means 2*N*max(conv)/S^2 ≈ 1.861, so max(conv)/S^2 ≈ 1.861/(2*400) ≈ 0.002326.
For constant: max(conv)/S^2 = 400/160000 = 0.0025.

So we need max(conv)/S^2 < 0.0025. Since max(conv) >= c[N-1] = sum(h^2) >= S^2/N,
we have max(conv)/S^2 >= 1/N = 0.0025. Contradiction!

Unless my understanding of np.convolve is wrong. Let me check: np.convolve(h,h) for
full convolution has length 2N-1. c[k] = sum_i h[i]*h[k-i] for valid indices.
c[N-1] = sum_i h[i]*h[N-1-i]. This is NOT sum(h^2) in general!

I made an error. c[N-1] = sum_i h[i]*h[N-1-i], which is the cross-correlation at lag 0
only if the sequence is symmetric. For general h, c[N-1] != sum(h^2).

So the lower bound of 2.0 may NOT hold! We can potentially get value < 2.0.

Strategy: Use sequences where the autoconvolution peak is spread out, minimizing
the maximum entry relative to S^2.

New approach: Use a smooth, low-frequency signal. A cosine with a small number of
periods will have an autoconvolution that's also a cosine-like shape with a clear peak,
but if we use multiple frequencies or a carefully shaped signal, we can reduce the peak.

Actually, the best strategy might be to use a signal that is approximately orthogonal
to its shifts, like a Zadoff-Chu or CAZAC sequence, but real-valued.

Let me try: h[n] = cos(2*pi*k*n/N) for some integer k. Then the autoconvolution
will have a peak at lag 0 of N/2 (for k>0) plus a DC component.

For h[n] = cos(2*pi*k*n/N):
S = sum cos(2*pi*k*n/N) = 0 for k not multiple of N.
So S = 0, which is invalid!

We need S != 0. So add a DC component: h[n] = a + b*cos(2*pi*k*n/N).
S = N*a (since sum of cosine over full periods is 0).
c[0] = sum h[n]^2 = N*(a^2 + b^2/2)
c[N-1] = sum h[n]*h[N-1-n] = sum (a + b*cos(2*pi*k*n/N))*(a + b*cos(2*pi*k*(N-1-n)/N))
= sum (a + b*cos(2*pi*k*n/N))*(a + b*cos(2*pi*k - 2*pi*k/n... 

This is getting complex. Let me just use numerical optimization with a better parameterization.

New structural approach: Parameterize h as a sum of a few sinusoids with adjustable
amplitudes, frequencies, and phases. Use scipy.optimize with a robust method.
"""

import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 400


def _evaluate(h):
    """Compute the objective value."""
    S = np.sum(h)
    if abs(S) < 1e-15:
        return 1e10
    conv = np.convolve(h, h)
    val = abs(2.0 * N * np.max(conv) / (S * S))
    return val


def construct():
    t0 = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    def _time_remaining():
        return DEADLINE - (time.time() - t0)

    # Start with the known best: constant ones gives 2.0
    best_h = np.ones(N, dtype=np.float64)
    best_val = 2.0

    # Strategy 1: Try sinusoidal parameterization
    # h[n] = a0 + sum_{k=1}^{K} (a_k * cos(2*pi*f_k*n/N + p_k))
    # We'll try a few different K values

    def _sinusoid_heights(params, K):
        """Build h from sinusoidal parameters.
        params = [a0, a1, f1, p1, a2, f2, p2, ..., aK, fK, pK]
        """
        a0 = params[0]
        h = np.full(N, a0, dtype=np.float64)
        n = np.arange(N, dtype=np.float64)
        for k in range(1, K + 1):
            idx = 1 + (k - 1) * 3
            a_k = params[idx]
            f_k = params[idx + 1]
            p_k = params[idx + 2]
            h += a_k * np.cos(2.0 * np.pi * f_k * n / N + p_k)
        return h

    def _obj_sin(params, K):
        if _time_remaining() < 2.0:
            return 1e10
        h = _sinusoid_heights(params, K)
        if not np.all(np.isfinite(h)):
            return 1e10
        return _evaluate(h)

    # Try different numbers of sinusoidal components
    for K in [1, 2, 3]:
        if _time_remaining() < 15.0:
            break

        n_params = 1 + 3 * K
        # Bounds: a0 in [-2,2], a_k in [-2,2], f_k in [0.5, 20], p_k in [0, 2*pi]
        bounds = [(-2.0, 2.0)]
        for _ in range(K):
            bounds.extend([(-2.0, 2.0), (0.5, 20.0), (0.0, 2.0 * np.pi)])

        # Initial guess: small perturbation of constant
        x0 = np.array([1.0] + [0.1, 1.0, 0.0] * K)

        try:
            # First try local optimization
            res = minimize(
                lambda p: _obj_sin(p, K),
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10}
            )
            if res.fun < best_val:
                h_candidate = _sinusoid_heights(res.x, K)
                if np.all(np.isfinite(h_candidate)) and abs(np.sum(h_candidate)) > 1e-12:
                    val = _evaluate(h_candidate)
                    if val < best_val:
                        best_val = val
                        best_h = h_candidate.copy()

        except Exception:
            pass

        if _time_remaining() < 10.0:
            continue

        # Try differential evolution for global search
        try:
            def _obj_de(params):
                if _time_remaining() < 3.0:
                    return 1e10
                return _obj_sin(params, K)

            res_de = differential_evolution(
                _obj_de,
                bounds,
                maxiter=15,
                popsize=8,
                tol=0.01,
                seed=42,
                polish=False,
                init='sobol'
            )
            if res_de.fun < best_val:
                h_candidate = _sinusoid_heights(res_de.x, K)
                if np.all(np.isfinite(h_candidate)) and abs(np.sum(h_candidate)) > 1e-12:
                    val = _evaluate(h_candidate)
                    if val < best_val:
                        best_val = val
                        best_h = h_candidate.copy()

                        # Polish with local optimization
                        if _time_remaining() > 5.0:
                            res_polish = minimize(
                                lambda p: _obj_sin(p, K),
                                res_de.x,
                                method='L-BFGS-B',
                                bounds=bounds,
                                options={'maxiter': 100, 'ftol': 1e-10}
                            )
                            if res_polish.fun < best_val:
                                h_candidate = _sinusoid_heights(res_polish.x, K)
                                if np.all(np.isfinite(h_candidate)) and abs(np.sum(h_candidate)) > 1e-12:
                                    val = _evaluate(h_candidate)
                                    if val < best_val:
                                        best_val = val
                                        best_h = h_candidate.copy()
        except Exception:
            pass

    # Strategy 2: Try a direct low-dimensional parameterization
    # h[n] = a + b * cos(2*pi*n/N + phi) + c * cos(4*pi*n/N + psi)
    # This is a 5-parameter problem, easier to optimize
    if _time_remaining() > 20.0:
        def _obj_2tone(params):
            if _time_remaining() < 2.0:
                return 1e10
            a, b, phi, c, psi = params
            n = np.arange(N, dtype=np.float64)
            h = a + b * np.cos(2.0 * np.pi * n / N + phi) + c * np.cos(4.0 * np.pi * n / N + psi)
            if not np.all(np.isfinite(h)):
                return 1e10
            return _evaluate(h)

        bounds_2tone = [(-2.0, 2.0), (-2.0, 2.0), (0.0, 2.0 * np.pi),
                        (-2.0, 2.0), (0.0, 2.0 * np.pi)]

        # Try several starting points
        starts = [
            [1.0, 0.1, 0.0, 0.1, 0.0],
            [1.0, 0.5, 0.0, -0.5, 0.0],
            [1.0, 0.5, np.pi/2, 0.5, np.pi/2],
            [0.5, 1.0, 0.0, 1.0, 0.0],
            [1.0, 0.3, np.pi, 0.3, 0.0],
        ]

        for x0 in starts:
            if _time_remaining() < 5.0:
                break
            try:
                res = minimize(
                    _obj_2tone,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds_2tone,
                    options={'maxiter': 300, 'ftol': 1e-12}
                )
                if res.fun < best_val:
                    a, b, phi, c, psi = res.x
                    n = np.arange(N, dtype=np.float64)
                    h_candidate = a + b * np.cos(2.0 * np.pi * n / N + phi) + c * np.cos(4.0 * np.pi * n / N + psi)
                    if np.all(np.isfinite(h_candidate)) and abs(np.sum(h_candidate)) > 1e-12:
                        val = _evaluate(h_candidate)
                        if val < best_val:
                            best_val = val
                            best_h = h_candidate.copy()
            except Exception:
                pass

        # Also try DE on the 2-tone problem
        if _time_remaining() > 10.0:
            try:
                res_de = differential_evolution(
                    _obj_2tone,
                    bounds_2tone,
                    maxiter=20,
                    popsize=10,
                    tol=0.01,
                    seed=123,
                    polish=False
                )
                if res_de.fun < best_val:
                    a, b, phi, c, psi = res_de.x
                    n = np.arange(N, dtype=np.float64)
                    h_candidate = a + b * np.cos(2.0 * np.pi * n / N + phi) + c * np.cos(4.0 * np.pi * n / N + psi)
                    if np.all(np.isfinite(h_candidate)) and abs(np.sum(h_candidate)) > 1e-12:
                        val = _evaluate(h_candidate)
                        if val < best_val:
                            best_val = val
                            best_h = h_candidate.copy()
            except Exception:
                pass

    # Strategy 3: Try a smooth bump function parameterization
    # h[n] = a + b * exp(-((n - c)/w)^2)
    if _time_remaining() > 10.0:
        def _obj_bump(params):
            if _time_remaining() < 2.0:
                return 1e10
            a, b, c, w = params
            n = np.arange(N, dtype=np.float64)
            h = a + b * np.exp(-((n - c) / w) ** 2)
            if not np.all(np.isfinite(h)):
                return 1e10
            return _evaluate(h)

        bounds_bump = [(-2.0, 2.0), (-2.0, 2.0), (0.0, float(N)), (5.0, 100.0)]

        starts_bump = [
            [1.0, 0.1, 200.0, 50.0],
            [1.0, 0.5, 100.0, 30.0],
            [1.0, -0.5, 200.0, 30.0],
        ]

        for x0 in starts_bump:
            if _time_remaining() < 3.0:
                break
            try:
                res = minimize(
                    _obj_bump,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds_bump,
                    options={'maxiter': 200, 'ftol': 1e-10}
                )
                if res.fun < best_val:
                    a, b, c, w = res.x
                    n = np.arange(N, dtype=np.float64)
                    h_candidate = a + b * np.exp(-((n - c) / w) ** 2)
                    if np.all(np.isfinite(h_candidate)) and abs(np.sum(h_candidate)) > 1e-12:
                        val = _evaluate(h_candidate)
                        if val < best_val:
                            best_val = val
                            best_h = h_candidate.copy()
            except Exception:
                pass

    # Final safety check
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        best_h = np.ones(N, dtype=np.float64)

    return best_h
