"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
h is a step function on N equal-width bins in [0,1], sum = N/2 (mean 0.5).
overlap(k) = sum_i h[i]*(1-h[i+k])  (circular).  We minimize max_k overlap(k).
"""
import numpy as np
from scipy.optimize import minimize

N = 100

# precompute the frequency set (real, zero-mean, conjugate-symmetric)
_freqs = np.arange(1, N // 2 + 1)
_NF = len(_freqs)

# deadline guard (wall clock). Evaluator kills at 150s; keep a safety margin.
import time as _time
_T_START = _time.time()
_DEADLINE = 125.0  # seconds of wall clock we allow ourselves


def _h_from_fourier(theta):
    """Reconstruct real h from Fourier coeffs; enforce mean 0.5 and [0,1]."""
    h = np.empty(N)
    h[0] = 0.0
    for j, f in enumerate(_freqs):
        c = theta[j]
        h[f] = c
        h[N - f] = c
    h = h * (N - 1)          # scale so mean(theta) -> mean(h) = 0.5
    h = h + 0.5              # enforce mean exactly 0.5
    h = np.clip(h, 0.0, 1.0)
    return h


def _autocorr(h):
    """Circular autocorrelation S_k = sum_i h[i]*h[i+k], via FFT."""
    H = np.fft.fft(h)
    return np.fft.ifft(np.abs(H) ** 2).real


def _softmax_overlap(h, beta):
    """Soft-max (log-sum-exp) of overlap values; smooth surrogate for max."""
    S = _autocorr(h)
    ov = S - h.sum()          # overlap at each shift
    m = ov.max()
    return m + np.log(np.sum(np.exp((ov - m) * beta)) / N) / beta


def _softmax_grad(h, beta):
    """Gradient of the soft-max overlap w.r.t. h (for the box-constrained solve)."""
    S = _autocorr(h)
    ov = S - h.sum()
    m = ov.max()
    p = np.exp((ov - m) * beta)
    p = p / p.sum()
    # d(ov_k)/dh_i = h[i-k] + h[i+k] - 2*h[i]
    d = np.zeros(N)
    for k in range(N):
        d += p[k] * (np.roll(h, -k) + np.roll(h, k) - 2.0 * h)
    return beta * d


def _objective_and_grad(theta, beta):
    h = _h_from_fourier(theta)
    val = _softmax_overlap(h, beta)
    gh = _softmax_grad(h, beta)
    # backprop through h = 0.5 + (N-1)*theta (clipping is subgradient; ignore)
    g = (N - 1) * gh
    return val, g


def _true_max_overlap(h):
    S = _autocorr(h)
    return float((S - h.sum()).max())


def construct():
    global _T_START
    # reset the wall clock at each construction call (each eval is fresh)
    _T_START = _time.time()
    budget = _DEADLINE - (_T_START - _T_START)  # = _DEADLINE
    # we track elapsed relative to _T_START
    def remaining():
        return _DEADLINE - (_time.time() - _T_START)

    best_h = np.full(N, 0.5)
    best_val = _true_max_overlap(best_h)   # 0.5 baseline

    rng = np.random.default_rng(1234)
    bounds = [(-0.25, 0.25)] * _NF

    # multi-start: random low-frequency modes + a few hand-tuned starts
    n_starts = 24
    for s in range(n_starts):
        if remaining() < 8.0:
            break
        if s == 0:
            theta0 = np.zeros(_NF)
        elif s == 1:
            theta0 = np.zeros(_NF); theta0[0] = 0.12
        elif s == 2:
            theta0 = np.zeros(_NF); theta0[1] = 0.10
        else:
            # random, low-pass weighted
            w = 1.0 / (1.0 + _freqs)
            theta0 = rng.standard_normal(_NF) * 0.10 * (w / w.max())

        beta = 30.0
        try:
            res = minimize(
                _objective_and_grad,
                theta0,
                args=(beta,),
                method='L-BFGS-B',
                jac=True,
                bounds=bounds,
                options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50},
            )
        except Exception:
            continue
        h = _h_from_fourier(res.x)
        val = _true_max_overlap(h)
        if val < best_val - 1e-9:
            best_val = val
            best_h = h

        # polish: increase beta and re-optimize from the best point found
        if remaining() < 8.0:
            break
        for beta in (60.0, 120.0, 240.0):
            if remaining() < 6.0:
                break
            try:
                res2 = minimize(
                    _objective_and_grad,
                    res.x,
                    args=(beta,),
                    method='L-BFGS-B',
                    jac=True,
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-13, 'gtol': 1e-9, 'maxls': 40},
                )
            except Exception:
                continue
            h2 = _h_from_fourier(res2.x)
            val2 = _true_max_overlap(h2)
            if val2 < best_val - 1e-9:
                best_val = val2
                best_h = h2

    # final safety: ensure validity (mean 0.5, in [0,1])
    best_h = np.clip(best_h, 0.0, 1.0)
    best_h = best_h + (0.5 - best_h.mean())   # restore exact mean 0.5
    best_h = np.clip(best_h, 0.0, 1.0)
    # re-enforce exact mean after the second clip (tiny correction)
    best_h = best_h + (0.5 - best_h.mean())
    return best_h
