"""Direct soft-min optimization of the max-overlap objective for the Erdos task.

We treat the problem as a smooth optimization: minimize a soft-min (log-sum-exp)
approximation of the maximum circular overlap O(s) = sum_i h_i (1 - h_{i-s})
over h in [0,1]^N with sum h = N/2.  We use L-BFGS-B with a projected-gradient
initialization from the spectral (circulant) construction.  The soft-min
temperature is annealed to sharpen toward the true max.
"""

import time
import numpy as np
from scipy.fft import fft, ifft
from scipy.optimize import minimize

N = 100
DEADLINE_S = 145.0


def _time_remaining():
    return DEADLINE_S - time.time()


def _max_overlap(h):
    H = fft(h)
    F = H * (1.0 - np.exp(-2j * np.pi * np.arange(N) / N))
    return float(np.max(np.real(ifft(F))))


def _project(h):
    """Project onto {0<=h_i<=1, sum_i h_i = N/2}."""
    h = np.clip(h, 0.0, 1.0)
    diff = h.sum() - 0.5 * N
    if diff > 1e-12:
        order = np.argsort(h)[::-1]
        for i in order:
            take = min(diff, h[i])
            h[i] -= take
            diff -= take
            if diff <= 1e-12:
                break
    elif diff < -1e-12:
        diff = -diff
        order = np.argsort(h)
        for i in order:
            take = min(diff, 1.0 - h[i])
            h[i] += take
            diff -= take
            if diff <= 1e-12:
                break
    return h


def _spectral_init():
    """Deterministic circulant spectral construction as initial guess."""
    EPS = 1e-4
    A = max(0.25 * N - (N - 3) * EPS, 0.0)
    mags = np.full(N, EPS)
    mags[0] = 0.5 * N
    mags[1] = A
    mags[N - 1] = A
    H = mags * np.exp(1j * 2 * np.pi * np.arange(N) / N)
    h = np.real(ifft(H))
    return _project(h)


def _soft_min_obj_and_grad(h, temp):
    """Soft-min (log-sum-exp) of O(s) and its gradient w.r.t. h."""
    H = fft(h)
    F = H * (1.0 - np.exp(-2j * np.pi * np.arange(N) / N))
    Ov = np.real(ifft(F))
    # Soft max: temp * logsumexp(Ov/temp)
    shifted = Ov / temp
    max_sh = np.max(shifted)
    logsumexp = max_sh + np.log(np.sum(np.exp(shifted - max_sh)))
    obj = temp * logsumexp
    # Gradient: d/d h_i of sum_s exp(Ov_s/temp) / sum exp = softmax(Ov/temp) * dOv/dh
    weights = np.exp(shifted - max_sh)
    weights /= weights.sum()  # softmax(Ov/temp)
    # dOv/dh: Ov(s) = sum_i h_i (1 - h_{i-s})
    # dOv(s)/dh_j = (1 - h_{j-s}) - h_{j+s}
    # grad_j = sum_s weights_s * [(1 - h_{j-s}) - h_{j+s}]
    idx = np.arange(N)
    # For each shift s, contribution to grad_j from weight_s:
    # grad_j += weight_s * (1 - h[(j-s)%N]) - weight_s * h[(j+s)%N]
    # grad = sum_s weight_s * (1 - h_shifted_minus) - sum_s weight_s * h_shifted_plus
    # where h_shifted_minus[j] = h[(j-s)%N] for the s with weight_s
    # This is a convolution: grad = (1 - h) * weights (circular) - h * weights_rev
    # Actually: sum_s w_s (1 - h_{j-s}) = sum_s w_s * (1-h)_{j-s} = circ_conv(w, 1-h)[j]
    # and sum_s w_s h_{j+s} = circ_conv(w_rev, h)[j] where w_rev[s] = w_{-s}
    W = fft(weights)
    Hm = fft(1.0 - h)
    Hp = fft(h)
    # circ_conv(a,b)[j] = sum_s a_s b_{j-s} => IFFT(FFT(a)*FFT(b))
    grad = np.real(ifft(W * Hm)) - np.real(ifft(W * Hp))
    return obj, grad


def _anneal_optimize(h0, t0):
    """Anneal soft-min temperature and run L-BFGS-B at each level."""
    h = h0.copy()
    best_h = h.copy()
    best_v = _max_overlap(h)
    
    temps = [1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01]
    
    for temp in temps:
        if _time_remaining() < 10.0:
            break
        try:
            res = minimize(
                _soft_min_obj_and_grad,
                h,
                args=(temp,),
                method='L-BFGS-B',
                jac=True,
                bounds=[(0.0, 1.0)] * N,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8,
                         'maxls': 20, 'maxfun': 1000}
            )
            h_new = _project(res.x)
            v_new = _max_overlap(h_new)
            if v_new < best_v:
                best_v = v_new
                best_h = h_new.copy()
            h = h_new
        except Exception:
            pass
    
    return best_h, best_v


def construct():
    t0 = time.time()
    h = _spectral_init()
    v = _max_overlap(h)
    
    if _time_remaining() > 15.0:
        h2, v2 = _anneal_optimize(h, t0)
        if v2 < v:
            h, v = h2, v2
    
    h = _project(h)
    return h
