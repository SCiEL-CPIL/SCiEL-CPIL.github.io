import numpy as np
import time

N = 600

def _autoconv_peak(vals):
    # vals: length N, nonneg. Returns max of the discrete autoconvolution (h*conv)[k].
    a = np.asarray(vals, dtype=np.float64)
    conv = np.convolve(a, a, mode='full')          # length 2N-1
    # weight by overlap counts w(k) = number of index pairs (i,j) with j-i = k
    n = a.size
    k = np.arange(2 * n - 1)
    d = k - (n - 1)
    w = np.maximum(n - np.abs(d), 0)
    return float(np.max(conv * w))

def _quad_peak(c):
    # peak of autoconv of c*(1 - (2i/N-1)^2), via piecewise-quadratic convolution
    if c == 0.0:
        return 0.0
    x = 2.0 * np.arange(N) / N - 1.0
    f = c * (1.0 - x * x)
    s2 = float(np.dot(f, f))
    s4 = float(np.dot(f, f, f, f))
    return (2.0 * s2 * s2 - (2.0 / 3.0) * s4) / N

def construct():
    t0 = time.time()
    deadline = 150.0
    margin = 30.0

    x = 2.0 * np.arange(N) / N - 1.0
    base = 1.0 - x * x            # concave quadratic profile

    # reference L2 (all-ones)
    ref = float(np.sum(np.ones(N)))
    base_norm = float(np.sum(base * base))
    c_ref = np.sqrt(ref / base_norm)
    c = c_ref                     # start exactly at the reference score
    peak = _autoconv_peak(base * c)

    # line search: increase c to reduce peak; keep best feasible
    best_c = c
    best_peak = peak
    step = 0.05
    it = 0
    while time.time() + 1.0 < deadline - margin:
        # try to grow c a bit; if peak falls, accept and continue
        cand = c * (1.0 + step)
        pk = _quad_peak(cand)
        if pk < best_peak * (1.0 - 1e-9):
            c = cand
            peak = pk
            best_c = c
            best_peak = pk
            it += 1
            if it > 4000:
                break
        else:
            # shrink the growth step to find the peak minimum
            step *= 0.8
            if step < 1e-5:
                break
    # safety: ensure nonneg (base is nonneg, c>0)
    return (best_c * base).tolist()
