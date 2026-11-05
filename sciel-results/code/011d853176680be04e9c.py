"""Model-and-optimize program for third autocorrelation inequality.

Strategy:
  * Represent h as a constant baseline plus a zero-mean oscillatory perturbation.
  * Use the Discrete Cosine Transform (DCT) to enforce a specific low-frequency
    spectral shape that minimizes the peak-to-mean ratio of the autoconvolution.
  * Use an FFT-based soft-max surrogate of the signed max of the autoconvolution
    with an exact analytic gradient computed via FFT correlation.
  * Optimize the DCT coefficients with L-BFGS-B, then polish with a
    coordinate / subgradient step on the full 400-dim vector against the
    hard objective.
"""
N = 400
import time
import numpy as np
from scipy.optimize import minimize
from scipy.fftpack import dct, idct


def _autoconv(h):
    """Full autoconvolution via FFT (c[k] for lag k = -(N-1)..(N-1))."""
    n = len(h)
    f = np.fft.rfft(h, n=2 * n)
    c = np.fft.irfft(f * f, n=2 * n)
    return c


def _soft_max_conv(h, beta):
    """Return (soft_max_value, gradient w.r.t. h) using FFT-based gradient."""
    n = len(h)
    conv = _autoconv(h)
    m = np.max(conv)
    shifted = beta * (conv - m)
    shifted = np.clip(shifted, -500, 500)
    ez = np.exp(shifted)
    z = ez.sum()
    sm = m + np.log(z) / beta
    w = ez / z
    
    # Gradient of soft-max w.r.t. h_i = sum_k w_k * d(conv_k)/d(h_i)
    # d(conv_k)/d(h_i) = h[k-i] + h[i-k] (zero outside 0..n-1)
    # We compute this efficiently using FFT-based convolution.
    
    # term2_i = sum_k w_k h[i-k] = (w conv h)[i]
    W = np.fft.rfft(w, n=2 * n)
    H = np.fft.rfft(h, n=2 * n)
    conv_wh = np.fft.irfft(W * H, n=2 * n)
    term2 = conv_wh[0:n]  # sum_k w_k h[i-k]
    
    # term1_i = sum_k w_k h[k-i]
    # This is the cross-correlation of w and h at lag i
    # = ifft( conj(W) * H ) at appropriate indices
    corr_wh = np.fft.irfft(np.conj(W) * H, n=2 * n)
    # corr_wh[p] = sum_k w_k h[p+k-n] for p in 0..2n-1
    # We need to extract the right portion for i in 0..n-1
    # For i >= 0: term1_i = sum_{k=0}^{n-1} w_k h[k-i]
    # This corresponds to corr_wh indices where the lag is -i
    # corr_wh[p] = sum_k w_k h[(p+k) mod 2n]
    # For non-circular: we need to be careful
    # Let's use: term1_i = sum_j h_j w_{j+i} for j where j+i < n
    # This is the correlation at positive lags
    # corr_wh[n-1+i] should give us sum_j h_j w_{j+i} for i in 0..n-1
    term1 = corr_wh[n-1:2*n-1]
    
    grad = term1 + term2
    return sm, grad


def _obj_and_grad(h, beta):
    s = h.sum()
    if abs(s) < 1e-9:
        return 1e12, np.zeros_like(h)
    sm, g_sm = _soft_max_conv(h, beta)
    val = 2.0 * N * sm / (s * s)
    g = 2.0 * N * (g_sm * s - 2.0 * sm) / (s * s * s)
    return val, g


def _hard(h):
    s = h.sum()
    if abs(s) < 1e-9:
        return 1e12
    conv = _autoconv(h)
    return abs(2.0 * N * np.max(conv) / (s * s))


def _param_to_h(p):
    """Map parameter vector to a 400-length step-height vector.
    p = [baseline, dct_coeff_1, dct_coeff_2, ..., dct_coeff_m]
    where m is the number of AC DCT coefficients to use.
    """
    baseline = p[0]
    m = len(p) - 1
    # Create full DCT coefficient vector of length N
    coeffs = np.zeros(N)
    coeffs[0] = 0.0  # DC is handled by baseline
    coeffs[1:m+1] = p[1:m+1]
    # Inverse DCT to get oscillatory part
    h_osc = idct(coeffs, type=2, norm='ortho')
    # h_osc should be zero-mean if coeffs[0]=0, but let's enforce it
    h_osc = h_osc - h_osc.mean()
    # Add baseline
    h = baseline + h_osc
    return h


def _param_obj(p):
    h = _param_to_h(p)
    s = h.sum()
    if abs(s) < 1e-9:
        return 1e12
    conv = _autoconv(h)
    return abs(2.0 * N * np.max(conv) / (s * s))


def construct():
    t0 = time.time()
    DEADLINE = 130.0  # wall-clock budget; keep margin before 150s kill

    def time_left():
        return DEADLINE - (time.time() - t0)

    rng = np.random.default_rng(12345)

    # Number of DCT coefficients to use for the oscillatory part
    m_dct = 50

    # ---- Phase 1: optimize the DCT parametric family ----
    # Initial guesses: various baseline values and random DCT coefficients
    best_h = np.ones(N, dtype=float)
    best_val = _hard(best_h)

    # Try different baseline values and random DCT coefficient patterns
    baselines = [0.5, 1.0, 1.5, 2.0, -0.5, -1.0, 0.8, 1.2]
    candidates = []
    for baseline in baselines:
        for _ in range(15):
            coeffs = rng.normal(0, 0.5, size=m_dct)
            p = np.concatenate([[baseline], coeffs])
            candidates.append(p)
    
    # Also add some structured initial guesses
    for baseline in [1.0, 0.8, 1.2]:
        # Low-frequency dominant
        coeffs = np.zeros(m_dct)
        coeffs[0] = 1.0
        coeffs[1] = 0.5
        coeffs[2] = 0.25
        coeffs[3] = 0.125
        candidates.append(np.concatenate([[baseline], coeffs]))
        
        # Higher frequency mix
        coeffs = np.zeros(m_dct)
        coeffs[0] = 0.5
        coeffs[5] = 0.3
        coeffs[10] = 0.2
        coeffs[15] = 0.1
        candidates.append(np.concatenate([[baseline], coeffs]))
        
        # Very low frequency
        coeffs = np.zeros(m_dct)
        coeffs[0] = 0.8
        coeffs[1] = 0.4
        candidates.append(np.concatenate([[baseline], coeffs]))

    # Evaluate all candidates and pick the best ones
    scored = []
    for p in candidates:
        if time_left() < 30.0:
            break
        h = _param_to_h(p)
        v = _hard(h)
        scored.append((v, p))
    scored.sort(key=lambda x: x[0])
    top = scored[:10]

    # Optimize each top candidate with L-BFGS-B
    for _, p0 in top:
        if time_left() < 25.0:
            break
        try:
            res = minimize(_param_obj, p0, method='L-BFGS-B',
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10,
                                    'maxfun': 5000})
            h = _param_to_h(res.x)
            v = _hard(h)
            if v < best_val:
                best_val = v
                best_h = h.copy()
        except Exception:
            pass

    # ---- Phase 2: full 400-dim polish with smooth surrogate (FFT gradient) ----
    h_curr = best_h.copy()
    # normalize sum to 1 to keep objective well-scaled
    s = h_curr.sum()
    if abs(s) > 1e-9:
        h_curr = h_curr / s

    beta_seq = [8.0, 16.0, 32.0, 64.0, 128.0]
    for beta in beta_seq:
        if time_left() < 20.0:
            break
        try:
            def obj(x):
                val, _ = _obj_and_grad(x, beta)
                return val
            def gr(x):
                _, g = _obj_and_grad(x, beta)
                return g
            res = minimize(obj, h_curr, jac=gr, method='L-BFGS-B',
                           options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12,
                                    'maxfun': 8000})
            h_new = res.x
            v = _hard(h_new)
            if v < best_val:
                best_val = v
                best_h = h_new.copy()
                h_curr = h_new.copy()
        except Exception:
            pass

    # ---- Phase 3: coordinate / subgradient refinement on hard objective ----
    h_curr = best_h.copy()
    s = h_curr.sum()
    if abs(s) > 1e-9:
        h_curr = h_curr / s
    best_local = _hard(h_curr)
    iters = 0
    while time_left() > 15.0 and iters < 100:
        iters += 1
        conv = _autoconv(h_curr)
        # signed max index (lag)
        kmax = int(np.argmax(conv))
        # subgradient of max(conv) w.r.t. h at the maximizing lag
        n = N
        sg = np.zeros(n)
        lag = kmax - (n - 1)
        # d c[lag] / d h_i = h[lag - i] + h[i - lag]  (zero outside)
        for i in range(n):
            a = lag - i
            b = i - lag
            if 0 <= a < n:
                sg[i] += h_curr[a]
            if 0 <= b < n:
                sg[i] += h_curr[b]
        # current sum
        ss = h_curr.sum()
        if abs(ss) < 1e-9:
            break
        # value = 2N*maxc / ss^2 ; gradient direction (for maxc, ignore abs sign flip)
        # We want to reduce maxc relative to ss^2. Take a step that lowers maxc
        # while keeping sum positive: step = -alpha*(sg - (ss/n)*ones) to keep sum const.
        sg_centered = sg - sg.mean()
        found_improvement = False
        for alpha in [1e-2, 1e-3, 1e-4, 1e-5, 5e-3, 2e-3]:
            cand = h_curr - alpha * sg_centered
            vv = _hard(cand)
            if vv < best_local:
                best_local = vv
                h_curr = cand
                if vv < best_val:
                    best_val = vv
                    best_h = cand.copy()
                found_improvement = True
                break
        if not found_improvement:
            # Try small random perturbations
            for _ in range(5):
                pert = rng.normal(0, 1e-4, size=n)
                pert -= pert.mean()
                cand = h_curr + pert
                vv = _hard(cand)
                if vv < best_local:
                    best_local = vv
                    h_curr = cand
                    if vv < best_val:
                        best_val = vv
                        best_h = cand.copy()
                    found_improvement = True
                    break
            if not found_improvement:
                break

    # ---- Final: a few L-BFGS-B runs from best with high beta for crisp max ----
    if time_left() > 15.0:
        for beta in [256.0, 512.0, 1024.0]:
            if time_left() < 10.0:
                break
            try:
                def obj(x):
                    val, _ = _obj_and_grad(x, beta)
                    return val
                def gr(x):
                    _, g = _obj_and_grad(x, beta)
                    return g
                res = minimize(obj, best_h, jac=gr, method='L-BFGS-B',
                               options={'maxiter': 800, 'ftol': 1e-15, 'gtol': 1e-13,
                                        'maxfun': 15000})
                v = _hard(res.x)
                if v < best_val:
                    best_val = v
                    best_h = res.x.copy()
            except Exception:
                pass

    # ---- Sanity / validity ----
    if not np.all(np.isfinite(best_h)):
        best_h = np.ones(N)
        best_val = _hard(best_h)
    if abs(best_h.sum()) < 1e-10:
        best_h = np.ones(N)
        best_val = _hard(best_h)

    return list(best_h)
