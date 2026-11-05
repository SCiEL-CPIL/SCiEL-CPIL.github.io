"""Model-and-optimize program for third autocorrelation inequality.

Strategy:
  * Represent h as a constant baseline plus a zero-mean oscillatory perturbation
    parameterized by DCT-II coefficients (low-frequency basis).
  * Use an FFT-based soft-max surrogate of the signed max of the autoconvolution
    with an exact analytic gradient.
  * Optimize the DCT coefficients + baseline with L-BFGS-B from multiple random
    restarts, then polish in the full 400-dim space with increasing beta and
    a final high-precision pass.
"""
N = 400
import time
import numpy as np
from scipy.optimize import minimize
from scipy.fft import dct, idct, fft, ifft


def _autoconv(h):
    """Full autoconvolution via FFT (length 2N-1)."""
    n = len(h)
    size = 2 * n
    f = np.fft.rfft(h, n=size)
    c = np.fft.irfft(f * f, n=size)
    return c[:2 * n - 1]


def _hard(h):
    """Exact objective value."""
    s = h.sum()
    if abs(s) < 1e-12:
        return 1e15
    conv = _autoconv(h)
    m = np.max(conv)
    return abs(2.0 * N * m / (s * s))


def _soft_max_conv_grad(h, beta):
    """
    Return (soft_max, grad_wrt_h) where soft_max approximates max(conv(h,h)).
    conv_k = sum_j h_j h_{k-j}  for k in 0..2N-2 (full convolution)
    d(conv_k)/d(h_i) = h_{k-i} + h_{i-k}  (zero when index out of [0,N-1])
    """
    n = len(h)
    conv = _autoconv(h)
    m = np.max(conv)
    shifted = beta * (conv - m)
    shifted = np.clip(shifted, -400, 400)
    ez = np.exp(shifted)
    z = ez.sum()
    sm = m + np.log(z) / beta
    w = ez / z  # weights over conv entries, length 2N-1

    # grad_i = sum_k w_k * (h_{k-i} + h_{i-k})
    # term1_i = sum_k w_k h_{k-i}
    # term2_i = sum_k w_k h_{i-k}
    
    # Use FFT for speed
    # term1: sum_k w_k h_{k-i} = cross-correlation
    # Pad w to length 2N-1, h to length N
    # Use the fact that sum_k w_k h_{k-i} for i=0..N-1
    # This is equivalent to convolving w with reversed h, then extracting
    
    # Let's use the direct correlation approach with FFT
    # term1_i = sum_{k=0}^{2N-2} w_k h_{k-i} where h_j = 0 for j outside [0,N-1]
    # = sum_{j=0}^{N-1} h_j w_{j+i}
    # This is the cross-correlation of h and w at lag i
    
    # For efficiency, use FFT:
    # cross_corr(h, w)[i] = sum_j h_j w_{j+i}
    # = ifft(fft(h, 2N) * conj(fft(w, 2N)))... no, that's circular
    # Let's just use np.correlate since N=400 is small enough
    
    corr_hw = np.correlate(h, w, mode='full')
    # corr_hw[p] = sum_j h[j] w[j + p - (len(w)-1)] = sum_j h[j] w[j + p - (2N-2)]
    # We want sum_j h[j] w[j+i] => j + p - (2N-2) = j+i => p = i + 2N-2
    term1 = corr_hw[2*N-2 : 3*N-2]  # length N
    
    # term2_i = sum_k w_k h_{i-k}
    # = sum_j h_j w_{i-j}
    # np.correlate(w, h, 'full')[p] = sum_j w[j] h[j + p - (N-1)]
    # We want sum_j w[j] h[i-j] => j + p - (N-1) = i-j => p = i + N-1
    corr_wh = np.correlate(w, h, mode='full')
    term2 = corr_wh[N-1 : 2*N-1]  # length N
    
    grad = term1 + term2
    return sm, grad


def _obj_and_grad(h, beta):
    """Objective and gradient for L-BFGS-B in full 400-dim space."""
    s = h.sum()
    if abs(s) < 1e-9:
        return 1e15, np.zeros_like(h)
    sm, g_sm = _soft_max_conv_grad(h, beta)
    val = 2.0 * N * sm / (s * s)
    g = 2.0 * N * (g_sm * s - 2.0 * sm * np.ones_like(h)) / (s ** 3)
    return val, g


def _param_to_h(p):
    """
    p[0] = baseline, p[1:] = DCT-II coefficients (AC, zero-mean enforced).
    Returns h of length N.
    """
    baseline = p[0]
    m = len(p) - 1
    coeffs = np.zeros(N)
    if m > 0:
        coeffs[1:m+1] = p[1:m+1]
    h_osc = idct(coeffs, type=2, norm='ortho')
    h_osc = h_osc - h_osc.mean()
    h = baseline + h_osc
    return h


def _param_obj(p, beta):
    h = _param_to_h(p)
    s = h.sum()
    if abs(s) < 1e-9:
        return 1e15
    sm, _ = _soft_max_conv_grad(h, beta)
    return abs(2.0 * N * sm / (s * s))


def construct():
    t0 = time.time()
    DEADLINE = 140.0

    def time_left():
        return DEADLINE - (time.time() - t0)

    rng = np.random.default_rng(42)
    m_dct = 40  # number of AC DCT coefficients

    best_h = np.ones(N, dtype=float)
    best_val = _hard(best_h)

    # ---- Phase 0: Quick parametric families (from inspiration) ----
    if time_left() > 120.0:
        # Family 1: h[i] = 1 + a*cos(2*pi*m*i/N)
        for m in range(1, 21):
            if time_left() < 115.0:
                break
            for a in np.linspace(-0.8, 0.8, 17):
                h = 1.0 + a * np.cos(2.0 * np.pi * m * np.arange(N) / N)
                v = _hard(h)
                if v < best_val:
                    best_val = v
                    best_h = h.copy()
        
        # Family 2: Smoothed block
        for beta_blk in [0.5, 1.0, 2.0, 3.0, 5.0]:
            if time_left() < 115.0:
                break
            h = np.tanh(beta_blk * (np.arange(N) - N / 2.0))
            s = np.sum(h)
            if abs(s) < 1e-6:
                continue
            v = _hard(h)
            if v < best_val:
                best_val = v
                best_h = h.copy()

    # ---- Phase 1: DCT-parameterized optimization with multiple restarts ----
    n_restarts = 15
    for r in range(n_restarts):
        if time_left() < 40.0:
            break
        # Generate random initial DCT coefficients with decaying amplitude
        scale = 0.3 / (np.arange(1, m_dct + 1) ** 0.5)
        coeffs = rng.normal(0, 1.0, size=m_dct) * scale
        baseline = rng.choice([0.5, 1.0, 1.5, 2.0, -1.0])
        p0 = np.concatenate([[baseline], coeffs])
        
        # Evaluate initial
        h0 = _param_to_h(p0)
        v0 = _hard(h0)
        if v0 < best_val:
            best_val = v0
            best_h = h0.copy()
        
        try:
            beta_p1 = 20.0
            res = minimize(_param_obj, p0, args=(beta_p1,), method='L-BFGS-B',
                           options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10,
                                    'maxfun': 3000})
            h_new = _param_to_h(res.x)
            v_new = _hard(h_new)
            if v_new < best_val:
                best_val = v_new
                best_h = h_new.copy()
        except Exception:
            pass

    # ---- Phase 2: Full 400-dim polish with increasing beta ----
    h_curr = best_h.copy()
    s = h_curr.sum()
    if abs(s) > 1e-9:
        h_curr = h_curr / s  # normalize sum to 1

    beta_seq = [10.0, 20.0, 40.0, 80.0, 160.0]
    for beta in beta_seq:
        if time_left() < 25.0:
            break
        try:
            def obj(x):
                val, _ = _obj_and_grad(x, beta)
                return val
            def gr(x):
                _, g = _obj_and_grad(x, beta)
                return g
            res = minimize(obj, h_curr, jac=gr, method='L-BFGS-B',
                           options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-12,
                                    'maxfun': 6000})
            h_new = res.x
            v = _hard(h_new)
            if v < best_val:
                best_val = v
                best_h = h_new.copy()
                h_curr = h_new.copy()
        except Exception:
            pass

    # ---- Phase 3: Coordinate-descent / subgradient refinement ----
    h_curr = best_h.copy()
    best_local = _hard(h_curr)
    max_iters = 30
    it = 0
    while time_left() > 15.0 and it < max_iters:
        it += 1
        conv = _autoconv(h_curr)
        kmax = int(np.argmax(conv))
        lag = kmax - (N - 1)
        # Subgradient of max conv at lag: d conv[lag] / d h_i = h[lag-i] + h[i-lag]
        sg = np.zeros(N)
        for i in range(N):
            a = lag - i
            b = i - lag
            if 0 <= a < N:
                sg[i] += h_curr[a]
            if 0 <= b < N:
                sg[i] += h_curr[b]
        sg_centered = sg - sg.mean()
        ss = h_curr.sum()
        # We want to reduce max(conv)/sum^2. Step in -sg direction (reduces peak)
        # while maintaining sum.
        for alpha in [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]:
            cand = h_curr - alpha * sg_centered
            vv = _hard(cand)
            if vv < best_local:
                best_local = vv
                h_curr = cand
                if vv < best_val:
                    best_val = vv
                    best_h = cand.copy()
                break

    # ---- Phase 4: Final high-precision L-BFGS-B polish ----
    if time_left() > 10.0:
        h_curr = best_h.copy()
        s = h_curr.sum()
        if abs(s) > 1e-9:
            h_curr = h_curr / s
        for beta in [200.0, 400.0]:
            if time_left() < 6.0:
                break
            try:
                def obj(x):
                    val, _ = _obj_and_grad(x, beta)
                    return val
                def gr(x):
                    _, g = _obj_and_grad(x, beta)
                    return g
                res = minimize(obj, h_curr, jac=gr, method='L-BFGS-B',
                               options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-13,
                                        'maxfun': 8000})
                v = _hard(res.x)
                if v < best_val:
                    best_val = v
                    best_h = res.x.copy()
            except Exception:
                pass

    # ---- Phase 5: Random walk exploration (from inspiration) ----
    if time_left() > 20.0:
        rng2 = np.random.default_rng(123)
        for noise_scale in [0.1, 0.2, 0.3, 0.5]:
            if time_left() < 15.0:
                break
            for trial in range(10):
                if time_left() < 15.0:
                    break
                noise = rng2.randn(N) * noise_scale
                walk = np.cumsum(noise)
                sum_i = np.sum(np.arange(N))
                drift = np.sum(walk) / sum_i
                h = 1.0 + walk - drift * np.arange(N)
                if not np.all(np.isfinite(h)) or abs(np.sum(h) - N) > 1e-6:
                    continue
                v = _hard(h)
                if v < best_val:
                    best_val = v
                    best_h = h.copy()
                    # Quick polish
                    if time_left() > 15.0:
                        try:
                            h_curr = best_h.copy()
                            s = h_curr.sum()
                            if abs(s) > 1e-9:
                                h_curr = h_curr / s
                            for beta in [50.0, 100.0]:
                                if time_left() < 12.0:
                                    break
                                def obj(x):
                                    val, _ = _obj_and_grad(x, beta)
                                    return val
                                def gr(x):
                                    _, g = _obj_and_grad(x, beta)
                                    return g
                                res = minimize(obj, h_curr, jac=gr, method='L-BFGS-B',
                                               options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12,
                                                        'maxfun': 3000})
                                v2 = _hard(res.x)
                                if v2 < best_val:
                                    best_val = v2
                                    best_h = res.x.copy()
                        except Exception:
                            pass

    # ---- Sanity checks ----
    if not np.all(np.isfinite(best_h)):
        best_h = np.ones(N)
        best_val = _hard(best_h)
    if abs(best_h.sum()) < 1e-10:
        best_h = np.ones(N)
        best_val = _hard(best_h)

    return list(best_h)
