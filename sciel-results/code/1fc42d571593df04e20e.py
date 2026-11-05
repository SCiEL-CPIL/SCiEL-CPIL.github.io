"""Model-and-optimize program for third autocorrelation inequality.

Strategy:
  * Parameterize h as a 2-parameter family (DC + single high-frequency cosine).
  * Optimize these parameters for several fixed frequencies.
  * Polish the resulting 400-dim vector using FFT-based soft-max surrogate
    and subgradient steps.
"""
N = 400
import time
import numpy as np
from scipy.optimize import minimize


def _autoconv(h):
    # full autoconvolution via FFT (c[k] for lag k = -(N-1)..(N-1))
    n = len(h)
    f = np.fft.rfft(h, n=2 * n)
    c = np.fft.irfft(f * f, n=2 * n)
    return c


def _soft_max_conv(h, beta):
    """Return (soft_max_value, gradient w.r.t. h) using FFT-based gradient."""
    n = len(h)
    conv = _autoconv(h)
    
    # Compute soft-max value and weights
    shifted = beta * conv
    shifted -= np.max(shifted)
    shifted = np.clip(shifted, -500, 500)
    ez = np.exp(shifted)
    z = ez.sum()
    sm = np.max(conv) + np.log(z) / beta
    w = ez / z
    
    # grad_i = sum_k w[k] * (h[k-i] + h[i-k]) where h[j]=0 for j outside [0,n-1]
    # = sum_k w[k]*h[k-i] + sum_k w[k]*h[i-k]
    # Let c[d] = sum_k w[k]*h[k-d] (cross-correlation of w and h at lag d)
    # Then grad_i = c[i] + c[-i]
    # c[d] for d in [-(n-1), 2n-2]: use FFT
    # c[d] = sum_k w[k]*h[k-d], valid when 0<=k-d<n
    # Use zero-padded FFT:
    size = 4 * n
    Wf = np.fft.rfft(w, n=size)
    Hf = np.fft.rfft(h, n=size)
    # ifft(Wf * conj(Hf))[p] = sum_k w[k] * h[(k-p) mod size]
    # For linear cross-corr: c[d] = cc[d] for d>=0, c[d] = cc[size+d] for d<0
    cc = np.fft.irfft(Wf * np.conj(Hf), n=size)
    
    grad = np.zeros(n)
    for i in range(n):
        # c[i] = cc[i]
        # c[-i] = cc[size-i] if i>0, else cc[0]
        if i == 0:
            grad[i] = 2.0 * cc[0]
        else:
            grad[i] = cc[i] + cc[size - i]
    
    return sm, grad


def _obj_and_grad(h, beta):
    s = h.sum()
    if abs(s) < 1e-9:
        return 1e12, np.zeros_like(h)
    sm, g_sm = _soft_max_conv(h, beta)
    val = 2.0 * N * sm / (s * s)
    # dval/dh_i = 2N * ( g_sm_i * s^2 - sm * 2 s ) / s^4
    #          = 2N * ( g_sm_i * s - 2 sm ) / s^3
    g = 2.0 * N * (g_sm * s - 2.0 * sm) / (s * s * s)
    return val, g


def _hard(h):
    s = h.sum()
    if abs(s) < 1e-9:
        return 1e12
    conv = _autoconv(h)
    return abs(2.0 * N * np.max(conv) / (s * s))


def _param_to_h(p, freq):
    """Map parameter vector [A, B] to a 400-length step-height vector.
    h[n] = A + B * cos(2 * pi * freq * n / N)
    """
    i = np.arange(N)
    A, B = p[0], p[1]
    h = A + B * np.cos(2 * np.pi * freq * i / N)
    return h


def _param_obj(p, freq):
    h = _param_to_h(p, freq)
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

    # ---- Phase 1: optimize the 2-parameter family for several frequencies ----
    # Focus on high frequencies as per directive
    freqs_to_try = [N/2.0, N/2.0 - 1.0, N/2.0 - 2.0, N/3.0, N/4.0, 1.0]
    
    best_h = np.ones(N, dtype=float)
    best_val = _hard(best_h)

    for freq in freqs_to_try:
        if time_left() < 15.0:
            break
        
        # Initial guess: A=1, B=0.5
        p0 = np.array([1.0, 0.5])
        # Also try A=1, B=-0.5
        p0_neg = np.array([1.0, -0.5])
        
        for p_init in [p0, p0_neg]:
            if time_left() < 10.0:
                break
            try:
                def obj_2p(p):
                    return _param_obj(p, freq)
                
                res = minimize(obj_2p, p_init, method='L-BFGS-B',
                               options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10,
                                        'maxfun': 1000})
                h = _param_to_h(res.x, freq)
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

    # ---- Phase 3: coordinate / subgradient refinement on hard objective ----
    h_curr = best_h.copy()
    s = h_curr.sum()
    if abs(s) > 1e-9:
        h_curr = h_curr / s
    best_local = _hard(h_curr)
    iters = 0
    while time_left() > 6.0 and iters < 120:
        iters += 1
        conv = _autoconv(h_curr)
        # signed max index (lag)
        kmax = int(np.argmax(conv))
        # subgradient of max(conv) w.r.t. h at the maximizing lag
        n = N
        lag = kmax - (n - 1)
        # d c[lag] / d h_i = h[lag - i] + h[i - lag]  (zero outside)
        # Vectorized computation
        sg = np.zeros(n)
        # For each i, h[lag-i] is valid when 0 <= lag-i < n => lag-n < i <= lag
        # h[i-lag] is valid when 0 <= i-lag < n => lag < i <= lag+n
        i_range = np.arange(n)
        # term1: h[lag-i] where 0 <= lag-i < n
        idx1 = lag - i_range
        mask1 = (idx1 >= 0) & (idx1 < n)
        sg[mask1] += h_curr[idx1[mask1]]
        # term2: h[i-lag] where 0 <= i-lag < n
        idx2 = i_range - lag
        mask2 = (idx2 >= 0) & (idx2 < n)
        sg[mask2] += h_curr[idx2[mask2]]
        
        # current sum
        ss = h_curr.sum()
        if abs(ss) < 1e-9:
            break
        # value = 2N*maxc / ss^2 ; we want to reduce maxc relative to ss^2
        # Take a step that lowers maxc while keeping sum approximately constant
        sg_centered = sg - sg.mean()
        # Try multiple step sizes in both directions
        for alpha in [5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-2, 2e-3, 8e-4]:
            for direction in [-1, 1]:
                cand = h_curr + direction * alpha * sg_centered
                vv = _hard(cand)
                if vv < best_local:
                    best_local = vv
                    h_curr = cand
                    if vv < best_val:
                        best_val = vv
                        best_h = cand.copy()
                    break
            else:
                continue
            break

    # ---- Final: a few L-BFGS-B runs from best with high beta for crisp max ----
    if time_left() > 8.0:
        for beta in [64.0, 128.0, 256.0]:
            if time_left() < 4.0:
                break
            try:
                def obj(x):
                    val, _ = _obj_and_grad(x, beta)
                    return val
                def gr(x):
                    _, g = _obj_and_grad(x, beta)
                    return g
                res = minimize(obj, best_h, jac=gr, method='L-BFGS-B',
                               options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12,
                                        'maxfun': 5000})
                v = _hard(res.x)
                if v < best_val:
                    best_val = v
                    best_h = res.x.copy()
            except Exception:
                pass

    # ---- Final polish: one more round of subgradient with smaller steps ----
    if time_left() > 4.0:
        h_curr = best_h.copy()
        best_local = _hard(h_curr)
        for _ in range(50):
            if time_left() < 2.0:
                break
            conv = _autoconv(h_curr)
            kmax = int(np.argmax(conv))
            n = N
            lag = kmax - (n - 1)
            sg = np.zeros(n)
            i_range = np.arange(n)
            idx1 = lag - i_range
            mask1 = (idx1 >= 0) & (idx1 < n)
            sg[mask1] += h_curr[idx1[mask1]]
            idx2 = i_range - lag
            mask2 = (idx2 >= 0) & (idx2 < n)
            sg[mask2] += h_curr[idx2[mask2]]
            ss = h_curr.sum()
            if abs(ss) < 1e-9:
                break
            sg_centered = sg - sg.mean()
            for alpha in [1e-4, 5e-5, 1e-5, 2e-4, 3e-5]:
                for direction in [-1, 1]:
                    cand = h_curr + direction * alpha * sg_centered
                    vv = _hard(cand)
                    if vv < best_local:
                        best_local = vv
                        h_curr = cand
                        if vv < best_val:
                            best_val = vv
                            best_h = cand.copy()
                        break
                else:
                    continue
                break

    # ---- Sanity / validity ----
    if not np.all(np.isfinite(best_h)):
        best_h = np.ones(N)
        best_val = _hard(best_h)
    if abs(best_h.sum()) < 1e-10:
        best_h = np.ones(N)
        best_val = _hard(best_h)

    return list(best_h)
