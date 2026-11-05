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
  * Add diverse initialization strategies from inspiration (sign sequences,
    sine waves, polynomial perturbations, Gaussian bumps).
  * Use faster gradient computation via FFT instead of O(n^2) loops.
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
    # Use FFT for efficiency:
    W = np.fft.rfft(w, n=2 * n)
    H = np.fft.rfft(h, n=2 * n)
    
    # conv_wh = ifft( W * H ) gives sum_k w_k h[i-k] for i in 0..2n-1
    conv_wh = np.fft.irfft(W * H, n=2 * n)
    term2 = conv_wh[0:n]  # sum_k w_k h[i-k]
    
    # For the first term: sum_k w_k h[k-i]
    # This is the cross-correlation of w and h at lag i
    # = ifft( W * conj(H) ) at appropriate indices
    # For real signals: ifft( W * conj(H) ) = cross-correlation
    # = sum_k w_k h[p-k] which is the same as conv_wh!
    # Wait, that's not right. Let me reconsider.
    
    # Actually, for real sequences:
    # sum_k w_k h[k-i] = (w * h)[i] where * is cross-correlation
    # sum_k w_k h[i-k] = (w * h)[i] where * is convolution
    
    # Let's compute both via direct convolution for correctness:
    # grad_i = sum_k w_k h[k-i] + sum_k w_k h[i-k]
    
    # The sum of these two is: 2 * (w conv h)[i] if w is symmetric, but w is not necessarily symmetric.
    # Actually, since conv is autoconvolution, w is the softmax of conv, which is symmetric.
    # conv_k = conv_{2n-2-k} (by symmetry of autoconvolution)
    # So w_k = w_{2n-2-k}
    
    # Let's just compute it directly for n=400 using vectorized operations:
    # grad_i = sum_k w_k h[k-i] + sum_k w_k h[i-k]
    
    # For efficiency, let's use the fact that:
    # sum_k w_k h[i-k] = (w * h)[i] (convolution)
    # sum_k w_k h[k-i] = (w * h)[i] (cross-correlation)
    
    # Actually, let me just compute it properly using FFT:
    # The gradient of the autoconvolution at lag k with respect to h[i] is:
    # d(conv_k)/d(h_i) = h[k-i] + h[i-k]
    
    # So grad_i = sum_k w_k (h[k-i] + h[i-k])
    
    # Let's compute this using two separate convolutions:
    # term1_i = sum_k w_k h[k-i]
    # term2_i = sum_k w_k h[i-k]
    
    # term2 is the convolution of w and h at index i
    # term1 is the cross-correlation of w and h at index i
    
    # Using FFT:
    # term2 = ifft(FFT(w) * FFT(h)) [0:n]
    # term1 = ifft(FFT(w) * conj(FFT(h))) [0:n]  (for real signals)
    
    # Wait, let me be more careful. For real signals:
    # ifft(FFT(w) * FFT(h))[i] = sum_k w[k] h[i-k] (with proper indexing)
    # ifft(FFT(w) * conj(FFT(h)))[i] = sum_k w[k] h[i+k] (with proper indexing)
    
    # Hmm, this is getting confusing. Let me just use a simpler approach.
    
    # Since n=400 is small enough, let's compute the gradient directly but vectorized:
    # grad_i = sum_k w_k h[k-i] + sum_k w_k h[i-k]
    
    # For each i, k ranges over 0..2n-2
    # Let's create a matrix approach or just use a loop with numpy operations
    
    # Actually, let's just use the fact that for the autoconvolution:
    # conv = h * h (full convolution)
    # The gradient of sum_k w_k conv_k w.r.t. h is:
    # = 2 * (w * h) where * is cross-correlation, if w is symmetric
    
    # Since w is symmetric (w_k = w_{2n-2-k}), we have:
    # grad = 2 * cross_correlation(w, h)
    
    # Let's compute cross_correlation(w, h) using FFT:
    # cross_corr[i] = sum_k w[k] h[i+k] (with proper bounds)
    
    # Using FFT: cross_corr = ifft(FFT(w) * conj(FFT(h)))
    
    W_full = np.fft.fft(w, n=2 * n)
    H_full = np.fft.fft(h, n=2 * n)
    cross_corr = np.fft.ifft(W_full * np.conj(H_full)).real
    term1 = cross_corr[0:n]
    
    # And term2 is the convolution:
    conv_wh2 = np.fft.irfft(np.fft.rfft(w, n=2 * n) * np.fft.rfft(h, n=2 * n), n=2 * n)
    term2 = conv_wh2[0:n]
    
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


def _generate_diverse_inits(rng, n_inits=30):
    """Generate diverse initializations from inspiration."""
    inits = []
    
    # 1. Random normal with positive mean
    for _ in range(8):
        h0 = rng.standard_normal(N)
        h0 += 0.5
        if abs(np.sum(h0)) < 1e-3:
            h0[0] += 1.0
        inits.append(h0)
        
    # 2. Sign sequences with DC offset
    for _ in range(8):
        h0 = rng.choice([-1, 1], size=N).astype(float)
        h0 += 0.3 * rng.standard_normal()
        if abs(np.sum(h0)) < 1e-3:
            h0[0] += 1.0
        inits.append(h0)
        
    # 3. Sine waves with various frequencies and phases
    for freq in [1, 2, 3, 5, 7, 10]:
        t = np.linspace(0, 2*np.pi*freq, N, endpoint=False)
        h0 = np.sin(t + rng.random()*2*np.pi)
        h0 += 0.8
        inits.append(h0)
        
    # 4. Polynomial perturbations
    for _ in range(4):
        h0 = np.ones(N)
        t = np.linspace(0, 1, N)
        h0 += 0.2 * (rng.random() * t**2 - 0.1)
        if abs(np.sum(h0)) < 1e-3:
            h0[0] += 1.0
        inits.append(h0)
        
    # 5. Gaussian bumps
    for _ in range(4):
        h0 = np.zeros(N)
        center = rng.integers(0, N)
        width = rng.uniform(10, 50)
        t = np.arange(N)
        h0 += np.exp(-0.5 * ((t - center) / width)**2)
        h0 += 0.1 * rng.standard_normal(N)
        if abs(np.sum(h0)) < 1e-3:
            h0[0] += 1.0
        inits.append(h0)
    
    return inits


def construct():
    t0 = time.time()
    DEADLINE = 145.0  # wall-clock budget; keep margin before 150s kill

    def time_left():
        return DEADLINE - (time.time() - t0)

    rng = np.random.default_rng(12345)

    # Number of DCT coefficients to use for the oscillatory part
    m_dct = 20

    # ---- Phase 1: optimize the DCT parametric family ----
    best_h = np.ones(N, dtype=float)
    best_val = _hard(best_h)

    # Try different baseline values and random DCT coefficient patterns
    baselines = [0.5, 1.0, 1.5, 2.0, -0.5, -1.0]
    for baseline in baselines:
        if time_left() < 30.0:
            break
        # Random DCT coefficients
        coeffs = rng.normal(0, 0.5, size=m_dct)
        p0 = np.concatenate([[baseline], coeffs])
        
        try:
            res = minimize(_param_obj, p0, method='L-BFGS-B',
                           options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10,
                                    'maxfun': 4000})
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
    while time_left() > 8.0 and iters < 60:
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
        for alpha in [1e-3, 1e-4, 1e-5, 1e-2]:
            cand = h_curr - alpha * sg_centered
            vv = _hard(cand)
            if vv < best_local:
                best_local = vv
                h_curr = cand
                if vv < best_val:
                    best_val = vv
                    best_h = cand.copy()
                break

    # ---- Phase 4: Diverse initialization search from inspiration ----
    if time_left() > 20.0:
        inits = _generate_diverse_inits(rng, n_inits=20)
        for h0 in inits:
            if time_left() < 15.0:
                break
            h_curr = h0.copy()
            s = h_curr.sum()
            if abs(s) > 1e-9:
                h_curr = h_curr / s
            
            # Quick optimization with moderate beta
            try:
                def obj(x):
                    val, _ = _obj_and_grad(x, 32.0)
                    return val
                def gr(x):
                    _, g = _obj_and_grad(x, 32.0)
                    return g
                res = minimize(obj, h_curr, jac=gr, method='L-BFGS-B',
                               options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10,
                                        'maxfun': 3000})
                h_new = res.x
                v = _hard(h_new)
                if v < best_val:
                    best_val = v
                    best_h = h_new.copy()
            except Exception:
                pass

    # ---- Final: a few L-BFGS-B runs from best with high beta for crisp max ----
    if time_left() > 10.0:
        for beta in [128.0, 256.0, 512.0]:
            if time_left() < 6.0:
                break
            try:
                def obj(x):
                    val, _ = _obj_and_grad(x, beta)
                    return val
                def gr(x):
                    _, g = _obj_and_grad(x, beta)
                    return g
                res = minimize(obj, best_h, jac=gr, method='L-BFGS-B',
                               options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12,
                                        'maxfun': 8000})
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
