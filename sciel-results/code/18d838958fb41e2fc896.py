"""
Combined approach: Band-limited Fourier representation with FFT-accelerated smooth optimization.

Structural change: Represent h as a band-limited signal (suppressing low frequencies to
reduce the DC peak in autocorrelation), then optimize in the frequency domain using
FFT-accelerated gradient computation for the smooth log-sum-exp surrogate.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 400
DEADLINE = 140.0  # Safety margin before 150s kill


def _eval_value(h):
    """Compute the objective value (lower is better)."""
    h = np.asarray(h, dtype=np.float64)
    if np.any(~np.isfinite(h)):
        return np.inf
    sh = float(np.sum(h))
    if abs(sh) < 1e-12:
        return np.inf
    conv = np.convolve(h, h)
    max_conv = float(np.max(conv))
    val = abs(2.0 * N * max_conv / (sh * sh))
    return val


def _fft_convolve(h):
    """Full convolution via FFT for speed."""
    size = 2 * N - 1
    fft_size = 1
    while fft_size < size:
        fft_size *= 2
    H = np.fft.rfft(h, fft_size)
    conv = np.fft.irfft(H * H, fft_size)[:size]
    return conv


def _smooth_objective(h, beta=50.0):
    """Smooth surrogate using log-sum-exp for max in autocorrelation."""
    s = float(np.sum(h))
    if abs(s) < 1e-12:
        return 1e18
    conv = _fft_convolve(h)
    m = np.max(conv)
    shifted = beta * (conv - m)
    shifted = np.clip(shifted, -700, 700)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_max = m + log_sum / beta
    val = 2.0 * N * soft_max / (s * s)
    return abs(val)


def _smooth_grad(h, beta=50.0):
    """
    Gradient of the smooth objective f(h) = |2N * softmax(conv) / s^2|.
    Uses FFT-accelerated correlation for d(soft_max)/d(h).
    """
    s = float(np.sum(h))
    if abs(s) < 1e-12:
        return np.zeros(N)
    
    conv = _fft_convolve(h)
    m = np.max(conv)
    shifted = beta * (conv - m)
    shifted = np.clip(shifted, -700, 700)
    weights = np.exp(shifted)
    weights /= np.sum(weights)
    soft_max = m + np.log(np.sum(weights)) / beta
    
    # d(soft_max)/d(h[j]) = sum_k weights[k] * d(conv[k])/d(h[j])
    # d(conv[k])/d(h[j]) = 2*h[k-j] if 0 <= k-j < N, else 0
    # So d(soft_max)/d(h[j]) = 2 * sum_{k=j}^{j+N-1} weights[k] * h[k-j]
    #                        = 2 * (weights * h)[j] in cross-correlation sense
    
    # Use FFT to compute: for each j, sum_{m=0}^{N-1} weights[m+j] * h[m]
    # This is the cross-correlation of weights and h, restricted to valid range.
    # corr[j] = sum_{m} weights[m+j] * h[m] for m=0..N-1
    
    # Pad weights and h to size >= 2N-1
    fft_size = 1
    while fft_size < 2 * N - 1:
        fft_size *= 2
    
    W = np.fft.rfft(weights, fft_size)
    H = np.fft.rfft(h, fft_size)
    
    # Cross-correlation: corr[j] = sum_m weights[m+j] * h[m]
    # = sum_m weights[j+m] * h[m]
    # This is like the 'valid' part of the cross-correlation.
    # Using FFT: if we compute ifft(W * conj(H)), we get circular cross-correlation.
    # For linear cross-correlation, we need to be careful.
    
    # Alternative: corr[j] = sum_{m=0}^{N-1} weights[j+m] * h[m]
    # Let's define: for j in [0, N-1], corr[j] = dot(weights[j:j+N], h)
    # This is a sliding dot product. We can compute it via FFT:
    # corr = ifft(fft(weights) * conj(fft(h))) gives circular correlation.
    # For the linear version with proper padding:
    
    W_padded = np.zeros(fft_size)
    W_padded[:len(weights)] = weights
    H_padded = np.zeros(fft_size)
    H_padded[:N] = h
    
    Wf = np.fft.rfft(W_padded)
    Hf = np.fft.rfft(H_padded)
    
    # Cross-correlation r[j] = sum_i W_padded[i+j] * H_padded[i] (circular)
    # Linear cross-correlation: r_lin[j] = sum_{i=0}^{N-1} weights[i+j] * h[i]
    # Using FFT: r_lin = ifft(fft(weights) * conj(fft(h))) ... but this gives circular.
    # With sufficient padding (fft_size >= 2N-1), the circular equals linear for the valid range.
    
    # Actually, let me use a different approach:
    # d(soft_max)/d(h[j]) = 2 * sum_{k=j}^{j+N-1} weights[k] * h[k-j]
    # Let n = k-j, then = 2 * sum_{n=0}^{N-1} weights[n+j] * h[n]
    # This is the cross-correlation of weights and h at lag j.
    
    # Compute via FFT:
    # cross_corr = ifft(fft(weights) * conj(fft(h)))
    # But we need the specific slice.
    
    # Let's just use the direct approach with numpy's correlate for N=400 (it's fast enough)
    # np.correlate(weights, h, mode='valid') gives sum_{i} weights[i] * h[i+j] for j=0..len(weights)-len(h)
    # We want sum_{i} weights[i+j] * h[i] = sum_{i} h[i] * weights[i+j]
    # This is np.correlate(h, weights, mode='valid')[::-1]? Let me check.
    
    # np.correlate(a, v, mode='valid'): out[k] = sum_m a[m+k] * conj(v[m])
    # So np.correlate(h, weights, mode='valid')[k] = sum_m h[m+k] * weights[m]
    # We want: sum_m h[m] * weights[m+j] = sum_m weights[m+j] * h[m]
    # Let n = m+j: sum_{n=j}^{j+N-1} weights[n] * h[n-j]
    # = sum_{n'} weights[n'+j] * h[n'] where n' = n-j, n'=0..N-1
    # This is NOT directly np.correlate in standard form.
    
    # Let me just compute it directly - for N=400, this is 400*400 = 160k operations, very fast.
    d_soft_max = np.zeros(N)
    for j in range(N):
        # d_soft_max[j] = 2 * sum_{n=0}^{N-1} weights[n+j] * h[n]
        # weights has length 2N-1, so weights[n+j] is valid for n=0..N-1, j=0..N-1
        d_soft_max[j] = 2.0 * np.dot(weights[j:j+N], h)
    
    # Objective: f = |2N * soft_max / s^2|
    inner = 2.0 * N * soft_max / (s * s)
    sign = 1.0 if inner >= 0 else -1.0
    
    # d(f)/d(h[j]) = sign * 2N * (d_soft_max[j]/s^2 - 2*soft_max/s^3)
    grad = sign * 2.0 * N * (d_soft_max / (s * s) - 2.0 * soft_max / (s ** 3))
    
    return grad


def _band_limited_signal(cutoff, n_harmonics=20, phase_seed=0):
    """
    Generate a band-limited signal where low frequencies (below cutoff) are suppressed.
    This reduces the DC peak in the autocorrelation.
    
    Parameters:
    - cutoff: frequency index below which amplitudes are zeroed
    - n_harmonics: number of harmonics to consider
    - phase_seed: seed for random phases
    
    Returns: array of length N
    """
    rng = np.random.default_rng(phase_seed)
    t = np.arange(N)
    h = np.zeros(N)
    
    for k in range(1, n_harmonics + 1):
        freq = k * N // 100  # Spread harmonics
        if freq < cutoff:
            continue
        amp = 1.0 / np.sqrt(k)
        phase = rng.uniform(0, 2 * np.pi)
        h += amp * np.cos(2 * np.pi * freq * t / N + phase)
    
    # Ensure non-zero sum
    s = np.sum(h)
    if abs(s) < 1e-10:
        h[0] += 0.01
    
    return h


def _generate_fourier_inits():
    """Generate diverse band-limited initial candidates."""
    candidates = []
    
    # Various cutoff frequencies
    for cutoff in [1, 2, 3, 5, 8, 12, 16, 20, 30, 40]:
        for seed in range(3):
            h = _band_limited_signal(cutoff, n_harmonics=30, phase_seed=seed)
            s = np.sum(h)
            if abs(s) > 1e-12:
                candidates.append(h / s)
    
    # Pure high-frequency cosine modes
    t = np.arange(N)
    for k in [5, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200]:
        for amp in [0.5, 1.0, 2.0, 3.0]:
            h = amp * np.cos(2 * np.pi * k * t / N)
            # Add small DC to ensure non-zero sum
            h += 0.01
            s = np.sum(h)
            if abs(s) > 1e-12:
                candidates.append(h / s)
    
    # Multi-frequency with gaps
    rng = np.random.default_rng(42)
    for trial in range(30):
        n_freq = rng.integers(3, 8)
        freqs = sorted(rng.choice(range(2, N//2), size=n_freq, replace=False))
        phases = rng.uniform(0, 2 * np.pi, n_freq)
        amps = rng.uniform(0.3, 1.0, n_freq)
        h = np.zeros(N)
        for f, p, a in zip(freqs, phases, amps):
            h += a * np.cos(2 * np.pi * f * t / N + p)
        h += rng.uniform(-0.05, 0.05)  # Small DC
        s = np.sum(h)
        if abs(s) > 1e-12:
            candidates.append(h / s)
    
    # M-sequence based (from current program)
    for seed in [1, 2, 3, 5, 7, 11, 13, 17]:
        n_lfsr = 16
        taps = [0, 1, 2, 4]
        state = seed
        if state == 0:
            state = 1
        seq = []
        for _ in range(N):
            feedback = 0
            for tap in taps:
                if (state >> tap) & 1:
                    feedback ^= 1
            state = (state >> 1) | (feedback << (n_lfsr - 1))
            seq.append(state & 1)
        h = 2.0 * np.array(seq, dtype=np.float64) - 1.0
        s = np.sum(h)
        if abs(s) < 1e-10:
            h[0] += 0.01
        candidates.append(h.copy())
    
    # Evaluate all candidates
    init_vals = []
    for h in candidates:
        if not np.all(np.isfinite(h)):
            continue
        v = _eval_value(h)
        if np.isfinite(v):
            init_vals.append((v, h.copy()))
    
    init_vals.sort(key=lambda x: x[0])
    return init_vals


def _local_refine(h0, max_time=25.0):
    """Local refinement using smooth objective with multiple betas."""
    t0 = time.time()
    best_h = h0.copy()
    best_val = _eval_value(h0)
    
    for beta in [30.0, 60.0, 100.0, 150.0, 200.0]:
        if time.time() - t0 > max_time * 0.8:
            break
        try:
            def obj_fn(h, b=beta):
                return _smooth_objective(h, beta=b)
            
            def grad_fn(h, b=beta):
                return _smooth_grad(h, beta=b)
            
            res = minimize(
                obj_fn,
                best_h,
                jac=grad_fn,
                method='L-BFGS-B',
                options={'maxiter': 300, 'maxfun': 600, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if res.x is not None and np.all(np.isfinite(res.x)):
                v = _eval_value(res.x)
                if v < best_val:
                    best_val = v
                    best_h = res.x.copy()
        except Exception:
            pass
    
    return best_h, best_val


def _multi_start(h_best, n_starts=8, max_time=25.0):
    """Multi-start local refinement with diverse perturbations."""
    t0 = time.time()
    best_h = h_best.copy()
    best_val = _eval_value(h_best)
    rng = np.random.default_rng(42)
    
    for trial in range(n_starts):
        if time.time() - t0 > max_time * 0.85:
            break
        
        # Vary perturbation type
        if trial % 3 == 0:
            # Gaussian perturbation
            scale = 0.05 + 0.03 * trial
            perturbation = rng.normal(0, scale, N)
            perturbation -= np.mean(perturbation)
            h_init = best_h + perturbation
        elif trial % 3 == 1:
            # Frequency-domain perturbation
            H = np.fft.rfft(best_h)
            noise = rng.normal(0, 0.1, len(H)) + 1j * rng.normal(0, 0.1, len(H))
            H += noise
            h_init = np.fft.irfft(H, N)
        else:
            # Add a random high-frequency component
            k_rand = rng.integers(20, N // 2)
            phase = rng.uniform(0, 2 * np.pi)
            t = np.arange(N)
            h_init = best_h + 0.2 * np.cos(2 * np.pi * k_rand * t / N + phase)
        
        s = np.sum(h_init)
        if abs(s) < 1e-12:
            continue
        h_init = h_init / s
        
        try:
            res = minimize(
                lambda h: _smooth_objective(h, beta=50.0),
                h_init,
                jac=lambda h: _smooth_grad(h, beta=50.0),
                method='L-BFGS-B',
                options={'maxiter': 150, 'maxfun': 300, 'ftol': 1e-14, 'gtol': 1e-11}
            )
            if res.x is not None and np.all(np.isfinite(res.x)):
                v = _eval_value(res.x)
                if v < best_val:
                    best_val = v
                    best_h = res.x.copy()
        except Exception:
            pass
    
    return best_h, best_val


def construct():
    t0 = time.time()
    
    def elapsed():
        return time.time() - t0
    
    def time_left():
        return DEADLINE - elapsed()
    
    # Start with seed
    best_h = np.ones(N)
    best_val = _eval_value(best_h)  # 2.0
    
    # Phase 1: Generate diverse band-limited initializations
    if time_left() > 60:
        init_vals = _generate_fourier_inits()
        if init_vals:
            top_k = min(6, len(init_vals))
            for i in range(top_k):
                if elapsed() > DEADLINE - 40:
                    break
                v_init, h_init = init_vals[i]
                if v_init < best_val:
                    best_val = v_init
                    best_h = h_init.copy()
    
    # Phase 2: Local refinement from best initialization
    if time_left() > 35:
        best_h, best_val = _local_refine(best_h, max_time=min(30.0, time_left() - 30.0))
    
    # Phase 3: Multi-start refinement
    if time_left() > 25:
        best_h, best_val = _multi_start(best_h, n_starts=8, max_time=min(25.0, time_left() - 20.0))
    
    # Phase 4: Final high-precision polish
    if time_left() > 12:
        try:
            res = minimize(
                lambda h: _smooth_objective(h, beta=150.0),
                best_h,
                jac=lambda h: _smooth_grad(h, beta=150.0),
                method='L-BFGS-B',
                options={'maxiter': 200, 'maxfun': 400, 'ftol': 1e-16, 'gtol': 1e-13}
            )
            if res.x is not None and np.all(np.isfinite(res.x)):
                v = _eval_value(res.x)
                if v < best_val:
                    best_val = v
                    best_h = res.x.copy()
        except Exception:
            pass
    
    # Phase 5: One more polish with even higher beta
    if time_left() > 6:
        try:
            res = minimize(
                lambda h: _smooth_objective(h, beta=250.0),
                best_h,
                jac=lambda h: _smooth_grad(h, beta=250.0),
                method='L-BFGS-B',
                options={'maxiter': 100, 'maxfun': 200, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            if res.x is not None and np.all(np.isfinite(res.x)):
                v = _eval_value(res.x)
                if v < best_val:
                    best_val = v
                    best_h = res.x.copy()
        except Exception:
            pass
    
    # Final safety check
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        best_h = np.ones(N)
    
    return best_h
