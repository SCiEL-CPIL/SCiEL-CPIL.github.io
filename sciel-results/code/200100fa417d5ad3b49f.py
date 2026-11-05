"""Rewritten program for third autocorrelation inequality optimization."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.fftpack import dct, idct

def _fft_autoconv(h):
    """Compute full autoconvolution using FFT for speed."""
    n = len(h)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    fh = np.fft.rfft(h, size)
    conv = np.fft.irfft(fh * fh, size)[:2 * n - 1]
    return conv

def _compute_value(h):
    s = np.sum(h)
    if abs(s) < 1e-12:
        return np.inf
    conv = _fft_autoconv(h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _soft_max_obj(h, beta=100.0):
    """Objective to minimize: 2*N*max(autoconv) / (sum(h))^2 using soft-max."""
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = _fft_autoconv(h)
    m = np.max(conv)
    if m > 0:
        shifted = beta * (conv - m)
        shifted = np.clip(shifted, -500, 500)
        soft_max = m + np.log(np.sum(np.exp(shifted))) / beta
        return 2.0 * N * soft_max / (s * s)
    else:
        return 2.0 * N * m / (s * s)

def _gradient_soft_max(h, beta=100.0):
    """Compute gradient of soft_max objective."""
    s = np.sum(h)
    if abs(s) < 1e-10:
        return np.zeros_like(h)
    
    conv = _fft_autoconv(h)
    m = np.max(conv)
    
    if m > 0:
        shifted = beta * (conv - m)
        shifted = np.clip(shifted, -500, 500)
        exp_shifted = np.exp(shifted)
        sum_exp = np.sum(exp_shifted)
        w = exp_shifted / sum_exp  # weights for soft-max
        
        # Gradient of soft_max w.r.t h
        # d(soft_max)/d(h[j]) = 2 * sum_k w[k] * h[k-j] for valid k-j
        # This is 2 * cross_correlation(w, h)[j]
        
        # Use FFT for efficient cross-correlation
        n_conv = len(conv)
        n_fft2 = 1
        while n_fft2 < n_conv + N - 1:
            n_fft2 *= 2
        
        W = np.fft.rfft(w, n=n_fft2)
        H = np.fft.rfft(h, n=n_fft2)
        
        # Cross-correlation: sum_k w[k] * h[k-j]
        # This is ifft(fft(w) * conj(fft(h))) but we need to handle the indexing carefully
        # Actually, let's compute it directly using the structure
        # grad[j] = 2 * sum_{k=0}^{2N-2} w[k] * h[k-j] where 0 <= k-j < N
        
        grad = np.zeros(N)
        for j in range(N):
            # k ranges from j to j + N - 1, but k must be in [0, 2N-2]
            k_start = j
            k_end = min(j + N, n_conv)
            if k_start < n_conv:
                h_slice = h[:k_end - k_start]
                w_slice = w[k_start:k_end]
                grad[j] = 2.0 * np.dot(w_slice, h_slice)
        
        val = 2.0 * N * (m + np.log(sum_exp) / beta) / (s * s)
        grad_val = 2.0 * N / (s * s) * grad - 4.0 * N * (m + np.log(sum_exp) / beta) / (s * s * s)
        return grad_val
    else:
        idx_max = np.argmax(conv)
        grad = np.zeros(N)
        for j in range(N):
            d = idx_max - j
            if 0 <= d < N:
                grad[j] = 2.0 * h[d]
        val = 2.0 * N * m / (s * s)
        grad_val = 2.0 * N / (s * s) * grad - 4.0 * N * m / (s * s * s)
        return grad_val

def _chirp_optimize(h0, deadline, rng, max_iter=200):
    """
    Optimize using geometric/chirp parameterization.
    h[k] = a * r^k * cos(theta*k + phi) + b
    """
    i = np.arange(N)
    
    def _chirp_h(params):
        a, r, theta, phi, b = params
        r = np.clip(r, -0.99, 0.99)
        h = a * (r ** i) * np.cos(theta * i + phi) + b
        return h
    
    def _chirp_objective(params):
        h = _chirp_h(params)
        s = np.sum(h)
        if abs(s) < 1e-10:
            return 1e10
        return _soft_max_obj(h, beta=100.0)
    
    best_val = _compute_value(h0)
    best_h = h0.copy()
    
    for a_init in [0.5, 1.0, -1.0, 2.0]:
        for r_init in [0.5, 0.8, 0.95, -0.5, -0.8]:
            for theta_init in [0.1, 0.5, 1.0, 2.0, 3.0]:
                for phi_init in [0.0, np.pi/2, np.pi]:
                    for b_init in [0.0, 1.0, -1.0]:
                        x0 = [a_init, r_init, theta_init, phi_init, b_init]
                        try:
                            res = minimize(_chirp_objective, x0, method='L-BFGS-B',
                                         options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8})
                            h_new = _chirp_h(res.x)
                            val = _compute_value(h_new)
                            if val < best_val:
                                best_val = val
                                best_h = h_new.copy()
                        except Exception:
                            pass
                        
                        if time.time() > deadline - 10.0:
                            return best_h
    
    return best_h

def _dct_optimize(h0, deadline, rng, max_iter=200):
    """Optimize using DCT parameterization with spectral amplitude control."""
    baseline = np.mean(h0)
    zero_mean = h0 - baseline
    
    dct_coeffs = dct(zero_mean, type=2, norm='ortho')
    
    def _dct_to_h(coeffs):
        zm = idct(coeffs, type=3, norm='ortho')
        return baseline + zm
    
    def _dct_objective(coeffs):
        h = _dct_to_h(coeffs)
        return _soft_max_obj(h)
    
    coeff_init = dct_coeffs.copy()
    
    try:
        res = minimize(_dct_objective, coeff_init,
                      method='L-BFGS-B',
                      options={'maxiter': max_iter, 'ftol': 1e-8, 'gtol': 1e-6})
        if res.success or res.fun < _compute_value(h0):
            h_new = _dct_to_h(res.x)
            return h_new
    except Exception:
        pass
    
    return h0

def _low_freq_spectral_optimize(h0, deadline, rng, num_modes=20):
    """Optimize only low-frequency DCT modes for better spectral shape."""
    baseline = np.mean(h0)
    zero_mean = h0 - baseline
    
    dct_coeffs = dct(zero_mean, type=2, norm='ortho')
    
    low_coeffs = dct_coeffs[:num_modes].copy()
    high_coeffs = dct_coeffs[num_modes:].copy()
    
    def _reconstruct_low(low_coeffs_vec):
        coeffs = np.concatenate([low_coeffs_vec, high_coeffs])
        zm = idct(coeffs, type=3, norm='ortho')
        return baseline + zm
    
    def _low_mode_objective(low_coeffs_vec):
        h = _reconstruct_low(low_coeffs_vec)
        return _soft_max_obj(h)
    
    try:
        res = minimize(_low_mode_objective, low_coeffs.copy(),
                      method='L-BFGS-B',
                      options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6})
        if res.success or res.fun < _compute_value(h0):
            h_new = _reconstruct_low(res.x)
            return h_new
    except Exception:
        pass
    
    return h0

def _random_walk_init(rng, N):
    """Generate random walk with negative drift as initialization."""
    drift = -0.01
    noise_scale = 0.5
    h = np.zeros(N)
    for i in range(N):
        if i == 0:
            h[i] = noise_scale * rng.standard_normal()
        else:
            h[i] = h[i-1] + drift + noise_scale * rng.standard_normal()
    return h

def _sinusoidal_init(rng, N):
    """Generate h as baseline + amplitude * cos(2*pi*f*i/N + phase)."""
    f = rng.uniform(1.0, N/2.0)
    amp = rng.uniform(-2.0, 2.0)
    phase = rng.uniform(0, 2*np.pi)
    baseline = rng.uniform(-1.0, 1.0)
    
    i = np.arange(N)
    h = baseline + amp * np.cos(2*np.pi*f*i/N + phase)
    return h

def _sinusoidal_optimize(h0, deadline, rng, max_iter=100):
    """Optimize h represented as baseline + amp * cos(2*pi*f*i/N + phase)."""
    i = np.arange(N)
    
    best_params = None
    best_val = _compute_value(h0)
    
    for f in [1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0, 30.0, 50.0, 75.0, 100.0]:
        def _obj_3param(params, freq=f):
            baseline, amp, phase = params
            h = baseline + amp * np.cos(2*np.pi*freq*i/N + phase)
            return _soft_max_obj(h, beta=50.0)
        
        cos_term = np.cos(2*np.pi*f*i/N)
        sin_term = np.sin(2*np.pi*f*i/N)
        A_mat = np.column_stack([np.ones(N), cos_term, sin_term])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A_mat, h0, rcond=None)
            baseline_init = coeffs[0]
            A_init = coeffs[1]
            B_init = coeffs[2]
            amp_init = np.sqrt(A_init**2 + B_init**2)
            phase_init = np.arctan2(-B_init, A_init)
        except:
            baseline_init = np.mean(h0)
            amp_init = 1.0
            phase_init = 0.0
        
        x0 = [baseline_init, amp_init, phase_init]
        
        try:
            res = minimize(_obj_3param, x0, method='L-BFGS-B',
                          options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8})
            if res.fun < best_val:
                best_val = res.fun
                best_params = (f, res.x[0], res.x[1], res.x[2])
        except Exception:
            pass
    
    if best_params is not None:
        f, baseline, amp, phase = best_params
        h_new = baseline + amp * np.cos(2*np.pi*f*i/N + phase)
        return h_new
    
    return h0

def _multi_sinusoidal_init(rng, N, num_modes=3):
    """Generate h as baseline + sum of num_modes sinusoidal components."""
    i = np.arange(N)
    h = np.zeros(N)
    baseline = rng.uniform(-1.0, 1.0)
    h += baseline
    
    for k in range(num_modes):
        f = rng.uniform(1.0, N/2.0)
        amp = rng.uniform(-2.0, 2.0) / (k + 1)
        phase = rng.uniform(0, 2*np.pi)
        h += amp * np.cos(2*np.pi*f*i/N + phase)
    
    return h

def _multi_sinusoidal_optimize(h0, deadline, rng, num_modes=3, max_iter=50):
    """Optimize h represented as baseline + sum of num_modes sinusoidal components."""
    i = np.arange(N)
    
    baseline = np.mean(h0)
    zero_mean = h0 - baseline
    dct_coeffs = dct(zero_mean, type=2, norm='ortho')
    
    params = [baseline]
    for k in range(num_modes):
        f = (k + 1) * 0.5
        amp = dct_coeffs[k+1] * 2.0
        phase = 0.0
        params.extend([f, amp, phase])
    
    def _multi_sin_objective(params):
        baseline = params[0]
        h = np.full(N, baseline)
        for k in range(num_modes):
            idx = 1 + 3*k
            f = params[idx]
            amp = params[idx+1]
            phase = params[idx+2]
            h += amp * np.cos(2*np.pi*f*i/N + phase)
        return _soft_max_obj(h, beta=50.0)
    
    try:
        res = minimize(_multi_sin_objective, params, method='L-BFGS-B',
                      options={'maxiter': max_iter, 'ftol': 1e-8, 'gtol': 1e-6})
        if res.fun < _compute_value(h0):
            baseline = res.x[0]
            h = np.full(N, baseline)
            for k in range(num_modes):
                idx = 1 + 3*k
                f = res.x[idx]
                amp = res.x[idx+1]
                phase = res.x[idx+2]
                h += amp * np.cos(2*np.pi*f*i/N + phase)
            return h
    except Exception:
        pass
    
    return h0

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_h = np.ones(N)
    best_val = _compute_value(best_h)
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Chirp/geometric optimization
    if time.time() < deadline - 60.0:
        h_chirp = _chirp_optimize(best_h, deadline, rng, max_iter=50)
        val_chirp = _compute_value(h_chirp)
        if val_chirp < best_val:
            best_val = val_chirp
            best_h = h_chirp.copy()
    
    # Phase 2: Generate diverse initializations
    initializations = []
    
    for _ in range(3):
        h0 = rng.standard_normal(N)
        if abs(np.sum(h0)) < 1e-6:
            h0[0] += 1.0
        initializations.append(h0)
    
    for period in [2, 3, 4, 5]:
        h0 = np.zeros(N)
        for i in range(N):
            h0[i] = 1.0 if (i % period) < period // 2 else -1.0
        h0 += 0.1 * rng.standard_normal(N)
        initializations.append(h0)
    
    for freq in [2, 5, 10, 20]:
        t = np.linspace(0, 2*np.pi*freq, N)
        h0 = np.sin(t) + 0.5 * np.cos(2*t)
        h0 += 0.1 * rng.standard_normal(N)
        initializations.append(h0)
    
    for deg in [1, 2, 3]:
        x = np.linspace(-1, 1, N)
        h0 = np.zeros(N)
        for d in range(deg + 1):
            h0 += rng.standard_normal() * x**d
        h0 += 0.1 * rng.standard_normal(N)
        initializations.append(h0)
    
    for block_size in [10, 20, 50]:
        h0 = np.zeros(N)
        n_blocks = N // block_size
        for b in range(n_blocks):
            h0[b*block_size:(b+1)*block_size] = rng.standard_normal()
        initializations.append(h0)
    
    for num_modes in [5, 10, 20]:
        h0 = np.zeros(N)
        coeffs = np.zeros(N)
        for k in range(num_modes):
            coeffs[k] = rng.standard_normal() * (1.0 / (k + 1))
        zm = idct(coeffs, type=3, norm='ortho')
        h0 = 1.0 + zm
        initializations.append(h0)
    
    for _ in range(3):
        h0 = _random_walk_init(rng, N)
        if abs(np.sum(h0)) < 1e-6:
            h0[0] += 1.0
        initializations.append(h0)
    
    for _ in range(5):
        h0 = _sinusoidal_init(rng, N)
        if abs(np.sum(h0)) < 1e-6:
            h0[0] += 1.0
        initializations.append(h0)
    
    for num_modes in [2, 3]:
        for _ in range(2):
            h0 = _multi_sinusoidal_init(rng, N, num_modes=num_modes)
            if abs(np.sum(h0)) < 1e-6:
                h0[0] += 1.0
            initializations.append(h0)
    
    # Optimize each initialization
    for init_idx, h0 in enumerate(initializations):
        if time.time() > deadline - 25.0:
            break
        
        h_curr = h0.copy()
        
        if time.time() < deadline - 20.0:
            h_dct = _dct_optimize(h_curr, deadline, rng, max_iter=20)
            val_dct = _compute_value(h_dct)
            if val_dct < _compute_value(h_curr):
                h_curr = h_dct
        
        if time.time() < deadline - 15.0:
            h_low = _low_freq_spectral_optimize(h_curr, deadline, rng, num_modes=10)
            val_low = _compute_value(h_low)
            if val_low < _compute_value(h_curr):
                h_curr = h_low
        
        if time.time() < deadline - 10.0:
            h_sin = _sinusoidal_optimize(h_curr, deadline, rng, max_iter=30)
            val_sin = _compute_value(h_sin)
            if val_sin < _compute_value(h_curr):
                h_curr = h_sin
        
        if time.time() < deadline - 8.0:
            h_multi = _multi_sinusoidal_optimize(h_curr, deadline, rng, num_modes=2, max_iter=25)
            val_multi = _compute_value(h_multi)
            if val_multi < _compute_value(h_curr):
                h_curr = h_multi
        
        val_curr = _compute_value(h_curr)
        if val_curr < best_val:
            best_val = val_curr
            best_h = h_curr.copy()
    
    # Phase 3: Final high-precision polish
    if time.time() < deadline - 10.0:
        for beta in [100.0, 200.0, 500.0]:
            try:
                res = minimize(_soft_max_obj, best_h,
                              method='L-BFGS-B',
                              args=(beta,),
                              options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
                val = _compute_value(res.x)
                if val < best_val:
                    best_val = val
                    best_h = res.x.copy()
            except Exception:
                pass
    
    if time.time() < deadline - 5.0:
        try:
            res = minimize(_soft_max_obj, best_h,
                          method='L-BFGS-B',
                          args=(1000.0,),
                          options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-10})
            val = _compute_value(res.x)
            if val < best_val:
                best_val = val
                best_h = res.x.copy()
        except Exception:
            pass
    
    if abs(np.sum(best_h)) < 1e-10:
        best_h = np.ones(N)
        best_val = _compute_value(best_h)
    
    return best_h
# EVOLVE-BLOCK-END
