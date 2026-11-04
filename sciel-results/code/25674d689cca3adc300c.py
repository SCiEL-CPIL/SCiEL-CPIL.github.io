import numpy as np
import time
import scipy.optimize

N = 400

def _autoconv_lin(x):
    """Compute linear autoconvolution using FFT."""
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    fx = np.fft.rfft(x, size)
    return np.fft.irfft(fx * fx, size)[:2 * n - 1]

def _metric_std(x):
    """Standard metric: max |c_k| / S^2 including k=0."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return float('inf')
    c = _autoconv_lin(x)
    return float(np.max(np.abs(c))) / (s * s)

def _admm_spectral(x0, rho=1.0, max_iters=200, deadline=None):
    """
    ADMM in spectral domain: minimize max|F_k|^2 / S^2
    by alternating between:
      1. Spectral mask update: soft-threshold |F_k| to flatten
      2. Signal update: project back to time domain preserving structure
    """
    x = x0.copy().astype(float)
    s = np.sum(x)
    if abs(s) < 1e-12:
        return x
    
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    
    # Initialize dual variable (spectral mask)
    fx = np.fft.rfft(x, size)
    mag = np.abs(fx)
    phase = np.angle(fx)
    
    best_x = x.copy()
    best_val = _metric_std(x)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute current spectral magnitude
        fx = np.fft.rfft(x, size)
        mag = np.abs(fx)
        phase = np.angle(fx)
        
        # Target: flatten the spectrum (minimize peak)
        # Use iterative soft-thresholding: push magnitudes toward mean
        target_mag = np.mean(mag)
        
        # ADMM-style: update spectral mask with soft-thresholding
        # New magnitude: blend between current and target
        alpha = 0.3 + 0.5 * (it / max_iters)  # anneal from 0.3 to 0.8
        new_mag = (1 - alpha) * mag + alpha * target_mag
        
        # Enforce minimum magnitude to avoid zeroing out
        min_mag = 0.05 * np.max(mag)
        new_mag = np.maximum(new_mag, min_mag)
        
        # Reconstruct in time domain
        fx_new = new_mag * np.exp(1j * phase)
        x_new = np.fft.irfft(fx_new, size)[:n]
        
        # Preserve sum (integral)
        s_new = np.sum(x_new)
        if abs(s_new) > 1e-12:
            x_new = x_new * (s / s_new)
        
        # Evaluate
        val = _metric_std(x_new)
        if val < best_val:
            best_val = val
            best_x = x_new.copy()
        
        x = x_new
    
    return best_x

def _multi_sine_construct(freqs, phases, amps, n=N):
    """Construct signal from sum of sinusoids."""
    t = np.arange(n)
    x = np.zeros(n)
    for f, p, a in zip(freqs, phases, amps):
        x += a * np.sin(2 * np.pi * f * t / n + p)
    return x

def _multi_sine_optimize(deadline=None):
    """
    Optimize multi-sine representation: minimize max|c_k|/S^2
    by optimizing frequencies, phases, and amplitudes.
    """
    n_sines = 12
    rng = np.random.default_rng(7)
    
    # Initial guess: evenly spaced frequencies with random phases
    best_x = np.ones(N)
    best_val = _metric_std(best_x)
    
    for trial in range(8):
        if deadline is not None and time.time() > deadline - 10:
            break
        
        # Random initialization
        freqs = np.sort(rng.uniform(1, n_sines * 2, size=n_sines))
        phases = rng.uniform(0, 2 * np.pi, size=n_sines)
        amps = rng.uniform(0.5, 2.0, size=n_sines)
        
        params = np.concatenate([freqs, phases, amps])
        
        def _ms_objective(p):
            f = p[:n_sines]
            ph = p[n_sines:2*n_sines]
            a = p[2*n_sines:]
            x = _multi_sine_construct(f, ph, a)
            return _metric_std(x)
        
        try:
            res = scipy.optimize.minimize(
                _ms_objective, params,
                method='L-BFGS-B',
                options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            x_opt = _multi_sine_construct(
                res.x[:n_sines], 
                res.x[n_sines:2*n_sines], 
                res.x[2*n_sines:]
            )
            val = _metric_std(x_opt)
            if val < best_val:
                best_val = val
                best_x = x_opt.copy()
        except Exception:
            pass
    
    return best_x

def _gradient_descent_polish(x, max_iters=300, deadline=None):
    """
    Gradient descent on the smooth surrogate for max|c_k|/S^2.
    Uses analytic gradient of the soft-max surrogate.
    """
    x = x.copy().astype(float)
    s = np.sum(x)
    if abs(s) < 1e-12:
        return x
    
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    
    beta = 30.0  # temperature for soft-max
    lr = 0.01
    best_x = x.copy()
    best_val = _metric_std(x)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute autoconvolution
        fx = np.fft.rfft(x, size)
        c = np.fft.irfft(fx * fx, size)[:2 * n - 1]
        abs_c = np.abs(c)
        
        s = np.sum(x)
        if abs(s) < 1e-12:
            break
        
        # Soft-max of |c_k| / s^2
        norm_c = abs_c / (s * s)
        m = np.max(norm_c)
        if m < 1e-15:
            break
        
        # Soft-max value and weights
        exps = np.exp(beta * (norm_c - m))
        weights = exps / np.sum(exps)
        soft_max_val = m + np.log(np.sum(exps)) / beta
        
        # Gradient of soft-max w.r.t. c_k
        # d(soft_max)/dc_k = beta * weights_k
        # dc_k/dx_i = x_{i-k} + x_{i+k} (for linear autoconv)
        grad_c = beta * weights  # gradient w.r.t. c_k
        
        # Chain rule: gradient w.r.t. x
        grad_x = np.zeros(n)
        for k in range(len(c)):
            if grad_c[k] < 1e-10:
                continue
            # c_k = sum_j x_j * x_{j-k}
            # dc_k/dx_i = x_{i-k} (if i>=k) + x_{i+k} (if i+k<n, k>0)
            for i in range(n):
                contrib = 0.0
                if i >= k:
                    contrib += x[i - k]
                if k > 0 and i + k < n:
                    contrib += x[i + k]
                grad_x[i] += grad_c[k] * contrib
        
        # Also account for the 1/s^2 factor
        # d(soft_max/s^2)/dx = (grad_x * s^2 - soft_max * 2*s * sum(grad_x)) / s^4
        # Simplify: just use grad_x / s^2 as approximation
        grad_x = grad_x / (s * s)
        
        # Normalize and step
        grad_norm = np.linalg.norm(grad_x)
        if grad_norm < 1e-15:
            break
        
        # Adaptive learning rate
        step = lr * (1.0 + it / max_iters)
        x_new = x - step * grad_x
        
        # Preserve sum
        s_new = np.sum(x_new)
        if abs(s_new) > 1e-12:
            x_new = x_new * (s / s_new)
        
        val = _metric_std(x_new)
        if val < best_val:
            best_val = val
            best_x = x_new.copy()
            lr *= 1.05  # increase step slightly on improvement
        else:
            lr *= 0.8  # decrease on failure
        
        x = x_new
    
    return best_x

def _spectral_flattening_iterative(x, max_iters=50, deadline=None):
    """
    Iterative spectral flattening: repeatedly modify spectrum to reduce peak.
    This is a structural approach - work directly in frequency domain.
    """
    x = x.copy().astype(float)
    s = np.sum(x)
    if abs(s) < 1e-12:
        return x
    
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    
    best_x = x.copy()
    best_val = _metric_std(x)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        fx = np.fft.rfft(x, size)
        mag = np.abs(fx)
        phase = np.angle(fx)
        
        # Identify peak frequencies and suppress them
        peak_threshold = 0.9 * np.max(mag)
        peak_mask = mag > peak_threshold
        
        # Reduce peak magnitudes, increase trough magnitudes
        target = np.mean(mag)
        new_mag = mag.copy()
        new_mag[peak_mask] = (1 - 0.3) * mag[peak_mask] + 0.3 * target
        trough_mask = mag < 0.7 * target
        new_mag[trough_mask] = (1 - 0.2) * mag[trough_mask] + 0.2 * target
        
        # Ensure non-negative
        new_mag = np.maximum(new_mag, 0.01 * np.max(mag))
        
        fx_new = new_mag * np.exp(1j * phase)
        x_new = np.fft.irfft(fx_new, size)[:n]
        
        # Preserve sum
        s_new = np.sum(x_new)
        if abs(s_new) > 1e-12:
            x_new = x_new * (s / s_new)
        
        val = _metric_std(x_new)
        if val < best_val:
            best_val = val
            best_x = x_new.copy()
        
        x = x_new
    
    return best_x

def construct():
    deadline = time.time() + 135.0
    
    # Start with constant sequence
    best_x = np.ones(N)
    best_val = _metric_std(best_x)
    
    # Phase 1: Spectral flattening from multiple random starts
    rng = np.random.default_rng(42)
    for trial in range(5):
        if time.time() > deadline - 40:
            break
        try:
            x_init = rng.choice([-1.0, 1.0], size=N).astype(float)
            s = np.sum(x_init)
            if abs(s) > 1e-6:
                x_init = x_init / s
            
            x_flattened = _spectral_flattening_iterative(x_init, max_iters=30, deadline=deadline - 30)
            val = _metric_std(x_flattened)
            if val < best_val:
                best_val = val
                best_x = x_flattened.copy()
        except Exception:
            pass
    
    # Phase 2: Multi-sine optimization (reduced dimensionality)
    if time.time() < deadline - 30:
        try:
            x_ms = _multi_sine_optimize(deadline=deadline - 15)
            val = _metric_std(x_ms)
            if val < best_val:
                best_val = val
                best_x = x_ms.copy()
        except Exception:
            pass
    
    # Phase 3: ADMM spectral refinement
    if time.time() < deadline - 20:
        try:
            x_admm = _admm_spectral(best_x, rho=1.0, max_iters=150, deadline=deadline - 10)
            val = _metric_std(x_admm)
            if val < best_val:
                best_val = val
                best_x = x_admm.copy()
        except Exception:
            pass
    
    # Phase 4: Gradient descent polish
    if time.time() < deadline - 10:
        try:
            x_grad = _gradient_descent_polish(best_x, max_iters=200, deadline=deadline - 5)
            val = _metric_std(x_grad)
            if val < best_val:
                best_val = val
                best_x = x_grad.copy()
        except Exception:
            pass
    
    # Final: Quick spectral flattening on best
    if time.time() < deadline - 5:
        try:
            x_final = _spectral_flattening_iterative(best_x, max_iters=20, deadline=deadline - 2)
            val = _metric_std(x_final)
            if val < best_val:
                best_val = val
                best_x = x_final.copy()
        except Exception:
            pass
    
    return best_x
