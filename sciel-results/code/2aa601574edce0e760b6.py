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

def _zadoff_chu_real(m, q, n):
    """
    Generate a real-valued sequence from a Zadoff-Chu-like construction.
    x[i] = cos(2*pi*q*i*(i-1)/(2*m) + phase)
    This is the real part of a complex Zadoff-Chu sequence.
    """
    i = np.arange(n)
    # Zadoff-Chu phase: 2*pi*q*i*(i-1)/(2*m)
    phase = np.pi * q * i * (i - 1) / m
    return np.cos(phase)

def _chirp_objective(params):
    """
    Objective function for chirp optimization.
    params: [q, phase_offset, freq_scale]
    """
    q, phase_offset, freq_scale = params
    i = np.arange(N)
    # Generalized chirp: x[i] = cos(2*pi*freq_scale*q*i*(i-1)/(2*N) + phase_offset)
    phase = np.pi * freq_scale * q * i * (i - 1) / N + phase_offset
    x = np.cos(phase)
    return _metric_std(x)

def _optimize_chirp_1d(deadline):
    """
    1D scalar search over the chirp parameter q.
    Zadoff-Chu sequences have constant amplitude and zero auto-correlation
    for the complex case. The real part will have reduced peaks.
    """
    best_x = np.ones(N)
    best_val = _metric_std(best_x)
    
    # Search over q values
    # For Zadoff-Chu, q should be coprime with m=N
    # But we can search continuously for the real part
    q_candidates = np.linspace(1.0, N/2, 200)
    
    for q in q_candidates:
        if time.time() > deadline - 5:
            break
        x = _zadoff_chu_real(N, q, N)
        val = _metric_std(x)
        if val < best_val:
            best_val = val
            best_x = x.copy()
    
    # Refine around best q with L-BFGS-B
    if best_val < _metric_std(np.ones(N)):
        # Find the best q from the grid
        q_candidates = np.linspace(1.0, N/2, 200)
        vals = []
        for q in q_candidates:
            x = _zadoff_chu_real(N, q, N)
            vals.append(_metric_std(x))
        vals = np.array(vals)
        best_idx = np.argmin(vals)
        q_best = q_candidates[best_idx]
        
        # Local refinement
        def _local_obj(p):
            q, ph = p
            i = np.arange(N)
            phase = np.pi * q * i * (i - 1) / N + ph
            x = np.cos(phase)
            return _metric_std(x)
        
        try:
            res = scipy.optimize.minimize(
                _local_obj, [q_best, 0.0],
                method='L-BFGS-B',
                bounds=[(q_best - 0.5, q_best + 0.5), (-np.pi, np.pi)],
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            q_opt, ph_opt = res.x
            i = np.arange(N)
            phase = np.pi * q_opt * i * (i - 1) / N + ph_opt
            x_opt = np.cos(phase)
            val = _metric_std(x_opt)
            if val < best_val:
                best_val = val
                best_x = x_opt.copy()
        except Exception:
            pass
    
    return best_x

def _generalized_chirp_optimize(deadline):
    """
    Optimize a generalized chirp: x[i] = cos(a*i^2 + b*i + c)
    This is a 3-parameter family that includes Zadoff-Chu as a special case.
    """
    best_x = np.ones(N)
    best_val = _metric_std(best_x)
    
    def _chirp3(a, b, c):
        i = np.arange(N)
        return np.cos(a * i * i + b * i + c)
    
    def _obj3(p):
        a, b, c = p
        x = _chirp3(a, b, c)
        return _metric_std(x)
    
    # Multiple random starts
    rng = np.random.default_rng(123)
    for trial in range(20):
        if time.time() > deadline - 10:
            break
        
        a0 = rng.uniform(-0.1, 0.1)
        b0 = rng.uniform(-0.5, 0.5)
        c0 = rng.uniform(-np.pi, np.pi)
        
        try:
            res = scipy.optimize.minimize(
                _obj3, [a0, b0, c0],
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            a_opt, b_opt, c_opt = res.x
            x_opt = _chirp3(a_opt, b_opt, c_opt)
            val = _metric_std(x_opt)
            if val < best_val:
                best_val = val
                best_x = x_opt.copy()
        except Exception:
            pass
    
    return best_x

def _multi_freq_chirp_optimize(deadline):
    """
    Sum of chirps: x[i] = sum_k cos(a_k * i^2 + b_k * i + c_k)
    This gives more flexibility than a single chirp.
    """
    n_chirps = 3
    best_x = np.ones(N)
    best_val = _metric_std(best_x)
    
    def _multi_chirp(params):
        i = np.arange(N)
        x = np.zeros(N)
        for k in range(n_chirps):
            a = params[3*k]
            b = params[3*k+1]
            c = params[3*k+2]
            x += np.cos(a * i * i + b * i + c)
        return x
    
    def _obj_multi(p):
        x = _multi_chirp(p)
        return _metric_std(x)
    
    rng = np.random.default_rng(456)
    for trial in range(15):
        if time.time() > deadline - 15:
            break
        
        # Initialize with small random parameters
        params = np.zeros(3 * n_chirps)
        for k in range(n_chirps):
            params[3*k] = rng.uniform(-0.05, 0.05)
            params[3*k+1] = rng.uniform(-0.3, 0.3)
            params[3*k+2] = rng.uniform(-np.pi, np.pi)
        
        try:
            res = scipy.optimize.minimize(
                _obj_multi, params,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            x_opt = _multi_chirp(res.x)
            val = _metric_std(x_opt)
            if val < best_val:
                best_val = val
                best_x = x_opt.copy()
        except Exception:
            pass
    
    return best_x

def _spectral_flattening_iterative(x, max_iters=50, deadline=None):
    """
    Iterative spectral flattening: repeatedly modify spectrum to reduce peak.
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
        peak_threshold = 0.85 * np.max(mag)
        peak_mask = mag > peak_threshold
        
        # Reduce peak magnitudes, increase trough magnitudes
        target = np.mean(mag)
        new_mag = mag.copy()
        new_mag[peak_mask] = (1 - 0.4) * mag[peak_mask] + 0.4 * target
        trough_mask = mag < 0.6 * target
        new_mag[trough_mask] = (1 - 0.3) * mag[trough_mask] + 0.3 * target
        
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

def _gradient_descent_polish(x, max_iters=200, deadline=None):
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
    
    beta = 50.0  # temperature for soft-max
    lr = 0.005
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
        
        # Gradient w.r.t. x using FFT-based convolution
        # grad_x[i] = sum_k weights_k * (x[i-k] + x[i+k]) / s^2
        # This can be computed efficiently using FFT
        w = np.zeros(2 * n - 1)
        w[:] = weights
        # Symmetrize: w[k] contributes to both x[i-k] and x[i+k]
        # So the gradient is the convolution of x with w (symmetric)
        w_sym = np.zeros(size)
        w_sym[:2*n-1] = w
        # Make it symmetric for convolution
        w_sym_fft = np.fft.rfft(w_sym, size)
        grad_x = np.fft.irfft(fx * w_sym_fft, size)[:n]
        
        # Account for the 1/s^2 factor
        grad_x = grad_x / (s * s)
        
        # Normalize and step
        grad_norm = np.linalg.norm(grad_x)
        if grad_norm < 1e-15:
            break
        
        # Adaptive learning rate
        x_new = x - lr * grad_x
        
        # Preserve sum
        s_new = np.sum(x_new)
        if abs(s_new) > 1e-12:
            x_new = x_new * (s / s_new)
        
        val = _metric_std(x_new)
        if val < best_val:
            best_val = val
            best_x = x_new.copy()
            lr *= 1.02  # increase step slightly on improvement
        else:
            lr *= 0.9  # decrease on failure
        
        x = x_new
    
    return best_x

def construct():
    deadline = time.time() + 140.0  # Safety margin from 150s limit
    
    # Start with constant sequence
    best_x = np.ones(N)
    best_val = _metric_std(best_x)
    
    # Phase 1: Zadoff-Chu 1D search (fast, structural)
    if time.time() < deadline - 30:
        try:
            x_zc = _optimize_chirp_1d(deadline - 20)
            val = _metric_std(x_zc)
            if val < best_val:
                best_val = val
                best_x = x_zc.copy()
        except Exception:
            pass
    
    # Phase 2: Generalized 3-parameter chirp optimization
    if time.time() < deadline - 25:
        try:
            x_chirp = _generalized_chirp_optimize(deadline - 20)
            val = _metric_std(x_chirp)
            if val < best_val:
                best_val = val
                best_x = x_chirp.copy()
        except Exception:
            pass
    
    # Phase 3: Multi-chirp optimization (9 parameters)
    if time.time() < deadline - 20:
        try:
            x_multi = _multi_freq_chirp_optimize(deadline - 15)
            val = _metric_std(x_multi)
            if val < best_val:
                best_val = val
                best_x = x_multi.copy()
        except Exception:
            pass
    
    # Phase 4: Spectral flattening refinement
    if time.time() < deadline - 15:
        try:
            x_flat = _spectral_flattening_iterative(best_x, max_iters=40, deadline=deadline - 10)
            val = _metric_std(x_flat)
            if val < best_val:
                best_val = val
                best_x = x_flat.copy()
        except Exception:
            pass
    
    # Phase 5: Gradient descent polish
    if time.time() < deadline - 10:
        try:
            x_grad = _gradient_descent_polish(best_x, max_iters=150, deadline=deadline - 5)
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
