"""
Signed autocorrelation maximum absolute-value variant.

Strategy:
- Represent the sequence in the frequency domain as a complex vector with fixed magnitude.
- Optimize the phase to minimize the peak-to-mean power ratio of the autocorrelation.
- Use L-BFGS-B on a smooth epigraph/soft-max surrogate of the off-diagonal autocorrelation.
- Apply a final discrete polish (flip individual elements) to refine.
- Deadline guard: return best result found before 140s wall-clock.

The objective is: max_{k != 0} |autoconv[k]| / (sum(x))^2
where autoconv[k] = sum_i x[i] * x[i+k] (linear autoconvolution).

We minimize this quantity. The score returned is the raw value (lower is better).
"""

import numpy as np
import time
from scipy.optimize import minimize

N = 400

def _autoconv(x):
    """Linear autoconvolution via FFT."""
    n = len(x)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    X = np.fft.rfft(x, fsize)
    c = np.fft.irfft(X * X, fsize)[:size]
    return c

def _metric_offdiag(x):
    """Compute max |c_k| for k != 0 divided by (sum x)^2."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return float('inf')
    c = _autoconv(x)
    if len(c) > 1:
        max_val = np.max(np.abs(c[1:]))
    else:
        max_val = 0.0
    return max_val / (s * s)

def _soft_max_abs_offdiag(x, temp=1.0):
    """Soft-max approximation of max |c_k| for k != 0."""
    c = _autoconv(x)
    if len(c) <= 1:
        return 0.0
    abs_c = np.abs(c[1:])
    if len(abs_c) == 0:
        return 0.0
    max_c = np.max(abs_c)
    if max_c < 1e-15:
        return 0.0
    exp_vals = np.exp((abs_c - max_c) / temp)
    soft_max = max_c + temp * np.log(np.sum(exp_vals))
    return soft_max

def _objective_phase(phase, temp=1.0):
    """
    Objective in terms of phase vector.
    x = real(ifft(exp(i*phase))) scaled to have sum = 1.
    Minimize soft_max_abs_offdiag / (sum x)^2.
    """
    # Reconstruct x from phase
    n = len(phase)
    # Use full complex FFT
    freqs = np.fft.fftfreq(n)
    # Magnitude: use a smooth window to ensure x is real and has good properties
    # Actually, let's use a simpler approach: x = ifft(exp(i*phase))
    # But ifft of a pure phase signal may not be real.
    # Better: work with the full complex spectrum.
    
    # Let's use a different parameterization:
    # x is real, so X[k] = conj(X[n-k]).
    # We optimize the phase of the positive-frequency components.
    half_n = n // 2 + 1
    # phase has length half_n
    X = np.zeros(half_n, dtype=complex)
    # Fixed magnitude: use a raised-cosine or constant magnitude
    # For simplicity, use constant magnitude = 1 for now, or a window
    mag = np.ones(half_n)
    # Apply a slight taper to reduce Gibbs
    if n > 10:
        w = 0.5 + 0.5 * np.cos(2 * np.pi * np.arange(half_n) / n)
        mag = w
    X = mag * np.exp(1j * phase)
    # Reconstruct x
    x = np.fft.irfft(X, n)
    s = np.sum(x)
    if abs(s) < 1e-12:
        return 1e10
    x = x / s
    sm = _soft_max_abs_offdiag(x, temp)
    return sm

def _objective_time(x, temp=1.0):
    """Smooth objective in time domain."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return 1e10
    x = x / s
    sm = _soft_max_abs_offdiag(x, temp)
    return sm

def _gradient_descent_polish(x0, max_iter=2000, initial_lr=0.01, deadline=None):
    """Projected gradient descent with annealed temperature."""
    x = x0.copy().astype(float)
    s = np.sum(x)
    if abs(s) < 1e-12:
        return x
    x = x / s
    
    best_x = x.copy()
    best_val = _metric_offdiag(best_x)
    
    lr = initial_lr
    temp = 0.5
    fsize = 1
    while fsize < 2 * N:
        fsize *= 2
    
    for i in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        c = _autoconv(x)
        if len(c) <= 1:
            break
        abs_c = np.abs(c[1:])
        max_c = np.max(abs_c)
        if max_c < 1e-15:
            break
        
        exp_vals = np.exp((abs_c - max_c) / temp)
        sum_exp = np.sum(exp_vals)
        soft_max = max_c + temp * np.log(sum_exp)
        grad_abs_c = exp_vals / sum_exp
        c_off = c[1:]
        sign_c = np.sign(c_off)
        grad_c = grad_abs_c * sign_c
        
        w = np.zeros(2 * N - 1)
        w[1:] = grad_c
        X = np.fft.rfft(x, fsize)
        W = np.fft.rfft(w, fsize)
        grad_x = np.fft.irfft(W * np.conj(X), fsize)[:N]
        
        s = np.sum(x)
        grad_metric = grad_x / (s * s) - 2.0 * soft_max * np.ones(N) / (s ** 3)
        
        x_new = x - lr * grad_metric
        s_new = np.sum(x_new)
        if abs(s_new) < 1e-12:
            x_new = x + 1e-6 * np.random.randn(N)
            s_new = np.sum(x_new)
        x_new = x_new / s_new
        
        val = _metric_offdiag(x_new)
        if val < best_val:
            best_val = val
            best_x = x_new.copy()
        
        x = x_new
        lr *= 0.995
        temp *= 0.999
        
    return best_x

def _discrete_polish(x0, max_flips=5000, deadline=None):
    """Try flipping individual elements to reduce the metric."""
    x = x0.copy().astype(float)
    best_val = _metric_offdiag(x)
    best_x = x.copy()
    rng = np.random.default_rng(123)
    
    # Precompute fsize
    fsize = 1
    while fsize < 2 * N:
        fsize *= 2
    
    for _ in range(max_flips):
        if deadline is not None and time.time() > deadline:
            break
        
        # Try a random subset of indices
        indices = rng.choice(N, size=min(50, N), replace=False)
        improved = False
        for idx in indices:
            if deadline is not None and time.time() > deadline:
                break
            # Flip sign
            x_trial = x.copy()
            x_trial[idx] = -x_trial[idx]
            val = _metric_offdiag(x_trial)
            if val < best_val - 1e-12:
                best_val = val
                best_x = x_trial.copy()
                x = x_trial
                improved = True
                break
        
        if not improved:
            # Try small perturbations
            idx = rng.integers(N)
            for delta in [0.1, -0.1, 0.2, -0.2, 0.5, -0.5]:
                x_trial = x.copy()
                x_trial[idx] += delta
                val = _metric_offdiag(x_trial)
                if val < best_val - 1e-12:
                    best_val = val
                    best_x = x_trial.copy()
                    x = x_trial
                    break
    
    return best_x

def construct():
    t0 = time.time()
    deadline = t0 + 140.0
    
    best_x = np.ones(N)
    best_val = _metric_offdiag(best_x)
    
    rng = np.random.default_rng(42)
    
    # Phase-based optimization
    half_n = N // 2 + 1
    candidates = []
    
    # 1. All-ones (baseline)
    candidates.append(np.ones(N))
    
    # 2. Random phase initializations
    for trial in range(20):
        phase = rng.random(half_n) * 2 * np.pi
        # Optimize phase using L-BFGS-B
        try:
            res = minimize(
                lambda p: _objective_phase(p, temp=1.0),
                phase,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            opt_phase = res.x
            # Reconstruct x
            mag = np.ones(half_n)
            if N > 10:
                w = 0.5 + 0.5 * np.cos(2 * np.pi * np.arange(half_n) / N)
                mag = w
            X = mag * np.exp(1j * opt_phase)
            x = np.fft.irfft(X, N)
            s = np.sum(x)
            if abs(s) > 1e-6:
                x = x / s
                val = _metric_offdiag(x)
                if val < best_val:
                    best_val = val
                    best_x = x.copy()
                candidates.append(x.copy())
        except Exception:
            pass
        
        if time.time() > deadline - 60:
            break
    
    # 3. Time-domain optimization from diverse starts
    for x0 in candidates:
        if time.time() > deadline - 40:
            break
        
        x0 = x0.copy()
        s0 = np.sum(x0)
        if abs(s0) < 1e-6:
            continue
        x0 = x0 / s0
        
        try:
            # Multi-temperature L-BFGS
            for temp in [1.0, 0.3, 0.1]:
                res = minimize(
                    lambda x, t=temp: _objective_time(x, t),
                    x0,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                x_opt = res.x
                s_opt = np.sum(x_opt)
                if abs(s_opt) > 1e-6:
                    x_opt = x_opt / s_opt
                    val = _metric_offdiag(x_opt)
                    if val < best_val:
                        best_val = val
                        best_x = x_opt.copy()
                        x0 = x_opt.copy()
        except Exception:
            pass
    
    # 4. Gradient descent polish
    if time.time() < deadline - 10:
        try:
            polished = _gradient_descent_polish(
                best_x,
                max_iter=3000,
                initial_lr=0.005,
                deadline=deadline - 5
            )
            val = _metric_offdiag(polished)
            if val < best_val:
                best_val = val
                best_x = polished
        except Exception:
            pass
    
    # 5. Discrete polish
    if time.time() < deadline - 2:
        try:
            polished = _discrete_polish(
                best_x,
                max_flips=10000,
                deadline=deadline - 1
            )
            val = _metric_offdiag(polished)
            if val < best_val:
                best_val = val
                best_x = polished
        except Exception:
            pass
    
    return best_x
