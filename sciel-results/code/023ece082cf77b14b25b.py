"""Minimize max|autoconv(x)| / (sum x)^2 via smooth optimization with diverse spectral starts."""
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

def _objective_smooth(x, temp=0.1):
    """Smooth approximation of the off-diagonal metric using soft-max."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return 1e10
    c = _autoconv(x)
    if len(c) <= 1:
        return 0.0
    abs_c = np.abs(c[1:])
    max_c = np.max(abs_c)
    soft_max = max_c + temp * np.log(np.sum(np.exp((abs_c - max_c) / temp)))
    return soft_max / (s * s)

def _gradient_descent_polish(x0, max_iter=2000, initial_lr=0.005, deadline=None):
    """Projected gradient descent with annealed temperature and backtracking line search."""
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
        
        # Soft-max gradient
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
        
        # Backtracking line search
        alpha = lr
        x_new = x - alpha * grad_metric
        s_new = np.sum(x_new)
        if abs(s_new) < 1e-12:
            x_new = x + 1e-6 * np.random.randn(N)
            s_new = np.sum(x_new)
        x_new = x_new / s_new
        
        val_new = _metric_offdiag(x_new)
        
        # Simple backtracking if value increases
        attempts = 0
        while val_new > best_val and attempts < 10:
            alpha *= 0.5
            x_new = x - alpha * grad_metric
            s_new = np.sum(x_new)
            if abs(s_new) < 1e-12:
                break
            x_new = x_new / s_new
            val_new = _metric_offdiag(x_new)
            attempts += 1
        
        if val_new < best_val:
            best_val = val_new
            best_x = x_new.copy()
        
        x = x_new
        lr *= 0.998
        temp *= 0.9995
        
    return best_x

def _generate_spectral_signal(freqs, amplitudes, phases, N):
    """Generate a real signal from spectral components."""
    t = np.arange(N)
    signal = np.zeros(N)
    for f, a, p in zip(freqs, amplitudes, phases):
        signal += a * np.cos(2 * np.pi * f * t / N + p)
    return signal

def construct():
    t0 = time.time()
    deadline = t0 + 140.0
    
    best_x = np.ones(N)
    best_val = _metric_offdiag(best_x)
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Constant (baseline)
    candidates.append(np.ones(N))
    
    # 2. Random ±1 sequences with non-zero mean
    for _ in range(20):
        x = rng.choice([-1, 1], N)
        if abs(np.sum(x)) > N * 0.1:
            candidates.append(x.astype(float))
    
    # 3. Block patterns
    for block_size in [5, 10, 20, 40, 80]:
        pattern = np.zeros(N)
        for i in range(N):
            pattern[i] = 1.0 if (i // block_size) % 2 == 0 else -1.0
        candidates.append(pattern)
    
    # 4. Spectral signals with random frequencies
    for n_comp in [3, 5, 8]:
        for _ in range(5):
            freqs = rng.choice(np.arange(1, N // 2), size=n_comp, replace=False)
            amplitudes = rng.uniform(0.5, 2.0, size=n_comp)
            phases = rng.uniform(0, 2 * np.pi, size=n_comp)
            signal = _generate_spectral_signal(freqs, amplitudes, phases, N)
            if abs(np.sum(signal)) > 1e-6:
                candidates.append(signal)
    
    # 5. Smoothed random
    for _ in range(10):
        x = rng.randn(N)
        # Simple moving average smoothing
        kernel = np.ones(5) / 5
        x = np.convolve(x, kernel, mode='same')
        candidates.append(x)
    
    # Optimize each candidate with L-BFGS-B and gradient polish
    for idx, x0 in enumerate(candidates):
        if time.time() > deadline - 10:
            break
        
        x0 = x0.copy().astype(float)
        s0 = np.sum(x0)
        if abs(s0) < 1e-6:
            continue
        x0 = x0 / s0
        
        # Phase 1: L-BFGS-B with decreasing temperature
        x_curr = x0.copy()
        for temp in [0.5, 0.1, 0.01]:
            if time.time() > deadline - 5:
                break
            try:
                res = minimize(
                    lambda x: _objective_smooth(x, temp=temp),
                    x_curr,
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                x_curr = res.x
            except Exception:
                break
        
        val = _metric_offdiag(x_curr)
        if val < best_val:
            best_val = val
            best_x = x_curr.copy()
        
        # Phase 2: Gradient polish
        if time.time() < deadline - 5:
            try:
                polished = _gradient_descent_polish(
                    x_curr,
                    max_iter=500,
                    initial_lr=0.003,
                    deadline=deadline - 3
                )
                val = _metric_offdiag(polished)
                if val < best_val:
                    best_val = val
                    best_x = polished.copy()
            except Exception:
                pass
    
    # Final polish on the best result
    if time.time() < deadline - 2:
        try:
            polished = _gradient_descent_polish(
                best_x,
                max_iter=3000,
                initial_lr=0.002,
                deadline=deadline - 1
            )
            val = _metric_offdiag(polished)
            if val < best_val:
                best_val = val
                best_x = polished
        except Exception:
            pass
    
    return best_x
