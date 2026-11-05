"""Rewritten program for third autocorrelation inequality optimization."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.fft import dct, idct, fft, ifft
from scipy.optimize import minimize

def _eval_value(h):
    """Compute the objective value."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _convolve_fft(h):
    """Compute full autoconvolution using FFT (faster than np.convolve for large N)."""
    n_conv = 2 * len(h) - 1
    n_fft = 1
    while n_fft < n_conv:
        n_fft *= 2
    H = fft(h, n=n_fft)
    conv = ifft(H * H, n=n_fft).real[:n_conv]
    return conv

def _soft_max(x, beta):
    """Numerically stable soft maximum."""
    m = np.max(x)
    if not np.isfinite(m):
        return m
    exp_vals = np.exp(beta * (x - m))
    sm = m + np.log(np.sum(exp_vals)) / beta
    return sm

def _objective_soft(h, beta):
    """
    Soft-max surrogate objective for autoconvolution peak.
    h is normalized internally so sum(h) = N.
    """
    if not np.all(np.isfinite(h)):
        return 1e10
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    h_norm = h * (N / s)
    conv = _convolve_fft(h_norm)
    sm = _soft_max(conv, beta)
    return abs(2.0 * N * sm / (N * N))

def _objective_exact(h):
    """Exact objective using FFT-based convolution."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    conv = _convolve_fft(h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _spike_objective(params):
    """
    Objective for spike+background parameterization.
    params: [a1, a2, ..., aK, loc1, loc2, ..., locK, b]
    where ai are spike amplitudes, loci are spike locations (continuous),
    b is the uniform background level.
    The spike at location loci is spread over a small window.
    """
    K = len(params) // 2
    amps = params[:K]
    locs = params[K:2*K]
    b = params[2*K]
    
    # Build h from background + spikes
    h = np.full(N, b)
    for k in range(K):
        loc = locs[k]
        # Spread spike over nearest 3 indices
        i0 = int(np.floor(loc)) % N
        i1 = (i0 + 1) % N
        i2 = (i0 - 1) % N
        frac = loc - int(np.floor(loc))
        h[i0] += amps[k] * (1.0 - frac)
        h[i1] += amps[k] * frac
    
    if not np.all(np.isfinite(h)):
        return 1e10
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = _convolve_fft(h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def construct():
    """
    Returns a sequence of 400 signed step heights.
    Uses FFT-based convolution with soft-max surrogate optimization.
    """
    deadline = time.time() + 140.0  # safety margin before 150s kill
    
    # Start with all-ones (score 2.0)
    best_h = np.ones(N)
    best_val = 2.0
    
    # Phase 1: Quick parametric families to get a good starting point
    # Family 1: h[i] = 1 + a*cos(2*pi*m*i/N)
    for m in range(1, 21):
        if time.time() > deadline - 80:
            break
        for a in np.linspace(-0.8, 0.8, 17):
            h = 1.0 + a * np.cos(2.0 * np.pi * m * np.arange(N) / N)
            v = _eval_value(h)
            if v < best_val:
                best_val = v
                best_h = h.copy()
    
    # Family 2: Smoothed block
    for beta in [0.5, 1.0, 2.0, 3.0, 5.0]:
        if time.time() > deadline - 80:
            break
        h = np.tanh(beta * (np.arange(N) - N / 2.0))
        s = np.sum(h)
        if abs(s) < 1e-6:
            continue
        v = _eval_value(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()
    
    # Phase 2: Optimization with soft-max surrogate using FFT convolution
    # Two-stage soft-max: low beta first, then high beta
    for beta in [1.0, 5.0, 20.0, 50.0, 100.0]:
        if time.time() > deadline - 20:
            break
        
        # Objective in time domain using soft-max
        def obj_time(h):
            return _objective_soft(h, beta)
        
        # Try random restarts around best_h
        rng = np.random.RandomState(42 + int(beta))
        
        for trial in range(4):
            if time.time() > deadline - 20:
                break
            
            # Start from best_h or perturbed version
            if trial == 0:
                h_init = best_h.copy()
            else:
                h_init = best_h + 0.1 * rng.randn(N)
                h_init -= np.mean(h_init)  # Keep zero-mean AC
                h_init += 1.0  # Add DC back
                h_init = h_init * (N / np.sum(h_init))
            
            try:
                result = minimize(
                    obj_time,
                    h_init,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                h_opt = result.x
                if np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-10:
                    v_opt = _objective_exact(h_opt)
                    if v_opt < best_val:
                        best_val = v_opt
                        best_h = h_opt.copy()
            except Exception:
                continue
    
    # Phase 3: High-precision polish with very high beta
    if time.time() < deadline - 15:
        for beta in [200.0, 500.0, 1000.0]:
            if time.time() > deadline - 10:
                break
            
            def obj_polish(h):
                return _objective_soft(h, beta)
            
            try:
                result = minimize(
                    obj_polish,
                    best_h.copy(),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                
                h_opt = result.x
                if np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-10:
                    v_opt = _objective_exact(h_opt)
                    if v_opt < best_val:
                        best_val = v_opt
                        best_h = h_opt.copy()
            except Exception:
                continue
    
    # Phase 4: Spike + background low-dimensional search
    if time.time() < deadline - 30:
        # Try different numbers of spikes
        for K in [1, 2, 3, 4]:
            if time.time() > deadline - 15:
                break
            
            # Initial guess: extract from best_h
            # Find the K largest deviations from mean
            mean_h = np.mean(best_h)
            deviations = np.abs(best_h - mean_h)
            top_idx = np.argsort(deviations)[::-1][:K]
            
            # Build initial params
            params_init = []
            for k in range(K):
                params_init.append(best_h[top_idx[k]] - mean_h)  # spike amplitude
            for k in range(K):
                params_init.append(float(top_idx[k]))  # location
            params_init.append(mean_h)  # background
            
            try:
                result = minimize(
                    _spike_objective,
                    params_init,
                    method='L-BFGS-B',
                    bounds=[(0.01, None)] * K + [(0.0, N - 1.0)] * K + [(-10.0, 10.0)],
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                # Convert back to h
                K_params = len(result.x)
                K_actual = K_params // 2
                amps = result.x[:K_actual]
                locs = result.x[K_actual:2*K_actual]
                b = result.x[2*K_actual]
                
                h_spike = np.full(N, b)
                for k in range(K_actual):
                    loc = locs[k]
                    i0 = int(np.floor(loc)) % N
                    i1 = (i0 + 1) % N
                    frac = loc - int(np.floor(loc))
                    h_spike[i0] += amps[k] * (1.0 - frac)
                    h_spike[i1] += amps[k] * frac
                
                if np.all(np.isfinite(h_spike)) and abs(np.sum(h_spike)) > 1e-10:
                    v_spike = _objective_exact(h_spike)
                    if v_spike < best_val:
                        best_val = v_spike
                        best_h = h_spike.copy()
            except Exception:
                continue
    
    # Phase 5: Coordinate descent polish in time domain
    if time.time() < deadline - 10:
        h = best_h.copy()
        step = 0.05
        for iteration in range(50):
            if time.time() > deadline - 5:
                break
            improved = False
            # Focus on indices where the signal has high magnitude
            idx = np.argsort(np.abs(h))[::-1][:50]  # top 50 indices
            for i in idx:
                if time.time() > deadline - 5:
                    break
                old = h[i]
                for delta in [step, -step]:
                    h[i] = old + delta
                    s = np.sum(h)
                    if abs(s) < 1e-10:
                        h[i] = old
                        continue
                    v = _objective_exact(h)
                    if v < best_val:
                        best_val = v
                        best_h = h.copy()
                        improved = True
                    else:
                        h[i] = old
            if not improved:
                step *= 0.5
                if step < 0.001:
                    break
    
    # Ensure finite and sum != 0
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        return np.ones(N)
    
    return best_h
# EVOLVE-BLOCK-END
