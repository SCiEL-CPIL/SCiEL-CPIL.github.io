"""
Optimization for the third autocorrelation inequality.

Strategy: Represent h as inverse FFT of a spectral mask. Optimize spectral
amplitudes to minimize the autoconvolution peak via soft-max surrogate.
By Wiener-Khinchin, autoconvolution peak relates to integrated spectral power.
Flattening the spectrum (equalizing non-DC amplitudes) reduces peak-to-average.
"""
import numpy as np
import time
from scipy.fft import fft, ifft, rfft, irfft
from scipy.optimize import minimize

N = 400

def _convolve_fft(h):
    """Compute full autoconvolution using FFT."""
    n_conv = 2 * N - 1
    n_fft = 1
    while n_fft < n_conv:
        n_fft *= 2
    H = fft(h, n=n_fft)
    conv = ifft(H * H, n=n_fft).real[:n_conv]
    return conv

def _eval_value(h):
    """Compute the exact objective value."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return np.inf
    conv = _convolve_fft(h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _soft_max(x, beta):
    """Numerically stable soft maximum (log-sum-exp based)."""
    m = np.max(x)
    if not np.isfinite(m):
        return m
    exp_vals = np.exp(beta * (x - m))
    return m + np.log(np.sum(exp_vals)) / beta

def _soft_max_grad(x, beta):
    """Gradient of soft-max w.r.t. x."""
    m = np.max(x)
    if not np.isfinite(m):
        return np.zeros_like(x)
    exp_vals = np.exp(beta * (x - m))
    return exp_vals / np.sum(exp_vals)

def _objective_spectral(amplitudes, phases, beta):
    """
    Objective in spectral domain.
    
    amplitudes: array of shape (N//2 + 1,) - magnitudes of rfft coefficients
    phases: array of shape (N//2 + 1,) - phases of rfft coefficients
    beta: soft-max parameter
    
    We fix the DC amplitude to ensure sum(h) = N (i.e., mean = 1).
    The non-DC amplitudes are optimized.
    """
    # Build complex spectrum
    H = np.zeros(N // 2 + 1, dtype=complex)
    H[0] = amplitudes[0]  # DC component (fixed to N for sum=N)
    for k in range(1, N // 2 + 1):
        H[k] = amplitudes[k] * np.exp(1j * phases[k])
    
    # Reconstruct h
    h = irfft(H, n=N)
    
    if not np.all(np.isfinite(h)):
        return 1e10
    
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    
    # Normalize so sum = N
    h_norm = h * (N / s)
    
    # Compute autoconvolution
    conv = _convolve_fft(h_norm)
    
    # Soft max of conv
    sm = _soft_max(conv, beta)
    
    # Objective: 2 * sm / N (since sum = N, value = 2*N*max(conv)/N^2 = 2*max(conv)/N)
    obj = 2.0 * sm / N
    
    return obj

def _objective_spectral_flatness(amplitudes, beta):
    """
    Alternative: minimize spectral variance (promote flatness) as a proxy.
    This encourages equal energy distribution across frequencies.
    """
    # Only consider non-DC amplitudes
    non_dc = amplitudes[1:]
    if len(non_dc) == 0:
        return 1e10
    
    mean_amp = np.mean(non_dc)
    if mean_amp < 1e-10:
        return 1e10
    
    # Variance of amplitudes (normalized)
    var = np.var(non_dc) / (mean_amp * mean_amp)
    
    # Also add a small term to keep amplitudes reasonable
    return var + 0.01 * np.sum((non_dc - 1.0) ** 2)

def construct():
    """
    Returns a sequence of 400 signed step heights.
    Uses spectral parameterization: h = IFFT(H) where H has fixed DC and
    optimized non-DC amplitudes/phases.
    """
    deadline = time.time() + 140.0  # Safety margin before 150s limit
    
    # Start with all-ones (value 2.0)
    best_h = np.ones(N)
    best_val = 2.0
    
    # Initial spectrum: all amplitudes = 1, phases = 0 (gives h = delta-like, but let's start from ones)
    # For h = ones, FFT has DC = N, all other = 0
    # Let's start with a mix: DC = N, small non-DC components
    
    # Parameterize: amplitudes[0] is fixed to N (DC), amplitudes[1:] are free
    # phases[0] = 0 (DC is real positive), phases[1:] are free
    
    n_freq = N // 2 + 1  # Number of rfft coefficients
    
    # Initial: flat spectrum with small perturbation
    # Start from a reasonable point: amplitudes slightly above 0 for non-DC
    init_amplitudes = np.ones(n_freq)
    init_amplitudes[0] = N  # DC fixed
    init_phases = np.zeros(n_freq)
    
    current_amplitudes = init_amplitudes.copy()
    current_phases = init_phases.copy()
    
    # Multi-stage optimization with increasing beta
    betas = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0]
    
    for beta in betas:
        if time.time() > deadline - 30:
            break
        
        # Try multiple restarts
        n_restarts = 4 if beta < 10 else 2
        
        for trial in range(n_restarts):
            if time.time() > deadline - 30:
                break
            
            if trial == 0:
                amps_trial = current_amplitudes.copy()
                phases_trial = current_phases.copy()
            else:
                # Random perturbation
                rng = np.random.RandomState(42 + int(beta) + trial)
                amp_scale = 0.5 / (1 + trial)
                phase_scale = 0.5 / (1 + trial)
                amps_trial = current_amplitudes.copy()
                phases_trial = current_phases.copy()
                # Perturb non-DC amplitudes
                amps_trial[1:] += amp_scale * rng.randn(n_freq - 1)
                amps_trial[1:] = np.maximum(amps_trial[1:], 0.01)  # Keep positive
                # Perturb non-DC phases
                phases_trial[1:] += phase_scale * rng.randn(n_freq - 1)
                phases_trial[1:] = np.arctan(np.tan(phases_trial[1:]))  # Wrap to [-pi, pi]
            
            def obj_func(params, beta=beta):
                # params: [amplitudes[1:], phases[1:]]
                amps_full = np.concatenate([[N], params[:n_freq - 1]])
                phs_full = np.concatenate([[0.0], params[n_freq - 1:]])
                return _objective_spectral(amps_full, phs_full, beta)
            
            # Initial params for optimizer
            init_params = np.concatenate([amps_trial[1:], phases_trial[1:]])
            
            try:
                max_iter = 400 if beta < 50 else 200
                result = minimize(
                    obj_func,
                    init_params,
                    method='L-BFGS-B',
                    options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                if result.fun < 1e9:
                    # Reconstruct h
                    params_opt = result.x
                    amps_opt = np.concatenate([[N], params_opt[:n_freq - 1]])
                    phs_opt = np.concatenate([[0.0], params_opt[n_freq - 1:]])
                    
                    # Build spectrum
                    H = np.zeros(n_freq, dtype=complex)
                    H[0] = amps_opt[0]
                    for k in range(1, n_freq):
                        H[k] = amps_opt[k] * np.exp(1j * phs_opt[k])
                    
                    h_opt = irfft(H, n=N)
                    
                    if np.all(np.isfinite(h_opt)):
                        s_opt = np.sum(h_opt)
                        if abs(s_opt) > 1e-10:
                            h_opt = h_opt * (N / s_opt)  # Normalize to sum = N
                            v_opt = _eval_value(h_opt)
                            if v_opt < best_val:
                                best_val = v_opt
                                best_h = h_opt.copy()
                                current_amplitudes = amps_opt.copy()
                                current_phases = phs_opt.copy()
            except Exception:
                continue
        
        # Update current to best
        if best_val < 2.0:
            # Recompute spectrum from best_h
            H_best = rfft(best_h)
            current_amplitudes = np.abs(H_best)
            current_phases = np.angle(H_best)
            current_amplitudes[0] = N  # Ensure DC is correct
    
    # Final polish with high beta
    if time.time() < deadline - 10:
        for beta in [1000.0, 2000.0, 5000.0]:
            if time.time() > deadline - 5:
                break
            
            def obj_func(params, beta=beta):
                amps_full = np.concatenate([[N], params[:n_freq - 1]])
                phs_full = np.concatenate([[0.0], params[n_freq - 1:]])
                return _objective_spectral(amps_full, phs_full, beta)
            
            init_params = np.concatenate([current_amplitudes[1:], current_phases[1:]])
            
            try:
                result = minimize(
                    obj_func,
                    init_params,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                
                if result.fun < 1e9:
                    params_opt = result.x
                    amps_opt = np.concatenate([[N], params_opt[:n_freq - 1]])
                    phs_opt = np.concatenate([[0.0], params_opt[n_freq - 1:]])
                    
                    H = np.zeros(n_freq, dtype=complex)
                    H[0] = amps_opt[0]
                    for k in range(1, n_freq):
                        H[k] = amps_opt[k] * np.exp(1j * phs_opt[k])
                    
                    h_opt = irfft(H, n=N)
                    
                    if np.all(np.isfinite(h_opt)):
                        s_opt = np.sum(h_opt)
                        if abs(s_opt) > 1e-10:
                            h_opt = h_opt * (N / s_opt)
                            v_opt = _eval_value(h_opt)
                            if v_opt < best_val:
                                best_val = v_opt
                                best_h = h_opt.copy()
                                current_amplitudes = amps_opt.copy()
                                current_phases = phs_opt.copy()
            except Exception:
                continue
    
    # Ensure finite and sum != 0
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        return np.ones(N)
    
    return best_h
