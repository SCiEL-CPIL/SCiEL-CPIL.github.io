import numpy as np
from scipy.optimize import minimize
import time

N = 600
DEADLINE = 140.0  # seconds

def compute_score(h):
    """Compute the normalized max of autoconvolution."""
    h = np.asarray(h, dtype=np.float64)
    dx = 1.0 / (2 * N)  # interval width: total width 1/2, N intervals
    # Autoconvolution via FFT
    # (f*f)(t) at discrete points
    # Use FFT: conv = ifft(fft(h, 2*N-1) ** 2)
    nfft = 2 * N - 1
    F = np.fft.fft(h, nfft)
    conv = np.real(np.fft.ifft(F * F))
    # conv[i] corresponds to offset i*dx
    # The max over all offsets
    max_val = np.max(conv)
    # Normalize by conv at t=0 (i.e., conv[0])
    # Actually for autoconvolution, the value at offset 0 is sum(h^2)
    # But we want normalization that gives 1.0 for uniform
    # For uniform h=1: conv[0] = N (sum of 1^2 * dx^2... wait)
    # Let me think discretely: (f*f)(k*dx) = sum_i h_i * h_{i+k} * dx
    # At k=0: sum_i h_i^2 * dx
    # For uniform: N * 1 * dx = N * dx = N * (1/(2N)) = 1/2
    # Max is also 1/2 (at k=0)
    # So max/conv[0] = 1.0 ✓
    return max_val / conv[0]

def soft_max_surrogate(conv, T):
    """Soft-max approximation of max(conv)."""
    # Use log-sum-exp for numerical stability
    max_c = np.max(conv)
    return max_c + T * np.log(np.sum(np.exp((conv - max_c) / T)))

def objective(params, T):
    """Objective function for optimization."""
    h = np.exp(params)  # ensures nonnegativity
    dx = 1.0 / (2 * N)
    nfft = 2 * N - 1
    F = np.fft.fft(h, nfft)
    conv = np.real(np.fft.ifft(F * F))
    conv = np.maximum(conv, 0)  # numerical safety
    if conv[0] < 1e-15:
        return 1e10
    # Soft-max surrogate
    smax = soft_max_surrogate(conv, T)
    return smax / conv[0]

def construct():
    start_time = time.time()
    
    best_h = np.ones(N)
    best_score = compute_score(best_h)
    
    # Try multiple restarts with different initializations
    rng = np.random.default_rng(42)
    
    n_restarts = 8
    max_time_per_restart = (DEADLINE - 5.0) / n_restarts  # leave 5s for final polish
    
    for r in range(n_restarts):
        if time.time() - start_time > DEADLINE - 10.0:
            break
        
        # Different initializations
        if r == 0:
            h0 = np.ones(N)
        elif r == 1:
            # Cosine bump
            x = np.linspace(-1/4, 1/4, N, endpoint=False) + 1/(2*N)
            h0 = 0.5 * (1 + np.cos(2 * np.pi * x / (1/2)))
            h0 = np.maximum(h0, 0.01)
        elif r == 2:
            # Gaussian-like
            x = np.linspace(-1/4, 1/4, N, endpoint=False) + 1/(2*N)
            h0 = np.exp(-8 * x**2 / (1/4)**2)
            h0 = np.maximum(h0, 0.01)
        elif r == 3:
            # Random smooth
            h0 = 0.5 + 0.5 * np.sin(2 * np.pi * np.arange(N) / N * 3)
            h0 = np.maximum(h0, 0.01)
        elif r == 4:
            # Random
            h0 = rng.random(N) * 0.5 + 0.25
        elif r == 5:
            # Triangular
            h0 = 1 - 2 * np.abs(np.arange(N) / N - 0.5)
            h0 = np.maximum(h0, 0.01)
        elif r == 6:
            # Two bumps
            x = np.linspace(-1/4, 1/4, N, endpoint=False) + 1/(2*N)
            h0 = np.exp(-20*(x-0.1)**2) + np.exp(-20*(x+0.1)**2)
            h0 = np.maximum(h0, 0.01)
        else:
            # Random with noise
            h0 = 0.5 + 0.3 * np.sin(4*np.pi*np.arange(N)/N) + 0.2*rng.random(N)
            h0 = np.maximum(h0, 0.01)
        
        # Convert to log parameters
        params0 = np.log(h0)
        
        # Optimize with decreasing temperature
        T_schedule = [0.05, 0.02, 0.01, 0.005, 0.002]
        
        current_params = params0.copy()
        for T in T_schedule:
            if time.time() - start_time > DEADLINE - 10.0:
                break
            
            res = minimize(
                objective, current_params,
                args=(T,),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res.success or res.fun < 1e10:
                current_params = res.x
        
        h_opt = np.exp(current_params)
        score_opt = compute_score(h_opt)
        
        if score_opt < best_score:
            best_score = score_opt
            best_h = h_opt.copy()
    
    # Final polish with very low temperature
    if time.time() - start_time < DEADLINE - 5.0:
        params_polish = np.log(np.maximum(best_h, 1e-6))
        for T in [0.001, 0.0005, 0.0002]:
            if time.time() - start_time > DEADLINE - 2.0:
                break
            res = minimize(
                objective, params_polish,
                args=(T,),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if res.success or res.fun < 1e10:
                params_polish = res.x
        
        h_final = np.exp(params_polish)
        score_final = compute_score(h_final)
        
        if score_final < best_score:
            best_score = score_final
            best_h = h_final
    
    # Ensure nonnegativity
    best_h = np.maximum(best_h, 0.0)
    
    return list(best_h)
