"""Spectral optimization for Erdos minimum overlap task."""
import numpy as np
import time
from scipy.optimize import minimize

N = 100

def _max_overlap_fft(h):
    """Compute max over all circular shifts of sum(h[i]*(1-h[(i+s)%N]))."""
    # overlap(s) = sum_i h[i] * (1 - h[(i+s)%N])
    #            = sum_i h[i] - sum_i h[i]*h[(i+s)%N]
    # sum_i h[i] = N/2 (constant)
    # sum_i h[i]*h[(i+s)%N] = circular autocorrelation at lag s
    # Use FFT: autocorr = ifft(|fft(h)|^2)
    F = np.fft.fft(h)
    autocorr = np.fft.ifft(F * np.conj(F)).real
    # autocorr[s] = sum_i h[i] * h[(i-s)%N] ... need to be careful with indexing
    # Actually np.fft.ifft(|F|^2)[s] = sum_i h[i] * h[(i-s)%N]
    # We want sum_i h[i] * h[(i+s)%N] which is autocorr[-s] = autocorr[N-s]
    # But since we take max over all s, and autocorr is symmetric for real signals,
    # max over s of (N/2 - autocorr[s]) is the same as max over s of (N/2 - autocorr[(N-s)%N])
    # So we can just take max over all autocorr values.
    overlaps = N / 2.0 - autocorr
    return np.max(overlaps)

def _objective(params):
    """Objective: maximize overlap, with penalty for constraint violations."""
    # params: [a1, b1, a2, b2, ..., aK, bK] for K Fourier modes
    K = len(params) // 2
    h = np.full(N, 0.5)
    for k in range(K):
        ak = params[2*k]
        bk = params[2*k+1]
        h += ak * np.cos(2*np.pi*k*np.arange(N)/N) + bk * np.sin(2*np.pi*k*np.arange(N)/N)
    # Constraint: h in [0,1]
    if np.any(h < -1e-10) or np.any(h > 1.0 + 1e-10):
        # Penalty for out-of-bounds
        penalty = 1000.0 * (np.sum(np.maximum(0, -h)) + np.sum(np.maximum(0, h-1.0)))
        return _max_overlap_fft(np.clip(h, 0, 1)) + penalty
    return _max_overlap_fft(h)

def _grad(params):
    """Gradient of objective w.r.t. Fourier coefficients."""
    K = len(params) // 2
    h = np.full(N, 0.5)
    for k in range(K):
        ak = params[2*k]
        bk = params[2*k+1]
        h += ak * np.cos(2*np.pi*k*np.arange(N)/N) + bk * np.sin(2*np.pi*k*np.arange(N)/N)
    h = np.clip(h, 0, 1)
    
    # Overlap = N/2 - autocorr[s_max]
    # d(overlap)/d(h[j]) = -d(autocorr[s_max])/d(h[j])
    # autocorr[s] = sum_i h[i]*h[(i+s)%N]
    # d(autocorr[s])/d(h[j]) = h[(j+s)%N] + h[(j-s)%N]  (for s != 0)
    # For s=0: d(autocorr[0])/d(h[j]) = 2*h[j]
    
    F = np.fft.fft(h)
    autocorr = np.fft.ifft(F * np.conj(F)).real
    s_max = np.argmax(N/2.0 - autocorr)
    
    grad_h = np.zeros(N)
    for j in range(N):
        if s_max == 0:
            grad_h[j] = -2*h[j]
        else:
            grad_h[j] = -(h[(j+s_max)%N] + h[(j-s_max)%N])
    
    # Chain rule: d(overlap)/d(params) = sum_j grad_h[j] * d(h[j])/d(params)
    grad = np.zeros(len(params))
    for k in range(K):
        cos_k = np.cos(2*np.pi*k*np.arange(N)/N)
        sin_k = np.sin(2*np.pi*k*np.arange(N)/N)
        grad[2*k] = np.dot(grad_h, cos_k)
        grad[2*k+1] = np.dot(grad_h, sin_k)
    return grad

def _construct_from_fourier(params):
    """Build h from Fourier coefficients, clip to [0,1], normalize to sum=N/2."""
    K = len(params) // 2
    h = np.full(N, 0.5)
    for k in range(K):
        ak = params[2*k]
        bk = params[2*k+1]
        h += ak * np.cos(2*np.pi*k*np.arange(N)/N) + bk * np.sin(2*np.pi*k*np.arange(N)/N)
    h = np.clip(h, 0.0, 1.0)
    # Normalize to sum = N/2
    s = h.sum()
    if s > 0:
        h = h * (N/2.0) / s
        h = np.clip(h, 0.0, 1.0)
        # Final exact fix
        diff = N/2.0 - h.sum()
        if abs(diff) > 1e-12:
            # Adjust the element closest to 0.5 to minimize disruption
            idx = np.argmin(np.abs(h - 0.5))
            h[idx] += diff
            h = np.clip(h, 0.0, 1.0)
    return h

def construct():
    deadline = time.time() + 110.0  # Safety margin before 150s kill
    
    # Start with uniform
    best_h = np.full(N, 0.5)
    best_score = _max_overlap_fft(best_h)  # Should be 0.25
    
    # Use first few Fourier modes
    K = 8  # 8 frequency components = 16 parameters
    rng = np.random.default_rng(42)
    
    # Multiple random restarts
    for restart in range(10):
        if time.time() > deadline:
            break
        
        # Random initial Fourier coefficients
        if restart == 0:
            params0 = np.zeros(2*K)
        else:
            params0 = rng.uniform(-0.3, 0.3, size=2*K)
        
        try:
            result = minimize(
                _objective,
                params0,
                jac=_grad,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            h_candidate = _construct_from_fourier(result.x)
            sc = _max_overlap_fft(h_candidate)
            
            if sc < best_score:
                best_score = sc
                best_h = h_candidate.copy()
                
        except Exception:
            continue
    
    # Local refinement: try small perturbations in Fourier space around best
    if time.time() < deadline:
        # Extract approximate Fourier coefficients of best_h
        F_best = np.fft.fft(best_h)
        # Use only first K modes
        params_best = np.zeros(2*K)
        for k in range(1, K+1):
            params_best[2*(k-1)] = 2.0 * np.real(F_best[k]) / N
            params_best[2*(k-1)+1] = -2.0 * np.imag(F_best[k]) / N
        
        # Fine-tune with more iterations
        try:
            result = minimize(
                _objective,
                params_best,
                jac=_grad,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            h_candidate = _construct_from_fourier(result.x)
            sc = _max_overlap_fft(h_candidate)
            if sc < best_score:
                best_score = sc
                best_h = h_candidate.copy()
        except Exception:
            pass
    
    # Final validation
    if abs(best_h.sum() - N/2.0) > 1e-8:
        best_h = best_h * (N/2.0) / best_h.sum()
        best_h = np.clip(best_h, 0.0, 1.0)
    
    return best_h
