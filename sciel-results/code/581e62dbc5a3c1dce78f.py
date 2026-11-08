"""
Optimized approach: Use FFT-based overlap computation with multi-start L-BFGS-B
to find non-uniform step heights that minimize maximum overlap below 0.5.
"""
import numpy as np
from scipy.optimize import minimize
import time

N = 100
DEADLINE = 120.0  # seconds, with safety margin from 150s kill
START_TIME = None

def _compute_overlap_fft(h):
    """Compute overlap O(k) = sum_i h[i] * (1 - h[(i+k) % N]) for all shifts k."""
    n = len(h)
    # O(k) = sum_i h[i] * (1 - h[(i+k) % n])
    # = sum_i h[i] - sum_i h[i] * h[(i+k) % n]
    # = 1.0 - cross_correlation(h, h)[k]
    # Using FFT for circular cross-correlation
    H = np.fft.fft(h)
    # circular cross-correlation: sum_i h[i] * h[(i+k) % n]
    # This is equivalent to ifft(fft(h) * conj(fft(h))) shifted
    # Actually: sum_i h[i] * h[(i+k)%n] = ifft(conj(fft(h)) * fft(h))[k] 
    # Let me think more carefully.
    # cross_corr[k] = sum_i h[i] * h[(i+k) % n]
    # In FFT domain: ifft(fft(h) * conj(fft(h))) gives sum_i h[(i+k)%n] * conj(h[i])
    # For real h: sum_i h[(i+k)%n] * h[i]
    # That's what we want: sum_i h[i] * h[(i+k)%n]
    corr = np.real(np.fft.ifft(np.fft.fft(h) * np.conj(np.fft.fft(h))))
    # O(k) = 1.0 - corr[k] (since sum h[i] = 1.0)
    return 1.0 - corr

def _max_overlap(h):
    """Compute maximum overlap value."""
    overlaps = _compute_overlap_fft(h)
    return np.max(overlaps)

def _objective(params):
    """Objective: maximize t subject to t <= O(k) for all k, using epigraph with smooth approximation."""
    # We want to minimize max_k O(k)
    # Use soft-max approximation: (1/tau) * log(sum exp(tau * O(k)))
    h = params
    overlaps = _compute_overlap_fft(h)
    # Soft-max with temperature
    tau = 10.0
    max_val = np.max(overlaps)
    # Numerically stable soft-max
    shifted = tau * (overlaps - max_val)
    log_sum = max_val + (1.0/tau) * np.log(np.sum(np.exp(shifted)))
    return log_sum

def _gradient(params):
    """Gradient of the soft-max objective."""
    h = params
    overlaps = _compute_overlap_fft(h)
    tau = 10.0
    max_val = np.max(overlaps)
    shifted = tau * (overlaps - max_val)
    weights = np.exp(shifted) / np.sum(np.exp(shifted))
    
    # Gradient of O(k) w.r.t. h[i]:
    # O(k) = 1 - sum_j h[j] * h[(j+k)%n]
    # dO(k)/dh[i] = -sum_j delta(j,i)*h[(j+k)%n] - sum_j h[j]*delta((j+k)%n, i)
    # = -h[(i+k)%n] - sum_j h[j] * delta((j+k)%n, i)
    # = -h[(i+k)%n] - h[(i-k)%n]
    
    # So dO(k)/dh[i] = -(h[(i+k)%n] + h[(i-k)%n])
    # grad_obj[i] = sum_k weights[k] * dO(k)/dh[i]
    # = -sum_k weights[k] * (h[(i+k)%n] + h[(i-k)%n])
    
    n = len(h)
    grad = np.zeros(n)
    for i in range(n):
        for k in range(n):
            grad[i] -= weights[k] * (h[(i+k) % n] + h[(i-k) % n])
    
    return grad

def _objective_and_grad(params):
    """Combined objective and gradient for L-BFGS-B."""
    val = _objective(params)
    grad = _gradient(params)
    return val, grad

def construct():
    global START_TIME
    if START_TIME is None:
        START_TIME = time.time()
    
    # Start with uniform solution as baseline
    best_h = np.full(N, 0.5)
    best_score = _max_overlap(best_h)
    
    # Multi-start optimization
    rng = np.random.RandomState(42)
    n_starts = 20
    max_iter = 200
    
    # Precompute FFT of basis for faster gradient? No, let's use analytic gradient.
    
    for start_idx in range(n_starts):
        # Check deadline
        if time.time() - START_TIME > DEADLINE - 5:
            break
        
        # Generate initial guess: uniform + random Fourier modes
        h_init = np.full(N, 0.5)
        # Add small random perturbations in Fourier space
        freqs = np.fft.fftfreq(N)
        # Skip DC component (index 0)
        perturbation = np.zeros(N, dtype=complex)
        for f_idx in range(1, N//2 + 1):
            if freqs[f_idx] != 0:
                perturbation[f_idx] = rng.randn() * 0.1
                perturbation[N - f_idx] = np.conj(perturbation[f_idx])
        h_init += np.real(np.fft.ifft(perturbation))
        h_init = np.clip(h_init, 0.0, 1.0)
        
        # Normalize to have integral 1 (sum = N/2)
        h_init = h_init / np.sum(h_init) * (N / 2.0)
        h_init = np.clip(h_init, 0.0, 1.0)
        
        # Optimize
        try:
            result = minimize(
                _objective,
                h_init,
                method='L-BFGS-B',
                jac=_gradient,
                bounds=[(0.0, 1.0)] * N,
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < best_score:
                h_opt = result.x
                h_opt = np.clip(h_opt, 0.0, 1.0)
                # Normalize
                h_opt = h_opt / np.sum(h_opt) * (N / 2.0)
                h_opt = np.clip(h_opt, 0.0, 1.0)
                
                score = _max_overlap(h_opt)
                if score < best_score:
                    best_score = score
                    best_h = h_opt
        except Exception:
            continue
    
    # Final polish: use higher temperature soft-max for finer optimization
    if time.time() - START_TIME < DEADLINE - 10:
        tau_old = 10.0
        # Re-optimize with higher tau for sharper max approximation
        global _tau_val
        _tau_val = 50.0
        
        try:
            result = minimize(
                _objective,
                best_h,
                method='L-BFGS-B',
                jac=_gradient,
                bounds=[(0.0, 1.0)] * N,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if result.fun < best_score:
                h_opt = np.clip(result.x, 0.0, 1.0)
                h_opt = h_opt / np.sum(h_opt) * (N / 2.0)
                h_opt = np.clip(h_opt, 0.0, 1.0)
                score = _max_overlap(h_opt)
                if score < best_score:
                    best_score = score
                    best_h = h_opt
        except Exception:
            pass
    
    return best_h

# Global for tau control
_tau_val = 10.0
