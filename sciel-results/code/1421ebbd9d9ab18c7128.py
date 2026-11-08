import time
import numpy as np
from scipy.optimize import minimize

N = 100
DEADLINE = 145.0  # Safety margin before 150s kill

def _h_from_coeffs(coeffs):
    """
    Reconstruct h from Fourier coefficients.
    coeffs[0] is the DC component (real), coeffs[1:] are complex coefficients for k=1..N/2-1,
    and we enforce h[0..N-1] real, in [0,1], sum = N/2.
    """
    # coeffs[0] = c0 (DC), coeffs[1:] = c1..c_{N/2-1} (complex)
    c0 = coeffs[0]
    c_complex = coeffs[1:]
    
    # Build full spectrum
    spectrum = np.zeros(N, dtype=complex)
    spectrum[0] = c0
    for k in range(1, N//2):
        spectrum[k] = c_complex[k-1]
        spectrum[N-k] = np.conj(c_complex[k-1])
    
    # If N is even, handle Nyquist frequency (k=N/2)
    if N % 2 == 0:
        # Nyquist frequency must be real
        spectrum[N//2] = np.real(c_complex[N//2-1])
    
    h = np.fft.ifft(spectrum).real
    return h

def _max_overlap(h):
    """
    Compute the maximum circular overlap of h with 1-h.
    Overlap at shift k: sum_i h[i] * (1 - h[(i+k) % N])
    """
    N = len(h)
    overlaps = np.zeros(N)
    for k in range(N):
        shifted = np.roll(h, k)
        overlaps[k] = np.sum(h * (1 - shifted))
    return np.max(overlaps)

def _objective_and_grad(coeffs):
    """
    Objective: maximize sum(h) = N/2 constraint + minimize max overlap.
    We use a smooth surrogate: soft-max of overlaps.
    """
    h = _h_from_coeffs(coeffs)
    
    # Penalty for constraints: h in [0,1], sum(h) = N/2
    sum_h = np.sum(h)
    sum_penalty = 1000.0 * (sum_h - N/2)**2
    
    # h in [0,1] penalty
    below = np.maximum(0, -h)
    above = np.maximum(0, h - 1)
    bound_penalty = 1000.0 * (np.sum(below**2) + np.sum(above**2))
    
    # Compute overlaps
    N = len(h)
    overlaps = np.zeros(N)
    for k in range(N):
        shifted = np.roll(h, k)
        overlaps[k] = np.sum(h * (1 - shifted))
    
    # Soft-max surrogate (smooth approximation to max)
    tau = 0.1
    log_sum_exp = np.log(np.sum(np.exp(overlaps / tau))) * tau
    soft_max = log_sum_exp
    
    obj = soft_max + sum_penalty + bound_penalty
    
    # Gradient via finite differences (for simplicity)
    grad = np.zeros_like(coeffs)
    eps = 1e-7
    for i in range(len(coeffs)):
        coeffs_plus = coeffs.copy()
        coeffs_plus[i] += eps
        h_plus = _h_from_coeffs(coeffs_plus)
        
        sum_h_plus = np.sum(h_plus)
        sum_pen_plus = 1000.0 * (sum_h_plus - N/2)**2
        
        below_plus = np.maximum(0, -h_plus)
        above_plus = np.maximum(0, h_plus - 1)
        bound_pen_plus = 1000.0 * (np.sum(below_plus**2) + np.sum(above_plus**2))
        
        overlaps_plus = np.zeros(N)
        for k in range(N):
            shifted_plus = np.roll(h_plus, k)
            overlaps_plus[k] = np.sum(h_plus * (1 - shifted_plus))
        
        soft_max_plus = np.log(np.sum(np.exp(overlaps_plus / tau))) * tau
        obj_plus = soft_max_plus + sum_pen_plus + bound_pen_plus
        
        grad[i] = (obj_plus - obj) / eps
    
    return obj, grad

def construct():
    """
    Returns a sequence of step heights h in [0, 1] whose sum is exactly N/2.
    Uses Fourier-domain optimization to minimize maximum circular overlap.
    """
    start_time = time.time()
    
    # Initial guess: h = 0.5 everywhere (corresponds to c0 = N/2, all other coeffs = 0)
    coeffs0 = np.zeros(N//2)
    coeffs0[0] = N/2.0  # DC component
    
    # Try L-BFGS-B optimization
    best_h = np.full(N, 0.5)
    best_score = _max_overlap(best_h)
    
    try:
        result = minimize(
            _objective_and_grad,
            coeffs0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
        )
        
        if result.success or time.time() - start_time < DEADLINE:
            h_opt = _h_from_coeffs(result.x)
            # Clamp to [0,1] and adjust sum
            h_opt = np.clip(h_opt, 0, 1)
            # Adjust to make sum exactly N/2
            diff = N/2.0 - np.sum(h_opt)
            h_opt += diff / N
            h_opt = np.clip(h_opt, 0, 1)
            
            score_opt = _max_overlap(h_opt)
            if score_opt < best_score:
                best_h = h_opt
                best_score = score_opt
    except Exception:
        pass
    
    # Check deadline
    if time.time() - start_time > DEADLINE:
        return best_h
    
    # Final polish: try to improve with a tighter optimization
    try:
        coeffs_init = np.zeros(N//2)
        # Estimate initial coeffs from best_h
        spectrum = np.fft.fft(best_h)
        coeffs_init[0] = np.real(spectrum[0]) / N
        for k in range(1, N//2):
            coeffs_init[k] = spectrum[k] / N
        if N % 2 == 0:
            coeffs_init[N//2-1] = np.real(spectrum[N//2]) / N
        
        result2 = minimize(
            _objective_and_grad,
            coeffs_init,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
        )
        
        if result2.success or time.time() - start_time < DEADLINE:
            h_opt2 = _h_from_coeffs(result2.x)
            h_opt2 = np.clip(h_opt2, 0, 1)
            diff2 = N/2.0 - np.sum(h_opt2)
            h_opt2 += diff2 / N
            h_opt2 = np.clip(h_opt2, 0, 1)
            
            score_opt2 = _max_overlap(h_opt2)
            if score_opt2 < best_score:
                best_h = h_opt2
                best_score = score_opt2
    except Exception:
        pass
    
    # Ensure exact sum
    diff_final = N/2.0 - np.sum(best_h)
    best_h += diff_final / N
    best_h = np.clip(best_h, 0, 1)
    
    return best_h
