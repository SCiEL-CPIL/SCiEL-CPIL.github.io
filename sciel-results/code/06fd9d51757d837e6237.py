"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 600
DEADLINE = 145.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START
def _autoconvolve_max_ratio(h):
    """Compute max_x (f*f)(x) / (∫f)^2 for step function h on N equal intervals."""
    h = np.asarray(h, dtype=np.float64)
    integral = np.sum(h)
    if integral <= 1e-12:
        return 1e10
    # Compute autoconvolution via FFT
    n_fft = 2 * N + 1
    h_padded = np.zeros(n_fft)
    h_padded[:N] = h
    conv = np.real(np.fft.ifft(np.fft.fft(h_padded, n=2*n_fft) ** 2))[:2*N-1]
    
    w = 1.0 / (2.0 * N)
    max_conv = np.max(conv)
    integral = w * np.sum(h)
    if integral <= 1e-12:
        return 1e10
    return max_conv * w / (integral * integral)


def _smooth_max_surrogate(conv, t):
    """Log-sum-exp smooth approximation to max(conv)."""
    m = np.max(conv)
    if m < -1e300:
        return 0.0
    shifted = conv - m
    # Clip to avoid overflow
    shifted = np.clip(shifted, -100, 100)
    return m + t * np.log(np.sum(np.exp(shifted / t)))


def _objective_and_grad(h_vec, t):
    """
    Objective: minimize smooth max of autoconvolution, with h normalized to sum=1.
    Returns (value, gradient) with respect to h_vec (before normalization).
    """
    h_vec = np.maximum(h_vec, 0.0)
    s = np.sum(h_vec)
    if s < 1e-12:
        return 1e10, np.zeros_like(h_vec)
    h = h_vec / s
    
    # Compute convolution
    n_fft = 2 * N + 1
    h_padded = np.zeros(n_fft)
    h_padded[:N] = h
    F = np.fft.fft(h_padded, n=2*n_fft)
    conv = np.real(np.fft.ifft(F ** 2))[:2*N-1]
    
    # Smooth max
    m = np.max(conv)
    shifted = conv - m
    shifted = np.clip(shifted, -100, 100)
    exp_vals = np.exp(shifted / t)
    sum_exp = np.sum(exp_vals)
    val = m + t * np.log(sum_exp)
    
    # Gradient of smooth max w.r.t. conv
    # d(smooth_max)/d(conv_k) = exp((conv_k - m)/t) / sum_exp
    grad_conv = exp_vals / sum_exp
    
    # Now chain rule: conv_k = sum_i h_i * h_{k-i}
    # d(conv_k)/d(h_j) = h_{k-j} + h_j (for valid indices, with symmetry)
    # But we need d(val)/d(h_j) = sum_k grad_conv_k * d(conv_k)/d(h_j)
    # 
    # This is equivalent to: gradient_j = sum_k grad_conv_k * (h_{k-j} + h_j)
    # = (grad_conv * h)_j + (grad_conv * h)_{reversed}_j ... let me think more carefully.
    #
    # conv_k = sum_i h_i * h_{k-i} where h is defined on [0, N-1]
    # d(conv_k)/d(h_j) = h_{k-j} (if 0 <= k-j < N) + h_j (if 0 <= k-j < N, from the other term)
    # Wait, conv_k = sum_i h_i h_{k-i}. 
    # d(conv_k)/d(h_j) = h_{k-j} (from i=j) + h_j (from k-i=j, i.e., i=k-j)
    # But these are the same term if we're careful. Actually:
    # conv_k = sum_{i=max(0,k-N+1)}^{min(k,N-1)} h_i * h_{k-i}
    # d(conv_k)/d(h_j) = h_{k-j} if 0<=k-j<N, plus h_j if 0<=k-j<N (from the symmetric term)
    # Actually no. Let me just use: d(conv_k)/d(h_j) = h_{k-j} + h_j * delta... 
    # 
    # Simpler: use the fact that d(conv)/d(h) = conv(grad_conv, h) + conv(grad_conv_flipped, h)
    # where conv here means cross-correlation.
    #
    # Actually, the cleanest way: 
    # val = smooth_max(conv), where conv = autoconv(h)
    # d(val)/d(h_j) = sum_k grad_conv_k * d(conv_k)/d(h_j)
    # d(conv_k)/d(h_j) = h_{k-j} + h_j (with appropriate zero-padding)
    # 
    # Let me compute this via another FFT.
    # sum_k grad_conv_k * h_{k-j} = (cross_correlation of grad_conv and h) evaluated at j
    # sum_k grad_conv_k * h_j = h_j * sum_k grad_conv_k = h_j * sum_exp (since grad_conv sums to 1)
    
    # Cross-correlation: sum_k grad_conv_k * h_{k-j}
    # This is like convolving grad_conv with reversed h, or equivalently
    # sum_k grad_conv_k * h_{k-j} = (grad_conv * h_flipped)_{N-1-j} ... let me just do it directly.
    
    # For each j, sum_k grad_conv_k * h_{k-j} where k-j is in [0,N-1]
    # = sum_{i=0}^{N-1} grad_conv_{i+j} * h_i (with grad_conv padded to length 2N-1)
    # This is a cross-correlation.
    
    # Use FFT for speed
    gc_padded = np.zeros(2 * N - 1)
    gc_padded[:len(grad_conv)] = grad_conv
    # Cross-correlation: sum_k gc[k] * h[k-j] for each j
    # = ifft(fft(gc) * conj(fft(h_padded_to_same_size)))
    h_pad_cc = np.zeros(2 * N - 1)
    h_pad_cc[:N] = h
    cc = np.real(np.fft.ifft(np.fft.fft(gc_padded) * np.conj(np.fft.fft(h_pad_cc))))[:N]
    
    # The other term: h_j * sum_k grad_conv_k
    sum_grad = np.sum(grad_conv)  # should be ~1
    
    grad_h = cc + h * sum_grad
    
    # Now account for normalization: h = h_vec / s
    # d(val)/d(h_vec_j) = (1/s) * d(val)/d(h_j) - (h_j/s^2) * sum_i d(val)/d(h_i)
    # = (1/s) * grad_h_j - (h_j/s^2) * dot(grad_h, h)
    
    dot_grad_h_h = np.dot(grad_h, h)
    grad_hvec = (grad_h / s) - (h * dot_grad_h_h) / (s * s)
    
    return val, grad_hvec


def construct():
    start_time = time.time()
    
    # Start with uniform
    h = np.ones(N)
    best_h = h.copy()
    best_val = _autoconvolve_max_ratio(h)
    
    # Normalize for optimization: sum(h) = 1
    h_norm = h / np.sum(h)
    
    bounds = [(0, None)] * N
    
    # Temperature annealing schedule
    t_start = 1.0 / N * 5  # start with loose approximation
    t_end = 1.0 / N * 0.1   # end with tight approximation
    
    # Phase 1: Coarse optimization with higher temperature
    t = t_start
    h_cur = h_norm.copy()
    
    n_phases = 5
    for phase in range(n_phases):
        if time.time() - start_time > DEADLINE - 30:
            break
        
        t = t_start * (t_end / t_start) ** (phase / max(n_phases - 1, 1))
        
        def obj(hf):
            v, _ = _objective_and_grad(hf, t)
            return v
        
        def grad(hf):
            _, g = _objective_and_grad(hf, t)
            return g
        
        maxiter = max(200, 800 - phase * 100)
        result = minimize(obj, h_cur, method='L-BFGS-B', jac=grad,
                          bounds=bounds, options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10})
        
        if result.x is not None:
            h_new = np.maximum(result.x, 0.0)
            s = np.sum(h_new)
            if s > 1e-12:
                h_new = h_new / s
                val = _autoconvolve_max_ratio(h_new)
                if val < best_val:
                    best_h = h_new.copy()
                    best_val = val
                h_cur = h_new.copy()
    
    # Phase 2: Fine polish with very low temperature
    if time.time() - start_time < DEADLINE - 15:
        t = t_end
        h_cur = best_h / np.sum(best_h)
        
        def obj(hf):
            v, _ = _objective_and_grad(hf, t)
            return v
        
        def grad(hf):
            _, g = _objective_and_grad(hf, t)
            return g
        
        result = minimize(obj, h_cur, method='L-BFGS-B', jac=grad,
                          bounds=bounds, options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12})
        
        if result.x is not None:
            h_new = np.maximum(result.x, 0.0)
            s = np.sum(h_new)
            if s > 1e-12:
                h_new = h_new / s
                val = _autoconvolve_max_ratio(h_new)
                if val < best_val:
                    best_h = h_new.copy()
                    best_val = val
    
    # Try alternative initializations if time permits
    if time.time() - start_time < DEADLINE - 20:
        x = np.linspace(-0.5, 0.5, N)
        inits = []
        
        # Bimodal
        h_bimodal = np.exp(-60 * (x - 0.22)**2) + np.exp(-60 * (x + 0.22)**2)
        inits.append(h_bimodal)
        
        # Triangle
        h_tri = np.maximum(0, 1 - 2.5 * np.abs(x))
        inits.append(h_tri)
        
        # Wide bump
        h_wide = np.exp(-15 * x**2)
        inits.append(h_wide)
        
        for h_init in inits:
            if time.time() - start_time > DEADLINE - 12:
                break
            
            h_init = np.maximum(h_init, 0.0)
            s = np.sum(h_init)
            if s < 1e-12:
                continue
            h_init = h_init / s
            
            # Quick optimization with moderate temperature
            t = 0.5 / N
            
            def obj_init(hf):
                v, _ = _objective_and_grad(hf, t)
                return v
            
            def grad_init(hf):
                _, g = _objective_and_grad(hf, t)
                return g
            
            result = minimize(obj_init, h_init, method='L-BFGS-B', jac=grad_init,
                              bounds=bounds, options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
            
            if result.x is not None:
                h_new = np.maximum(result.x, 0.0)
                s = np.sum(h_new)
                if s > 1e-12:
                    h_new = h_new / s
                    val = _autoconvolve_max_ratio(h_new)
                    if val < best_val:
                        best_h = h_new.copy()
                        best_val = val
    
    return best_h.tolist()
# EVOLVE-BLOCK-END
