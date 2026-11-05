"""Rewritten program for third autocorrelation inequality optimization."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _fft_autoconv(h):
    """Compute full autoconvolution using FFT for speed."""
    n = len(h)
    size = 2 * n - 1
    fft_size = 1
    while fft_size < size:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    conv = np.fft.irfft(H * H, n=fft_size)[:size]
    return conv

def _compute_value(h):
    """Compute the exact objective value."""
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = _fft_autoconv(h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _grad_autoconv(h, weights):
    """
    Compute gradient of weighted autoconvolution sum with respect to h.
    weights has length 2N-1, corresponding to conv[0..2N-2].
    Returns gradient of length N.
    """
    n = len(h)
    size = 2 * n - 1
    fft_size = 1
    while fft_size < size:
        fft_size *= 2
    
    H = np.fft.rfft(h, n=fft_size)
    # Pad weights to fft_size
    w_padded = np.zeros(fft_size)
    w_padded[:len(weights)] = weights
    G = np.fft.rfft(w_padded, n=fft_size)
    
    # d(conv)/d(h) gives: grad_h = 2 * ifft(fft(h) * fft(weights_padded))
    # But we need to be careful about the real-valued constraint
    grad_h = 2.0 * np.fft.irfft(H * G, n=fft_size)[:n]
    return grad_h

def _soft_max_and_weights(conv, beta):
    """
    Compute soft-max value and weights (normalized exponentials).
    Returns (soft_max_val, weights) where weights sums to 1.
    """
    m = np.max(conv)
    shifted = beta * (conv - m)
    shifted = np.clip(shifted, -500, 500)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    
    if sum_exp < 1e-300:
        return m, np.zeros_like(conv)
    
    soft_max_val = m + np.log(sum_exp) / beta
    weights = exp_shifted / sum_exp
    return soft_max_val, weights

def _objective_soft(x, beta=100.0):
    """Soft-max objective: 2*N*softmax(conv) / (sum(h))^2"""
    h = x
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = _fft_autoconv(h)
    soft_max_val, _ = _soft_max_and_weights(conv, beta)
    return 2.0 * N * soft_max_val / (s * s)

def _objective_soft_grad(x, beta=100.0):
    """Soft-max objective with gradient."""
    h = x
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10, np.zeros_like(h)
    
    conv = _fft_autoconv(h)
    soft_max_val, weights = _soft_max_and_weights(conv, beta)
    
    # d(soft_max_val)/d(conv_k) = weights[k]
    # d(conv_k)/d(h_j) leads to: grad_conv_part = 2 * ifft(fft(h) * fft(weights))
    grad_h_conv = _grad_autoconv(h, weights)
    
    obj = 2.0 * N * soft_max_val / (s * s)
    
    # Chain rule
    d_obj_d_softmax = 2.0 * N / (s * s)
    d_obj_d_s = -4.0 * N * soft_max_val / (s * s * s)
    
    grad = d_obj_d_softmax * grad_h_conv + d_obj_d_s * np.ones_like(h)
    return obj, grad

def _spectral_penalty(h):
    """L4 norm of normalized power spectrum."""
    H = np.fft.rfft(h)
    power = np.abs(H) ** 2
    mean_power = np.mean(power)
    if mean_power < 1e-10:
        return 1e10
    normalized_power = power / mean_power
    return np.sum(normalized_power ** 2)

def _spectral_penalty_grad(h):
    """Gradient of spectral penalty."""
    H = np.fft.rfft(h)
    power = np.abs(H) ** 2
    mean_power = np.mean(power)
    if mean_power < 1e-10:
        return np.zeros_like(h)
    normalized_power = power / mean_power
    
    # d/dh sum((power/mean)^2)
    # This is complex; use numerical or skip gradient
    # For simplicity, return zero and let L-BFGS-B estimate
    return np.zeros_like(h)

def _combined_obj(x, beta=100.0, spectral_weight=0.01):
    """Combined objective: soft-max + spectral penalty."""
    obj_soft, _ = _objective_soft_grad(x, beta)
    obj_spec = _spectral_penalty(x)
    return obj_soft + spectral_weight * obj_spec

def _combined_grad(x, beta=100.0, spectral_weight=0.01):
    """Combined objective gradient."""
    obj_soft, grad_soft = _objective_soft_grad(x, beta)
    grad_spec = _spectral_penalty_grad(x)
    obj_spec = _spectral_penalty(x)
    return obj_soft + spectral_weight * obj_spec, grad_soft + spectral_weight * grad_spec

def _refine(h, beta, max_iter=200, spectral_weight=0.0):
    """Refine h using L-BFGS-B."""
    if spectral_weight > 0:
        res = minimize(_combined_obj, h, method='L-BFGS-B',
                      args=(beta, spectral_weight),
                      jac=_combined_grad,
                      options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8})
    else:
        res = minimize(_objective_soft, h, method='L-BFGS-B',
                      args=(beta,),
                      jac=_objective_soft_grad,
                      options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8})
    return res.x

def _unitary_initializations(rng, count=20):
    """Generate diverse initializations."""
    inits = []
    for _ in range(count):
        # Random with controlled mean
        h0 = rng.standard_normal(N)
        mean_target = rng.uniform(0.5, 2.0)
        h0 += mean_target - np.mean(h0)
        inits.append(h0)
    
    # Mixed cosines
    for _ in range(10):
        h0 = np.zeros(N)
        for k in range(rng.integers(2, 6)):
            freq = rng.integers(1, 10)
            phase = rng.random() * 2 * np.pi
            amp = rng.uniform(0.3, 1.0)
            t = np.linspace(0, 2*np.pi*freq, N, endpoint=False)
            h0 += amp * np.cos(t + phase)
        h0 += 1.0
        inits.append(h0)
    
    # Barker-like
    for _ in range(10):
        h0 = rng.choice([-1, 1], size=N).astype(float)
        h0 += 0.2
        inits.append(h0)
    
    return inits

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_h = np.ones(N)
    best_val = _compute_value(best_h)
    
    rng = np.random.default_rng(42)
    
    # Generate initializations
    inits = _unitary_initializations(rng, count=30)
    
    # Multi-stage beta annealing
    beta_schedule = [30.0, 60.0, 120.0, 250.0, 500.0]
    
    for init_idx, h0 in enumerate(inits):
        if time.time() > deadline - 30.0:
            break
        
        h_curr = h0.copy()
        
        # Progressive refinement with increasing beta
        for beta in beta_schedule:
            if time.time() > deadline - 30.0:
                break
            
            # Refine with soft-max objective
            h_curr = _refine(h_curr, beta=beta, max_iter=150)
            
            # Add spectral flattening in later stages
            if beta >= 120.0:
                h_curr = _refine(h_curr, beta=beta, max_iter=100, spectral_weight=0.01)
            
            val_curr = _compute_value(h_curr)
            if val_curr < best_val:
                best_val = val_curr
                best_h = h_curr.copy()
    
    # Extended spectral flattening phase
    if time.time() < deadline - 40.0:
        h_curr = best_h.copy()
        for phase in range(8):
            if time.time() > deadline - 30.0:
                break
            
            # Spectral flattening
            h_curr = _refine(h_curr, beta=200.0, max_iter=50, spectral_weight=0.02)
            
            # Peak reduction
            for beta in [150.0, 300.0]:
                if time.time() > deadline - 30.0:
                    break
                h_curr = _refine(h_curr, beta=beta, max_iter=100)
            
            val_curr = _compute_value(h_curr)
            if val_curr < best_val:
                best_val = val_curr
                best_h = h_curr.copy()
    
    # Final high-precision polish
    if time.time() < deadline - 15.0:
        for beta in [300.0, 600.0, 1000.0]:
            if time.time() > deadline - 15.0:
                break
            h_new = _refine(best_h, beta=beta, max_iter=500)
            val = _compute_value(h_new)
            if val < best_val:
                best_val = val
                best_h = h_new.copy()
    
    # Ensure validity
    if abs(np.sum(best_h)) < 1e-10:
        best_h = np.ones(N)
        best_val = _compute_value(best_h)
        
    return best_h
# EVOLVE-BLOCK-END
