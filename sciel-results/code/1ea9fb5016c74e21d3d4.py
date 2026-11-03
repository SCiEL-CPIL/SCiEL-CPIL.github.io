import numpy as np
import time
import sys

N = 600
W = 1.0 / 1200.0

# Precompute fine grid for fast evaluation
FINE_N = 12000
X_FINE = np.linspace(-0.25, 0.25, FINE_N, endpoint=False)
DX = 0.5 / FINE_N
CELL = np.clip(((X_FINE + 0.25) / W).astype(np.int64), 0, N - 1)
N_FFT = 1 << (2 * FINE_N - 1).bit_length()

def ratio_from_h(h):
    """Compute the normalized maximum of the autoconvolution."""
    h = np.asarray(h, dtype=np.float64)
    if h.size != N:
        h = np.resize(h, N)
    h = np.clip(h, 0.0, None)
    s = h.sum()
    if s < 1e-15:
        return float('inf')
    
    f_fine = h[CELL]
    
    F = np.fft.rfft(f_fine, N_FFT)
    conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1]
    conv *= DX
    max_conv = np.max(conv)
    l2_sq = np.sum(f_fine ** 2) * DX
    if l2_sq < 1e-15:
        return float('inf')
    return max_conv / l2_sq

def compute_grad(h):
    """
    Compute the gradient of the objective J(h) = max_k conv_k / ||h||_2^2.
    Uses a soft-max approximation for differentiability:
    M_soft = sum(conv * exp(conv/tau)) / sum(exp(conv/tau))
    grad_M = sum(conv * exp(conv/tau) * grad_conv) / sum(exp(conv/tau))
    """
    h = np.clip(h, 0.0, None)
    l2_sq = np.sum(h**2)
    if l2_sq < 1e-15:
        return np.zeros_like(h), 1.0
    
    f_fine = h[CELL]
    F = np.fft.rfft(f_fine, N_FFT)
    conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1]
    conv *= DX
    
    # Soft max temperature
    tau = 0.01 
    # Avoid overflow in exp
    conv_shifted = conv - np.max(conv)
    weights = np.exp(conv_shifted / tau)
    sum_weights = np.sum(weights)
    
    if sum_weights < 1e-15:
        return np.zeros_like(h), 1.0
        
    M = (np.sum(conv * weights) / sum_weights) / l2_sq
    
    # Gradient of M with respect to conv
    # dM/dconv = (weights * (l2_sq - M*conv) / M) / (sum_weights * l2_sq) ? No.
    # Let S = sum(conv_i * w_i). M = S / (sum_w * l2_sq).
    # dM/dconv_k = w_k / (sum_w * l2_sq).
    # Wait, M = (sum c_i w_i) / (W_sum * L).
    # dM/dc_k = w_k / (W_sum * L).
    
    grad_conv = weights / (sum_weights * l2_sq)
    
    # Gradient of conv_k with respect to h_j
    # conv_k = sum_{i,l: i+l=k} h_i h_l (discrete approx)
    # d(conv_k)/dh_j = sum_{l: j+l=k} h_l + sum_{i: i+j=k} h_i = 2 * (h * h)_k (with boundary care)
    # In the FFT domain, conv is the circular convolution of f_fine with itself.
    # Since we are working on the fine grid, let's compute the gradient on the fine grid first.
    
    # f_fine is a step function sampled on X_FINE.
    # conv_fine = f_fine * f_fine (circular).
    # d(conv_fine)_m / d(f_fine)_j = (f_fine * f_fine)_m (circular) = 2 * (f_fine shifted by m-j).
    # Actually, d/dx_j (sum_i x_i x_{k-i}) = x_{k-j} + x_{k-j} = 2 x_{k-j}.
    # So grad_f_fine[m] = 2 * sum_k grad_conv[k] * conv_kernel[k-m]
    # where conv_kernel is the autoconvolution of f_fine? No.
    # Let C = f * f. d C_k / d f_j = f_{k-j} + f_{k-j} = 2 f_{k-j} (indices mod N_fine).
    # So grad_f = 2 * (grad_conv * f) (circular convolution).
    
    grad_f = np.fft.irfft(np.fft.rfft(grad_conv, N_FFT) * np.fft.rfft(f_fine, N_FFT), N_FFT)[:2*FINE_N-1]
    # Wait, the gradient of the circular convolution c_k = sum_i f_i f_{k-i} wrt f_j is sum_k (d c_k / d f_j) g_k = sum_k (f_{k-j} + f_{k-j}) g_k = 2 sum_k f_{k-j} g_k.
    # This is 2 * (f * g) (circular).
    
    # We need to map grad_f back to h.
    # h is the step function. f_fine[j] = h[CELL[j]].
    # d J / d h_m = sum_{j: CELL[j]==m} d J / d f_fine[j].
    
    grad_h = np.zeros(N)
    np.add.at(grad_h, CELL, grad_f)
    
    # Apply quotient rule for M = Num / Den
    # dM/dh = (Den * dNum/dh - Num * dDen/dh) / Den^2
    # dNum/dh = grad_h (computed above, which is d(M_soft * L)/dh? No, M_soft was defined as S/Wsum. 
    # Let's stick to the exact derivative of the soft-max approximation of max(conv)/L.
    
    # Let U = sum conv_i exp(conv_i/tau). V = sum exp(conv_i/tau).
    # M_soft = U/V.
    # dM_soft/dconv_k = (exp(conv_k/tau)/V) * (1 - M_soft * tau) ? No.
    # d(U/V)/dc_k = (dU/dc_k * V - U * dV/dc_k) / V^2
    # dU/dc_k = exp(conv_k/tau)/tau * (1 + conv_k/tau * 0) ? No, d/dc (c * e^{c/tau}) = e^{c/tau} + c/tau e^{c/tau} = e^{c/tau}(1 + c/tau).
    # dV/dc_k = e^{c_k/tau} / tau.
    # This is getting complicated. Let's use the standard softmax gradient.
    # Let p_k = exp(c_k/tau) / V.
    # M_soft = sum p_k c_k.
    # d M_soft / d c_k = p_k (1 - p_k tau) ? No.
    # d/dc_k (sum p_i c_i) = p_k + sum_i c_i dp_i/dc_k.
    # dp_i/dc_k = p_i (delta_{ik}/tau - p_k/tau).
    # sum_i c_i dp_i/dc_k = p_k (c_k/tau - sum_i c_i p_i / tau) = p_k (c_k - M_soft)/tau.
    # So d M_soft / d c_k = p_k + p_k (c_k - M_soft)/tau = p_k (1 + (c_k - M_soft)/tau).
    
    # Let's recompute grad_conv with this formula.
    p = weights / sum_weights
    # M_soft_val = np.sum(p * conv)
    # dM_soft_dc = p * (1 + (conv - M_soft_val)/tau)
    
    M_soft_val = np.sum(p * conv)
    dM_soft_dc = p * (1 + (conv - M_soft_val)/tau)
    
    # Now, M = M_soft / L.
    # dM/dh = (L * dM_soft/dh - M_soft * dL/dh) / L^2
    # dM_soft/dh is obtained by convolving dM_soft_dc with f_fine (circular) and mapping to h.
    
    # dM_soft_dc is the gradient of the numerator (approx max) wrt conv.
    # grad_Msoft_f = 2 * (dM_soft_dc * f_fine) (circular)
    grad_Msoft_f = np.fft.irfft(np.fft.rfft(dM_soft_dc, N_FFT) * np.fft.rfft(f_fine, N_FFT), N_FFT)[:2*FINE_N-1]
    
    grad_Msoft_h = np.zeros(N)
    np.add.at(grad_Msoft_h, CELL, grad_Msoft_f)
    
    # dL/dh = 2 * h
    dL_dh = 2 * h
    
    # M = M_soft_val / L
    M = M_soft_val / l2_sq
    
    # dM/dh = (l2_sq * grad_Msoft_h - M_soft_val * dL_dh) / l2_sq^2
    grad_M = (l2_sq * grad_Msoft_h - M_soft_val * dL_dh) / (l2_sq * l2_sq)
    
    return grad_M, M

def construct():
    deadline = time.time() + 145.0
    np.random.seed(42)
    
    best_h = np.ones(N, dtype=np.float64)
    best_val = ratio_from_h(best_h)
    
    # Initialization strategies
    inits = []
    
    # 1. Uniform
    inits.append(np.ones(N))
    
    # 2. Gaussian bump
    x = np.linspace(-0.25, 0.25, N)
    for sigma in [0.05, 0.1, 0.2]:
        g = np.exp(-0.5 * (x/sigma)**2)
        inits.append(g)
        
    # 3. Random smooth
    for _ in range(5):
        # Low frequency noise
        freqs = np.fft.rfftfreq(N)
        noise = np.random.randn(N)
        # Apply low pass filter
        mask = freqs < 10
        noise_f = np.fft.rfft(noise)
        noise_f[~mask] = 0
        noise_filtered = np.fft.irfft(noise_f, N)
        # Make positive
        noise_filtered = np.abs(noise_filtered) + 0.1
        inits.append(noise_filtered)
        
    # 4. Asymmetric
    for shift in [-0.1, 0.1]:
        g = np.exp(-0.5 * ((x-shift)/0.1)**2)
        inits.append(g)

    for h0 in inits:
        if time.time() > deadline - 10:
            break
            
        h = h0.copy()
        # Normalize to unit L2
        norm = np.linalg.norm(h)
        if norm > 1e-15:
            h /= norm
            
        # Projected Gradient Descent
        lr = 0.01
        momentum = 0.9
        v = np.zeros_like(h)
        
        for it in range(2000):
            if time.time() > deadline - 2:
                break
                
            grad, val = compute_grad(h)
            
            # Momentum
            v = momentum * v - lr * grad
            h_new = h + v
            
            # Project to non-negative
            h_new = np.clip(h_new, 0, None)
            
            # Normalize to unit L2 (scale invariant)
            norm_new = np.linalg.norm(h_new)
            if norm_new < 1e-15:
                break
            h_new /= norm_new
            
            h = h_new
            
            # Check convergence
            if it % 100 == 0:
                exact_val = ratio_from_h(h)
                if exact_val < best_val:
                    best_val = exact_val
                    best_h = h.copy()
                    
        # Final check
        final_val = ratio_from_h(h)
        if final_val < best_val:
            best_val = final_val
            best_h = h.copy()

    return list(best_h)
