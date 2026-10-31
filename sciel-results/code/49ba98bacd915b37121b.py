import numpy as np
import time
import random

N = 600
DEADLINE = 140

def construct():
    start_time = time.time()

    def time_left():
        return DEADLINE - (time.time() - start_time)

    def compute_autoconv_max(h):
        """Compute max of autoconvolution of h (assumed sum=1)."""
        size = 2 * N
        h_pad = np.zeros(size)
        h_pad[:N] = h
        H = np.fft.fft(h_pad, n=size)
        conv = np.fft.ifft(H * H).real
        return np.max(conv)

    def smooth_max_autoconv(h, beta=50.0):
        """Compute smooth max (log-sum-exp) of autoconvolution of h."""
        size = 2 * N
        h_pad = np.zeros(size)
        h_pad[:N] = h
        H = np.fft.fft(h_pad, n=size)
        conv = np.fft.ifft(H * H).real
        m = np.max(conv)
        if m < 1e-15:
            return 0.0
        exp_conv = np.exp((conv - m) * beta)
        return m + np.log(np.sum(exp_conv)) / beta

    def compute_autoconv(h):
        """Compute full autoconvolution of h."""
        size = 2 * N
        h_pad = np.zeros(size)
        h_pad[:N] = h
        H = np.fft.fft(h_pad, n=size)
        conv = np.fft.ifft(H * H).real
        return conv

    # Spectral approach: represent h as squared magnitude of complex sequence with constant-amplitude DFT
    # h[n] = |z[n]|^2 where z has DFT Z[k] = A * exp(i*theta[k])
    # We optimize the phases theta to minimize max autoconvolution
    
    # First, get a good initial h from a few parametric families
    i = np.arange(N, dtype=float)
    center = (N - 1) / 2.0
    
    # Try a few simple shapes to get a starting point
    best_h = None
    best_val = np.inf
    
    # Uniform
    h_uniform = np.ones(N) / N
    val_uniform = compute_autoconv_max(h_uniform)
    best_h = h_uniform.copy()
    best_val = val_uniform
    
    # Bimodal
    for sigma in [N/8, N/6]:
        for alpha in [0.2, 0.3]:
            pos1 = alpha * N
            pos2 = (1 - alpha) * N
            g1 = np.exp(-((i - pos1) ** 2) / (2 * sigma ** 2))
            g2 = np.exp(-((i - pos2) ** 2) / (2 * sigma ** 2))
            h_bimodal = g1 + g2
            h_bimodal = h_bimodal / np.sum(h_bimodal)
            val_bimodal = compute_autoconv_max(h_bimodal)
            if val_bimodal < best_val:
                best_val = val_bimodal
                best_h = h_bimodal.copy()

    # Now use spectral approach
    # Compute DFT of best_h, get its amplitude and phase
    size = 2 * N
    h_pad = np.zeros(size)
    h_pad[:N] = best_h
    H = np.fft.fft(h_pad, n=size)
    
    # The autoconvolution is ifft(H * H)
    # We want to modify H to reduce the peak of ifft(H*H)
    # Strategy: project H onto a manifold of constant-amplitude spectra
    
    # Get amplitude and phase of H
    amp_H = np.abs(H)
    phase_H = np.angle(H)
    
    # The constraint is that H[k] * H[-k] should be real and positive (for h to be real)
    # And we want to minimize max of ifft(H*H)
    
    # Use gradient descent on the phase of H
    # The autoconvolution is c[m] = ifft(H * H)[m]
    # Gradient of max(c) w.r.t. phase is complex
    
    # Let's work with the real-valued h directly but use spectral insights
    # Actually, let's try a different approach: optimize h directly with spectral regularization
    
    # Spectral flatness: we want H to be as flat as possible in magnitude
    # This corresponds to h being close to a Dirac comb in frequency domain
    
    # Alternative: use the fact that for a given h, the autoconvolution peak is related to the L2 norm of H
    # By Parseval, sum(c[m]^2) = (1/size) * sum(|H[k]|^4)
    # But we want to minimize max(c[m]), not sum(c[m]^2)
    
    # Let's use iterative spectral projection
    # Start with current h, compute H, then modify H to reduce peak of ifft(H*H)
    
    current_h = best_h.copy()
    current_val = best_val
    
    # Iterative spectral optimization
    for iteration in range(1000):
        if time_left() < 15:
            break
        
        # Compute current autoconvolution
        h_pad = np.zeros(size)
        h_pad[:N] = current_h
        H = np.fft.fft(h_pad, n=size)
        conv = np.fft.ifft(H * H).real
        
        m = np.max(conv)
        if m < current_val - 1e-12:
            current_val = m
            if m < best_val:
                best_val = m
                best_h = current_h.copy()
        
        # Find where the peak is
        peak_idx = np.argmax(conv)
        
        # Create a weight that emphasizes the peak region
        beta = 50.0
        exp_conv = np.exp((conv - m) * beta)
        weights = exp_conv / np.sum(exp_conv)
        
        # Gradient of smooth max w.r.t. h
        w_full = np.zeros(size)
        w_full[:len(conv)] = weights
        W = np.fft.fft(w_full, n=size)
        grad_h = 2.0 * np.fft.ifft(H * W).real[:N]
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad_h)
        if grad_norm < 1e-10:
            break
        grad_h = grad_h / grad_norm
        
        # Adaptive step size
        eta = 0.05
        for _ in range(10):
            new_h = current_h - eta * grad_h
            new_h = np.maximum(new_h, 0)
            s = np.sum(new_h)
            if s < 1e-10:
                eta *= 0.5
                continue
            new_h = new_h / s
            
            new_val = compute_autoconv_max(new_h)
            if new_val < current_val:
                current_h = new_h
                current_val = new_val
                if current_val < best_val:
                    best_val = current_val
                    best_h = current_h.copy()
                break
            else:
                eta *= 0.5
                if eta < 1e-8:
                    break
        
        if eta < 1e-8:
            break

    # Final polish with smaller steps
    if time_left() > 20:
        current_h = best_h.copy()
        current_val = best_val
        eta = 0.01
        for k in range(500):
            if time_left() < 5:
                break
            
            h_pad = np.zeros(size)
            h_pad[:N] = current_h
            H = np.fft.fft(h_pad, n=size)
            conv = np.fft.ifft(H * H).real
            
            m = np.max(conv)
            exp_conv = np.exp((conv - m) * 100.0)
            weights = exp_conv / np.sum(exp_conv)
            
            w_full = np.zeros(size)
            w_full[:len(conv)] = weights
            W = np.fft.fft(w_full, n=size)
            grad_h = 2.0 * np.fft.ifft(H * W).real[:N]
            
            grad_norm = np.linalg.norm(grad_h)
            if grad_norm < 1e-10:
                break
            grad_h = grad_h / grad_norm
            
            new_h = current_h - eta * grad_h
            new_h = np.maximum(new_h, 0)
            s = np.sum(new_h)
            if s < 1e-10:
                eta *= 0.5
                continue
            new_h = new_h / s
            
            new_val = compute_autoconv_max(new_h)
            if new_val < current_val:
                current_h = new_h
                current_val = new_val
                if current_val < best_val:
                    best_val = current_val
                    best_h = current_h.copy()
                eta = min(eta * 1.05, 0.05)
            else:
                eta *= 0.8
                if eta < 1e-8:
                    break

    return best_h.tolist()
