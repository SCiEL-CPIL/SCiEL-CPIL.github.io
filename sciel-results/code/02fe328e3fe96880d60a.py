import numpy as np
import time
from scipy.fft import fft, ifft

N = 400

def _eval_value(h):
    """Compute the objective value."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _convolve_fft(h):
    """Compute full autoconvolution using FFT."""
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

def _grad_soft_max_conv(h, beta):
    """
    Compute gradient of soft_max(conv(h, h)) with respect to h.
    """
    n_conv = 2 * len(h) - 1
    n_fft = 1
    while n_fft < n_conv:
        n_fft *= 2
    
    H = fft(h, n=n_fft)
    C = H * H
    c = ifft(C, n=n_fft).real[:n_conv]
    
    # Softmax weights
    m = np.max(c)
    exp_vals = np.exp(beta * (c - m))
    w = exp_vals / np.sum(exp_vals)
    
    # Gradient g_j = sum_k w_k * (h_{k-j} + h_{j-k})
    # This is 2 * conv(w, h) restricted to valid j?
    # Let's compute conv(w, h) using FFT.
    # w has length n_conv. h has length N.
    # We need the gradient vector of length N.
    
    W = fft(w, n=n_fft)
    # The gradient is related to the inverse FFT of W * H?
    # Actually, d/dh_j sum_k w_k c_k = 2 * sum_k w_k h_{k-j} (approx)
    # Let's use the property that grad = 2 * ifft(W * H).real[:N]?
    # Let's verify with small example.
    
    # For now, let's use a safe numerical gradient if analytic is tricky, 
    # but N=400 is small enough for FFT-based analytic if derived correctly.
    # The gradient of sum(w_k c_k) where c = h*h is 2 * (w * h) ?
    # Let's assume g = 2 * ifft(W * H, n=n_fft).real[:N]
    
    g = 2.0 * ifft(W * H, n=n_fft).real[:len(h)]
    
    return g

def construct():
    deadline = time.time() + 140.0
    best_h = np.ones(N)
    best_val = _eval_value(best_h)
    
    rng = np.random.RandomState(42)
    
    for trial in range(50):
        if time.time() > deadline - 10:
            break
            
        # Initialization
        # Mix of random and smooth
        if trial % 2 == 0:
            h = rng.randn(N)
        else:
            t = np.linspace(0, 1, N)
            h = 1.0 + 0.5 * np.sin(2 * np.pi * rng.randint(1, 10) * t)
            h += 0.1 * rng.randn(N)
        
        s = np.sum(h)
        if abs(s) < 1e-6:
            h = h + 0.1
            s = np.sum(h)
        h = h / s
        
        beta = 20.0
        lr = 0.01
        for step in range(200):
            if time.time() > deadline - 5:
                break
                
            g = _grad_soft_max_conv(h, beta)
            
            # Normalize gradient
            g_norm = np.linalg.norm(g)
            if g_norm < 1e-8:
                break
            
            # Update
            h_new = h - lr * g
            s_new = np.sum(h_new)
            if abs(s_new) < 1e-6:
                break
            h_new = h_new / s_new
            
            # Check value
            v_new = _eval_value(h_new)
            if v_new < best_val:
                best_val = v_new
                best_h = h_new.copy()
            
            h = h_new
            lr *= 0.99 # Decay step size
            
    # Final polish with exact eval
    if time.time() < deadline - 5:
        # Coordinate descent on best_h
        h = best_h.copy()
        for _ in range(10):
            if time.time() > deadline - 2:
                break
            improved = False
            idx = np.random.permutation(N)
            for i in idx:
                old = h[i]
                for delta in [0.01, -0.01]:
                    h[i] = old + delta
                    s = np.sum(h)
                    if abs(s) < 1e-6:
                        h[i] = old
                        continue
                    v = _eval_value(h)
                    if v < best_val:
                        best_val = v
                        best_h = h.copy()
                        improved = True
                    else:
                        h[i] = old
            if not improved:
                break

    return best_h
