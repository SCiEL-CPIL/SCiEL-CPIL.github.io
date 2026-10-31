import time
import numpy as np

N = 600

def compute_metric(h):
    """Compute normalized maximum autoconvolution."""
    h = np.asarray(h, dtype=float)
    h = np.maximum(h, 0.0)
    s = h.sum()
    if s < 1e-15:
        return 1e10
    hn = h / s
    fft_size = 2 * N
    F = np.fft.rfft(hn, fft_size)
    conv = np.fft.irfft(F * F, fft_size)[:2 * N - 1]
    return conv.max()

def project_simplex(v):
    """Project v onto the probability simplex."""
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1.0)
    w = np.maximum(v - theta, 0)
    return w

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    def elapsed():
        return time.time() - start_time

    # Initialize with Gaussian
    x = np.linspace(-1, 1, N)
    h = np.exp(-2.0 * x**2)
    h = project_simplex(h)
    
    best_val = compute_metric(h)
    h_best = h.copy()
    
    beta = 0.05
    eta = 0.05
    min_beta = 0.001
    fft_size = 2 * N
    
    while elapsed() < deadline:
        # Compute convolution
        F = np.fft.rfft(h, fft_size)
        conv = np.fft.irfft(F * F, fft_size)[:2 * N - 1]
        
        val = np.max(conv)
        if val < best_val:
            best_val = val
            h_best = h.copy()
            
        # Compute soft-max weights
        max_conv = val
        exp_conv = np.exp((conv - max_conv) / beta)
        sum_exp = np.sum(exp_conv)
        p = exp_conv / sum_exp
        
        # Compute gradient
        h_rev = h[::-1]
        grad_full = np.convolve(p, h_rev, mode='full')
        grad = grad_full[N-1 : N-1+N]
        
        # Update h
        h_new = h - eta * grad
        h_new = project_simplex(h_new)
        
        # Simple adaptive learning rate
        # If we are making progress, keep eta. If not, reduce it?
        # For now, decay eta slowly to ensure convergence
        eta *= 0.999
        beta *= 0.995
        
        if beta < min_beta:
            beta = min_beta
            
        h = h_new
        
        if elapsed() > deadline - 5:
            break

    h_best = np.maximum(h_best, 0.0)
    if h_best.sum() < 1e-15:
        h_best = np.ones(N)
    h_best = h_best / h_best.sum()
    
    return list(h_best)
