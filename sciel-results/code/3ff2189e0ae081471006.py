import numpy as np
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initialize with uniform distribution
    h = np.ones(N)
    h /= np.sum(h)  # normalize L1 to 1
    
    def compute_ratio(h):
        # h is normalized so sum(h) = 1
        l2_sq = np.sum(h**2)
        # Discrete autoconvolution
        h_conv = np.convolve(h, h, mode='full')
        max_conv = np.max(h_conv)
        if max_conv < 1e-15:
            return 0.0
        return l2_sq / max_conv
    
    best_h = h.copy()
    best_ratio = compute_ratio(best_h)
    
    # Try a few simple patterns
    patterns = []
    # Uniform
    patterns.append(np.ones(N) / N)
    # Spike
    spike = np.zeros(N)
    spike[0] = 1.0
    patterns.append(spike)
    # Center spike
    spike_c = np.zeros(N)
    spike_c[N//2] = 1.0
    patterns.append(spike_c)
    # Gaussian-like
    x = np.linspace(-1, 1, N)
    gauss = np.exp(-x**2 * 5)
    gauss /= np.sum(gauss)
    patterns.append(gauss)
    # Triangular
    tri = np.abs(1 - 2*np.arange(N)/N)
    tri /= np.sum(tri)
    patterns.append(tri)
    
    for p in patterns:
        r = compute_ratio(p)
        if r > best_ratio:
            best_ratio = r
            best_h = p.copy()
    
    # Local search with random perturbations
    np.random.seed(42)
    step_size = 0.1
    for i in range(5000):
        if time.time() > deadline:
            break
        # Random perturbation
        delta = np.random.randn(N) * step_size
        h_new = best_h + delta
        h_new = np.maximum(h_new, 0)
        s = np.sum(h_new)
        if s < 1e-15:
            continue
        h_new /= s
        
        r = compute_ratio(h_new)
        if r > best_ratio:
            best_ratio = r
            best_h = h_new.copy()
            step_size = max(step_size * 0.95, 0.001)
        else:
            step_size = min(step_size * 1.05, 0.5)
    
    return list(best_h)
