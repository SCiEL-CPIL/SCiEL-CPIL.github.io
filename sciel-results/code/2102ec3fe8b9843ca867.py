import numpy as np
import time

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    def max_overlap(h):
        # Vectorized computation of all overlaps
        # overlap[k] = sum_i h[i] * (1 - h[(i+k) mod N])
        # = sum_i h[i] - sum_i h[i]*h[(i+k) mod N]
        # = N/2 - sum_i h[i]*h[(i+k) mod N]
        # So we want to minimize max_k (N/2 - corr[k]) = N/2 - min_k corr[k]
        # where corr[k] = sum_i h[i]*h[(i+k) mod N]
        
        # Compute all circular cross-correlations
        # Use FFT for speed
        h_pad = np.zeros(2*N)
        h_pad[:N] = h
        h_flip = np.zeros(2*N)
        h_flip[:N] = h[::-1]  # reverse for correlation
        
        # Circular correlation
        fft_h = np.fft.fft(h, n=2*N)
        fft_h_flip = np.fft.fft(h_flip, n=2*N)
        corr = np.fft.ifft(fft_h * np.conj(fft_h_flip)).real[:N]
        
        # corr[k] = sum_i h[i] * h[(i-k) mod N] for k=0..N-1
        # We need sum_i h[i] * h[(i+k) mod N] = corr[-k mod N]
        # So the set of values is the same, just permuted
        min_corr = np.min(corr)
        return N/2 - min_corr
    
    # Try sinusoidal patterns with different frequencies
    best_h = None
    best_score = float('inf')
    
    # Grid search over frequency and amplitude
    for freq in range(1, N//2 + 1):
        if time.time() > deadline - 20:
            break
        for amp in np.linspace(0.1, 0.49, 10):
            h = 0.5 + amp * np.sin(2 * np.pi * freq * np.arange(N) / N)
            h = np.clip(h, 0, 1)
            h = h / h.sum() * (N / 2)
            h = np.clip(h, 0, 1)
            
            score = max_overlap(h)
            if score < best_score:
                best_score = score
                best_h = h.copy()
    
    # Local refinement
    if best_h is not None and time.time() < deadline - 10:
        h = best_h.copy()
        lr = 0.005
        for step in range(2000):
            if time.time() > deadline - 5:
                break
            # Simple subgradient step
            # Compute which shift gives the max overlap
            h_pad = np.zeros(2*N)
            h_pad[:N] = h
            h_flip = np.zeros(2*N)
            h_flip[:N] = h[::-1]
            fft_h = np.fft.fft(h, n=2*N)
            fft_h_flip = np.fft.fft(h_flip, n=2*N)
            corr = np.fft.ifft(fft_h * np.conj(fft_h_flip)).real[:N]
            
            # The gradient of the max overlap w.r.t. h
            # max_k (N/2 - corr[k]) where corr[k] = sum_i h[i]*h[(i+k) mod N]
            # d/dh[j] corr[k] = h[(j+k) mod N] + h[(j-k) mod N]
            # So d/dh[j] (N/2 - corr[k]) = -(h[(j+k) mod N] + h[(j-k) mod N])
            
            k_max = np.argmin(corr)  # k that minimizes corr, thus maximizes overlap
            grad = -(h[(np.arange(N) + k_max) % N] + h[(np.arange(N) - k_max) % N])
            
            h = h - lr * grad
            h = np.clip(h, 0, 1)
            h = h / h.sum() * (N / 2)
            h = np.clip(h, 0, 1)
        
        score = max_overlap(h)
        if score < best_score:
            best_score = score
            best_h = h
    
    if best_h is None:
        return np.full(N, 0.5)
    
    # Ensure exact sum
    best_h = best_h / best_h.sum() * (N / 2)
    best_h = np.clip(best_h, 0, 1)
    
    return best_h
