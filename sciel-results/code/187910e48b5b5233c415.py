import numpy as np
import time

N = 100

def _max_overlap(h):
    """Calculate the maximum overlap of h with shifted 1-h."""
    # Overlap O_k = N/2 - C_k
    # Max overlap = N/2 - Min C_k
    H = np.fft.fft(h)
    # Autocorrelation
    C = np.real(np.fft.ifft(np.abs(H)**2))
    min_C = np.min(C)
    return N / 2.0 - min_C

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin for 150s limit
    
    # Initial guess: uniform
    h = np.full(N, 0.5)
    
    # Parameters
    eta = 0.1 # Learning rate
    lambda_soft = 5.0 # Softness parameter for min
    T = 2000 # Number of iterations
    
    # Precompute FFT of h? No, h changes.
    
    for t in range(T):
        if time.time() > deadline:
            break
            
        # 1. Compute Autocorrelation C_k
        H = np.fft.fft(h)
        C = np.real(np.fft.ifft(np.abs(H)**2))
        
        # 2. Compute weights for soft-min
        # We want to maximize min(C_k), so we minimize sum(exp(-lambda * C_k))
        # Gradient of objective J = sum(exp(-lambda * C_k)) w.r.t h
        # dJ/dh = -lambda * sum_k exp(-lambda * C_k) * dC_k/dh
        # We want to ascend J, so we descend -J? 
        # Wait, we want to MAXIMIZE min C_k.
        # Let's define Loss L = sum(exp(-lambda * C_k)). Minimizing L pushes up the smallest C_k.
        # Gradient of L w.r.t h:
        # dL/dh_i = -lambda * sum_k w_k * (h_{i-k} + h_{i+k})
        # where w_k = exp(-lambda * C_k)
        # We want to update h to DECREASE L.
        # So h <- h - eta * dL/dh = h + eta * lambda * sum_k w_k * (h_{i-k} + h_{i+k})
        
        w = np.exp(-lambda_soft * C)
        
        # Compute Gradient G = sum_k w_k * (h_{i-k} + h_{i+k})
        # This is a convolution of h with kernel K where K_k = w_k + w_{-k}
        # Since C_k is real and symmetric (autocorrelation), w_k is symmetric.
        # So K_k = 2 * w_k.
        K = 2.0 * w
        # Circular Convolution
        # G = ifft(fft(h) * fft(K))
        G = np.real(np.fft.ifft(H * np.fft.fft(K)))
        
        # 3. Update h
        # Preserve sum: subtract mean of G
        G_mean = np.mean(G)
        h_new = h + eta * lambda_soft * (G - G_mean)
        
        # 4. Project to feasible set
        # Iterate projection to satisfy bounds [0,1] and sum N/2
        for _ in range(5):
            # Clip to bounds
            h_new = np.clip(h_new, 0.0, 1.0)
            # Adjust sum
            diff = N / 2.0 - np.sum(h_new)
            h_new += diff / N
            # Clip again
            h_new = np.clip(h_new, 0.0, 1.0)
            
        h = h_new
        
        # Optional: Add small noise to escape local minima
        if t % 100 == 0 and t > 0:
            h += np.random.normal(0, 0.01, N)
            h = np.clip(h, 0, 1)
            diff = N / 2.0 - np.sum(h)
            h += diff / N
            h = np.clip(h, 0, 1)
            
    # Final projection to ensure constraints are met
    for _ in range(10):
        h = np.clip(h, 0.0, 1.0)
        diff = N / 2.0 - np.sum(h)
        h += diff / N
        h = np.clip(h, 0.0, 1.0)
        
    # Verify sum is close
    if abs(np.sum(h) - N/2.0) > 1e-6:
        # Fallback to uniform if projection failed badly
        h = np.full(N, 0.5)
        
    return h
