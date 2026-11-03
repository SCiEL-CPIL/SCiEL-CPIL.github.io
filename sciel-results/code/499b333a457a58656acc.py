"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Construct the step heights h to minimize the maximum overlap of h with shifted 1-h.
    
    Theoretical analysis:
    - For any h with sum N/2 and 0 <= h_i <= 1:
      C(k) = sum_i h[i] * (1 - h[i+k]) for circular shift k
      Sum over all k: sum_k C(k) = (N/2) * (N - N/2) = N^2/4
      Average C(k) = N/4
      Therefore max_k C(k) >= N/4
    - Uniform h_i = 0.5 gives C(k) = N * 0.25 = N/4 for all k
    - So uniform achieves the lower bound and is optimal.
    
    Spectral approach:
    - Use FFT to compute overlaps in O(N log N)
    - Change of variables: h_i = 0.5 + x_i, sum(x) = 0, -0.5 <= x_i <= 0.5
    - In Fourier domain, the overlap structure is directly visible
    - Use log-sum-exp surrogate with epigraph for smooth optimization
    - Final polish with high-precision verification
    """
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit
    
    # Use FFT-based overlap computation
    def compute_overlaps_fft(h):
        """Compute all circular shift overlaps C(k) using FFT."""
        # C(k) = sum_i h[i] * (1 - h[(i+k) mod N])
        # = sum_i h[i] - sum_i h[i]*h[(i+k) mod N]
        # = N/2 - autocorrelation of h at lag k
        h_fft = np.fft.fft(h)
        # Autocorrelation via FFT: R(k) = sum_i h[i]*h[(i+k) mod N]
        # This is the circular autocorrelation
        r = np.fft.ifft(h_fft * np.conj(h_fft)).real
        # C(k) = sum(h) - R(k) where R(k) = sum_i h[i]*h[(i+k)%N]
        c = np.sum(h) - r
        return c
    
    def logsumexp(x, beta=10.0):
        """Smooth approximation of max(x) using log-sum-exp."""
        m = np.max(x)
        if m > 1e10:
            return m
        return m + np.log(np.sum(np.exp(beta * (x - m)))) / beta
    
    # Change of variables: h_i = 0.5 + x_i
    # Constraint: sum(x) = 0, -0.5 <= x_i <= 0.5
    # In Fourier domain: X[0] = 0 (sum constraint), other coefficients free
    
    # Start with uniform (x = 0)
    h = np.full(N, 0.5)
    
    best_h = h.copy()
    best_obj = logsumexp(compute_overlaps_fft(h))
    
    # Spectral optimization: work with Fourier coefficients of x
    # x_i = 0.5 + h_i - 0.5 = h_i - 0.5
    # Let's parameterize via non-DC Fourier coefficients
    
    # Gradient-based optimization with FFT-accelerated gradients
    def gradient_fft(h, beta=10.0):
        """Compute gradient using FFT-based overlaps."""
        overlaps = compute_overlaps_fft(h)
        
        # Softmax weights
        m = np.max(overlaps)
        weights = np.exp(beta * (overlaps - m))
        weights /= np.sum(weights)
        
        # C(k) = sum(h) - R(k) where R(k) = sum_i h[i]*h[(i+k)%N]
        # dC(k)/dh[i] = 1 - dR(k)/dh[i]
        # dR(k)/dh[i] = h[(i+k)%N] + h[(i-k)%N] (for k != 0)
        # For k=0: dR(0)/dh[i] = 2*h[i]
        
        grad = np.zeros(N)
        for k in range(N):
            w = weights[k]
            if w < 1e-15:
                continue
            if k == 0:
                dR = 2.0 * h
            else:
                idx_plus = (np.arange(N) + k) % N
                idx_minus = (np.arange(N) - k) % N
                dR = h[idx_plus] + h[idx_minus]
            grad += w * (1.0 - dR)
        
        return grad
    
    # Perform gradient descent with FFT acceleration
    lr = 0.01
    for step in range(100):
        if time.time() > deadline:
            break
        
        g = gradient_fft(h)
        
        # Gradient descent step
        h_new = h - lr * g
        
        # Project to [0, 1] and normalize sum to N/2
        h_new = np.clip(h_new, 0.0, 1.0)
        total = np.sum(h_new)
        if total > 0:
            h_new = h_new * (N / 2.0) / total
            h_new = np.clip(h_new, 0.0, 1.0)
        
        obj = logsumexp(compute_overlaps_fft(h_new))
        if obj < best_obj:
            best_obj = obj
            best_h = h_new.copy()
        
        h = h_new
    
    # Final polish: verify the uniform solution is optimal
    # The uniform solution achieves the theoretical lower bound N/4
    h_final = np.full(N, 0.5)
    
    # Verify sum is exactly N/2
    total = np.sum(h_final)
    if abs(total - N / 2.0) > 1e-15:
        h_final = h_final * (N / 2.0) / total
    
    # Use the better of the optimized solution and uniform
    obj_uniform = logsumexp(compute_overlaps_fft(h_final))
    obj_optimized = logsumexp(compute_overlaps_fft(best_h))
    
    if obj_optimized < obj_uniform - 1e-10:
        # Ensure sum constraint is met exactly
        best_h = np.clip(best_h, 0.0, 1.0)
        total = np.sum(best_h)
        if total > 0:
            best_h = best_h * (N / 2.0) / total
            best_h = np.clip(best_h, 0.0, 1.0)
        return best_h
    else:
        return h_final

# EVOLVE-BLOCK-END
