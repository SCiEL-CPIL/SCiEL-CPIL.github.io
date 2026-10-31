import numpy as np
import time

N = 400

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 100.0  # 20s margin
    
    best_val = np.inf
    best_seq = np.ones(N)
    
    # Strategy: Search for binary quadratic phase codes (chirps)
    # x_i = sign(cos(pi * alpha * i^2 + phase))
    # We want to minimize max |autoconv(x)|
    
    # Since sum(x^2) = N is constant for +/-1, we just minimize max |c_k|
    
    # Try a range of alphas
    # Common good alphas are related to 1/N or 2/N
    alphas = [1.0/N, 2.0/N, 1.0/(N-1), 2.0/(N-1), 1.0/np.sqrt(N), 0.5/N]
    
    # Also try random phases
    rng = np.random.default_rng(42)
    
    for alpha in alphas:
        if time.time() > deadline:
            break
        for phase in [0.0, np.pi/4, np.pi/2, np.pi]:
            if time.time() > deadline:
                break
            # Generate sequence
            i = np.arange(N)
            # Quadratic phase
            phase_vals = np.pi * alpha * i**2 + phase
            seq = np.sign(np.cos(phase_vals))
            # Ensure non-zero (sign(0)=0, replace with 1)
            seq[seq == 0] = 1
            
            # Calculate Autoconvolution
            # Use FFT for speed
            # Zero pad to 2N to avoid circular convolution issues
            f = np.fft.fft(seq, n=2*N)
            ac = np.fft.ifft(f * np.conj(f)).real
            ac = ac[:N] # Only need lags 0 to N-1
            
            max_abs = np.max(np.abs(ac))
            
            if max_abs < best_val:
                best_val = max_abs
                best_seq = seq.copy()
                
    # If no improvement found, return seed
    # Note: If the metric is actually (sum)^2, this will be bad.
    # But based on the directive, this is the intended path.
    
    return best_seq

# EVOLVE-BLOCK-START
# The code above is the new design.
# EVOLVE-BLOCK-END
