import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    # Helper: Check primality
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(np.sqrt(n)) + 1):
            if n % i == 0:
                return False
        return True

    # 1. Initialization using Legendre Symbols
    # Find next prime >= N
    p = N
    while not is_prime(p):
        p += 1
        
    # Generate Legendre symbols for 1..p-1
    # (x/p) = 1 if x is quadratic residue, -1 otherwise
    x_full = np.zeros(p, dtype=np.int8)
    x_full[0] = 1
    for i in range(1, p):
        # Euler's criterion
        if np.power(i, (p - 1) // 2, p) == 1:
            x_full[i] = 1
        else:
            x_full[i] = -1
            
    # Truncate to N
    x = x_full[:N].copy()
    
    # 2. Autoconvolution calculation
    def calc_autoconv(x):
        n = len(x)
        # FFT size 2n to avoid circular convolution
        X = np.fft.rfft(x, n=2 * n)
        R = np.fft.irfft(X * X.conj(), n=2 * n)
        # R[0:N] contains lags 0 to N-1
        # Due to symmetry, max abs is in R[0:N]
        return R[:N]

    # 3. Local Search (Hill Climbing)
    # Minimize max(|R(k)|)
    
    best_x = x.copy()
    best_max_val = np.max(np.abs(calc_autoconv(x)))
    
    # Optimization loop
    while time.time() < deadline:
        R = calc_autoconv(x)
        current_max_val = np.max(np.abs(R))
        
        if current_max_val < best_max_val:
            best_max_val = current_max_val
            best_x = x.copy()
            
        # Try to improve by flipping single bits
        # We look for the flip that most reduces the current maximum
        # To be efficient, we can just try all flips and pick the best one
        best_flip_idx = -1
        best_new_max = current_max_val
        
        # Iterate through all bits
        for i in range(N):
            # Flip
            x[i] = -x[i]
            
            # Calculate new max
            # To save time, we could update R incrementally, but for N=400, FFT is fast enough.
            # Let's check time. 400 flips * (FFT cost).
            # If FFT is ~1ms, 400ms per pass. 145s / 0.4s = 360 passes.
            # This is plenty.
            
            R_new = calc_autoconv(x)
            new_max = np.max(np.abs(R_new))
            
            if new_max < best_new_max:
                best_new_max = new_max
                best_flip_idx = i
                # We can store R_new to avoid recalculating if this is the best, 
                # but we need to revert x[i] to try others.
                # So we just keep the index and value.
            else:
                # Revert immediately to try next bit
                x[i] = -x[i]
                
        if best_flip_idx != -1:
            # Apply the best flip found
            x[best_flip_idx] = -x[best_flip_idx]
        else:
            # No single flip improves the objective. 
            # We are in a local minimum.
            # Try a few random double flips or single random flips to escape?
            # Given the strong start, simple hill climbing often suffices.
            # To prevent infinite loop, break.
            break
            
    return best_x
