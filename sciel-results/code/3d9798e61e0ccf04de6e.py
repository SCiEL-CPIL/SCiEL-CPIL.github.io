import numpy as np
import time

N = 400
_start_time = time.time()
_TIME_LIMIT = 145.0  # Safety margin before 150s kill

def construct():
    """
    Minimizes max(|autoconv(f)|) / (sum(f))^2 using local search on binary sequences.
    """
    # Initialize with Thue-Morse sequence
    # Thue-Morse has low autocorrelation
    # t(n) = (-1)^(popcount(n))
    # Using numpy bit_count is not available in older versions, use bin(x).count('1')
    # Or use the recursive property: t(2n) = t(n), t(2n+1) = -t(n)
    
    f = np.zeros(N, dtype=np.int8)
    f[0] = 1
    for i in range(1, N):
        # Find largest power of 2 <= i
        # Or simply use the recurrence
        # t(i) = -t(i // 2) if i is odd? No.
        # t(2n) = t(n)
        # t(2n+1) = -t(n)
        
        # Efficient way:
        # i = 2n + r
        # f[i] = f[n] if r==0 else -f[n]
        n = i // 2
        r = i % 2
        f[i] = f[n] if r == 0 else -f[n]
        
    S = np.sum(f)
    
    # Compute autoconvolution
    C = np.convolve(f, f)
    
    max_abs_C = np.max(np.abs(C))
    current_score = max_abs_C / (S * S) if S != 0 else float('inf')
    
    best_f = f.copy()
    best_score = current_score
    
    # Local search
    improved = True
    while improved:
        improved = False
        # Check time
        if time.time() - _start_time > _TIME_LIMIT:
            break
            
        # Try flipping each bit
        for i in range(N):
            # Calculate new sum
            new_S = S - 2 * f[i]
            
            # Calculate change in C
            # delta_C[k] = -4 * f[i] * f[k-i] (for k != 2i)
            # delta_C[2i] = 0
            
            k_start = max(0, i)
            k_end = min(2*N - 1, i + N)
            
            idx = np.arange(k_start, k_end)
            f_idx = idx - i
            
            delta = -4 * f[i] * f[f_idx]
            
            # Mask out the center
            center_mask = (idx == 2 * i)
            delta[center_mask] = 0
            
            # New C values
            C_new_vals = C[idx] + delta
            
            # Update C
            C_temp = C.copy()
            C_temp[idx] = C_new_vals
            
            new_max_abs_C = np.max(np.abs(C_temp))
            
            if new_S == 0:
                new_score = float('inf')
            else:
                new_score = new_max_abs_C / (new_S * new_S)
            
            if new_score < current_score:
                # Accept
                f[i] = -f[i]
                S = new_S
                C = C_temp
                max_abs_C = new_max_abs_C
                current_score = new_score
                improved = True
                best_f = f.copy()
                best_score = current_score

    return best_f

# EVOLVE-BLOCK-END
