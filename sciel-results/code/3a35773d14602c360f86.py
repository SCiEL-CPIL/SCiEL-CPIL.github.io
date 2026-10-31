import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Initialize with all ones
    x = np.ones(N)
    
    # Precompute FFT size
    fft_size = 1 << (2 * N - 1).bit_length()
    
    def calc_score(x):
        # Calculate autocorrelation
        f = np.fft.fft(x, n=fft_size)
        a = np.fft.ifft(f * np.conj(f))
        a = np.real(a[:N])
        
        # Sum
        s = np.sum(x)
        
        # Peak sidelobe (exclude lag 0)
        if N > 1:
            peak = np.max(np.abs(a[1:]))
        else:
            peak = 0.0
            
        if s == 0:
            return float('inf')
            
        return peak / (s * s)

    current_score = calc_score(x)
    
    # Local Search
    # We will try flipping signs
    # To speed up, we can update the autocorrelation incrementally?
    # Flipping x[j] from 1 to -1 changes x by 2 * (-e_j)
    # A_new = A_old + 2 * ( -e_j * x + x * -e_j )?
    # Actually, A[x+dx] = A[x] + A[dx] + A[x,dx] + A[dx,x]
    # dx = -2 * e_j
    # A[dx] = 4 * delta(k)
    # A[x, dx] = -2 * shift(x, j)
    # So A_new[k] = A_old[k] + 4*delta(k) - 2*x[k-j] - 2*x[k+j]
    
    # Let's use FFT for simplicity first, then optimize if needed.
    # N=400, FFT is fast.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_j = -1
        best_score = current_score
        
        # Compute current autocorrelation
        f = np.fft.fft(x, n=fft_size)
        a = np.fft.ifft(f * np.conj(f))
        a = np.real(a[:N])
        
        s = np.sum(x)
        
        # Try flipping each index
        # We can vectorize the score calculation for all flips?
        # s_new = s - 2*x[j]
        # peak_new depends on a_new
        
        # Iterate
        for j in range(N):
            if time.time() > deadline:
                break
                
            # Calculate new sum
            s_new = s - 2 * x[j]
            if s_new == 0:
                continue
                
            # Calculate new autocorrelation approx
            # a_new[k] = a[k] - 2*x[j]*(x[k-j] + x[k+j]) + 4*delta(k)
            # Note: indices wrap? No, linear autocorrelation.
            # x is length N.
            # a[k] = sum_{i=0}^{N-1-k} x[i] x[i+k]
            
            # This update is O(N) per j. Total O(N^2) per iteration.
            # 400 * 400 = 160,000 ops. Very fast.
            
            a_new = a.copy()
            
            # Term 1: -2 * x[j] * x[k-j]
            # k-j is valid if 0 <= k-j < N
            # Term 2: -2 * x[j] * x[k+j]
            # k+j is valid if 0 <= k+j < N
            
            # Vectorized update
            # a_new[k] -= 2 * x[j] * x[k-j] (where valid)
            # a_new[k] -= 2 * x[j] * x[k+j] (where valid)
            
            # Mask for k-j
            idx1 = np.arange(N) - j
            mask1 = (idx1 >= 0) & (idx1 < N)
            a_new[mask1] -= 2 * x[j] * x[idx1[mask1]]
            
            # Mask for k+j
            idx2 = np.arange(N) + j
            mask2 = (idx2 >= 0) & (idx2 < N)
            a_new[mask2] -= 2 * x[j] * x[idx2[mask2]]
            
            # Delta term
            a_new[0] += 4 * x[j] * x[j] # x[j] is +/- 1, so 4
            
            # Calculate peak
            peak_new = np.max(np.abs(a_new[1:]))
            
            score_new = peak_new / (s_new * s_new)
            
            if score_new < best_score:
                best_score = score_new
                best_j = j
        
        if best_j != -1:
            x[best_j] *= -1
            current_score = best_score
            improved = True

    return x

# EVOLVE-BLOCK-END
