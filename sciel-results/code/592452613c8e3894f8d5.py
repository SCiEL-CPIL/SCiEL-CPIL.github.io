import numpy as np
import time

N = 400

def _calc_acf(s):
    """Compute full autocorrelation array A[0..N-1]"""
    n_fft = 1
    while n_fft < 2 * N:
        n_fft *= 2
    s_f = s.astype(np.float64)
    F = np.fft.rfft(s_f, n=n_fft)
    P = F * np.conj(F)
    acf = np.fft.irfft(P, n=n_fft)
    return np.real(acf[:N])

def _max_abs_acf_from_array(A):
    return float(np.max(np.abs(A[1:])))

def construct():
    start_time = time.time()
    deadline = 140.0
    
    best_val = float('inf')
    best_seq = np.ones(N)
    
    while time.time() < deadline:
        # Initialize with random signs
        s = np.random.randint(0, 2, size=N) * 2 - 1
        A = _calc_acf(s)
        current_val = _max_abs_acf_from_array(A)
        
        # Greedy coordinate descent with full pass until no improvement
        # Use a time limit per restart to allow multiple restarts
        restart_start = time.time()
        max_restart_time = 2.0
        
        while time.time() - restart_start < max_restart_time and time.time() < deadline:
            improved = False
            # Random permutation of indices
            indices = np.random.permutation(N)
            
            for i in indices:
                if time.time() - restart_start > max_restart_time or time.time() > deadline:
                    break
                    
                old_si = s[i]
                new_si = -old_si
                
                # Compute delta in A
                # A_new[k] = A[k] - 2 * old_si * (s[(i+k)%N] + s[(i-k)%N])
                # But note: the autocorrelation is linear (not circular)
                # A[k] = sum_{j=0}^{N-1-k} s[j] * s[j+k]
                # When we flip s[i], the change is:
                # For each k, A[k] involves s[i] if i is in [0, N-1-k] or i+k is in [0, N-1]
                
                # Let's compute the delta efficiently
                # A[k] = sum_{j} s[j] s[j+k]
                # The terms involving s[i] are:
                # 1. j = i: s[i] s[i+k] (if i+k < N)
                # 2. j+k = i => j = i-k: s[i-k] s[i] (if i-k >= 0)
                
                # So delta_A[k] = -2 * old_si * (s[i+k] if i+k<N else 0 + s[i-k] if i-k>=0 else 0)
                
                # Vectorized computation of delta
                delta = np.zeros(N)
                
                # Part 1: k where i+k < N, i.e., k < N-i
                # delta[k] += -2 * old_si * s[i+k]
                if N - i > 0:
                    delta[:N-i] += -2 * old_si * s[i:]
                    
                # Part 2: k where i-k >= 0, i.e., k <= i
                # delta[k] += -2 * old_si * s[i-k]
                # s[i-k] for k=0..i is s[i], s[i-1], ..., s[0]
                if i + 1 > 0:
                    delta[:i+1] += -2 * old_si * s[i::-1]
                
                A_new = A + delta
                val_new = _max_abs_acf_from_array(A_new)
                
                if val_new < current_val:
                    s[i] = new_si
                    A = A_new
                    current_val = val_new
                    improved = True
            
            if not improved:
                break
        
        if current_val < best_val:
            best_val = current_val
            best_seq = s.copy()

    return best_seq
