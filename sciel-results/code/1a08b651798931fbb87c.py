import numpy as np
import time
import math

N = 400

def _calc_acf(s):
    """Compute full autocorrelation array A[0..N-1] using FFT."""
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

def _update_acf_flip(A, s, i, old_si):
    """
    Update ACF after flipping s[i] in-place.
    old_si is the value before the flip.
    Returns new A array (copy).
    """
    A_new = A.copy()
    for k in range(1, N):
        change = 0.0
        if i + k < N:
            change -= 2.0 * old_si * s[i + k]
        if i - k >= 0:
            change -= 2.0 * s[i - k] * old_si
        A_new[k] += change
    return A_new

def _local_search(s, A, current_val, deadline, max_passes=100):
    """Perform local search with incremental ACF updates."""
    improved = True
    passes = 0
    while improved and passes < max_passes:
        if time.time() > deadline:
            break
        improved = False
        passes += 1
        indices = np.random.permutation(N)
        for i in indices:
            if time.time() > deadline:
                break
            old_si = s[i]
            s[i] = -old_si
            A_new = _update_acf_flip(A, s, i, old_si)
            val_new = _max_abs_acf_from_array(A_new)
            if val_new < current_val:
                current_val = val_new
                A = A_new
                improved = True
            else:
                s[i] = old_si
    return s, A, current_val

def _simulated_annealing(s, A, current_val, deadline, T0=1.0, cooling=0.999):
    """Perform simulated annealing with block flips."""
    T = T0
    best_val = current_val
    best_s = s.copy()
    best_A = A.copy()
    
    while time.time() < deadline and T > 1e-6:
        # Random block flip
        block_size = np.random.randint(1, 8)
        if block_size > N:
            block_size = N
        start = np.random.randint(0, N - block_size + 1)
        indices = np.arange(start, start + block_size)
        
        # Store old values
        old_vals = s[indices].copy()
        
        # Flip
        s[indices] = -old_vals
        
        # Compute new ACF (using full FFT for block flips as incremental update is complex/slow for blocks)
        # For efficiency, we could use incremental, but for blocks of size > 1, 
        # the change in ACF involves cross terms between the flipped block and the rest,
        # and within the block.
        # Let's stick to full FFT for block flips to ensure correctness and simplicity, 
        # as N=400 is small enough for FFT to be very fast.
        A_new = _calc_acf(s)
        val_new = _max_abs_acf_from_array(A_new)
        
        delta = val_new - current_val
        if delta < 0 or np.random.random() < math.exp(-delta / T):
            current_val = val_new
            A = A_new
            if val_new < best_val:
                best_val = val_new
                best_s = s.copy()
                best_A = A.copy()
        else:
            # Revert
            s[indices] = old_vals
            
        T *= cooling
        
    return best_s, best_A, best_val

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin
    best_val = float('inf')
    best_seq = np.ones(N)
    
    # Generate diverse initializations
    initializations = []
    
    # Thue-Morse sequence
    def _thue_morse(n):
        result = np.ones(n)
        for i in range(1, n):
            count = bin(i).count('1')
            result[i] = 1.0 if count % 2 == 0 else -1.0
        return result
    initializations.append(_thue_morse(N))
    
    # Random initializations
    for _ in range(20):
        s = np.random.randint(0, 2, size=N) * 2 - 1
        initializations.append(s)
    
    # Evaluate each initialization
    for s in initializations:
        if time.time() > deadline - 5:
            break
        A = _calc_acf(s)
        current_val = _max_abs_acf_from_array(A)
        
        # Local search first to quickly find a local minimum
        s, A, current_val = _local_search(s, A, current_val, deadline - 5, max_passes=20)
        
        # Then Simulated Annealing to escape local minima
        if time.time() < deadline - 2:
            s, A, current_val = _simulated_annealing(s, A, current_val, deadline - 2, T0=2.0, cooling=0.9995)
        
        if current_val < best_val:
            best_val = current_val
            best_seq = s.copy()
            
        # If we found a very good solution, we might spend less time on others
        # But for now, let's just continue.
        
    # Final polish with simulated annealing on the best sequence
    if time.time() < deadline - 1:
        s = best_seq.copy()
        A = _calc_acf(s)
        current_val = _max_abs_acf_from_array(A)
        # Use a lower temperature for final polish
        s, A, current_val = _simulated_annealing(s, A, current_val, deadline - 1, T0=0.5, cooling=0.999)
        
        if current_val < best_val:
            best_val = current_val
            best_seq = s.copy()
            
    return best_seq
