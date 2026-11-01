# EVOLVE-BLOCK-START
import numpy as np
import time

N = 400

def _generate_de_bruijn(k, n=2):
    """Generate a de Bruijn sequence of order k over alphabet of size n using the FKM algorithm."""
    a = [0] * (n * (n ** k))
    sequence = []
    
    def db(t, p):
        if t > k:
            if k % p == 0:
                sequence.extend(a[1:p+1])
        else:
            a[t] = a[t-p]
            db(t+1, p)
            for j in range(a[t-p]+1, n):
                a[t] = j
                db(t+1, t)
                
    db(1, 1)
    return sequence

def _autocorrelation(x):
    """Compute full autocorrelation of sequence x (length N)."""
    n = len(x)
    size = 1
    while size < 2 * n:
        size *= 2
    X = np.fft.fft(x, size)
    acf = np.fft.ifft(X * np.conj(X)).real
    return acf[:n]

def _compute_c3(x):
    """Compute the C3 upper bound from the autocorrelation."""
    acf = _autocorrelation(x)
    sidelobes = acf[1:]
    return np.max(np.abs(sidelobes))

def construct():
    start_time = time.time()
    
    # Generate de Bruijn sequence of order 9 (length 512)
    # This structure ensures maximal uniformity of subsequences of length 9.
    k = 9
    seq_bits = _generate_de_bruijn(k)
    
    # Map bits to +/- 1
    # 0 -> 1, 1 -> -1
    seq_pm1 = np.array([1.0 if b == 0 else -1.0 for b in seq_bits])
    L = len(seq_pm1) # 512
    
    best_c3 = float('inf')
    best_seq = None
    
    # Try all cyclic shifts to find the best window of length N=400
    # This exploits the "torus" nature of the de Bruijn sequence
    for shift in range(L):
        # Extract window with wrap-around
        if shift + N <= L:
            window = seq_pm1[shift : shift + N]
        else:
            window = np.concatenate([seq_pm1[shift:], seq_pm1[:shift + N - L]])
        
        # Compute C3
        c3 = _compute_c3(window)
        
        if c3 < best_c3:
            best_c3 = c3
            best_seq = window.copy()
            
        # Time check
        if time.time() - start_time > 100:
            break
            
    if best_seq is None:
        # Fallback: random or all ones
        best_seq = np.ones(N)
        
    return best_seq
# EVOLVE-BLOCK-END
