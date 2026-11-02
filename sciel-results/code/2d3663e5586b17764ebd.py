import numpy as np
import time

N = 100

def _overlap_max(h):
    """Compute max cyclic overlap of h with 1-h, scaled by step width."""
    w = 2.0 / N
    H = np.fft.fft(h)
    autocorr = np.fft.ifft(H * np.conj(H)).real
    overlaps = N / 2.0 - autocorr
    return w * np.max(overlaps)

def construct():
    deadline = time.time() + 110
    
    # Start with a binary pattern: 50 ones, 50 zeros
    h = np.zeros(N)
    h[0::2] = 1.0
    
    best_h = h.copy()
    best_val = _overlap_max(h)
    curr_val = best_val
    
    # Simulated annealing
    T = 1.0
    cooling = 0.9995
    min_T = 1e-6
    
    while time.time() < deadline - 5 and T > min_T:
        # Pick a random 1 and a random 0 to swap
        ones = np.where(h == 1)[0]
        zeros = np.where(h == 0)[0]
        
        if len(ones) == 0 or len(zeros) == 0:
            break
        
        i = ones[np.random.randint(len(ones))]
        j = zeros[np.random.randint(len(zeros))]
        
        if i == j:
            continue
        
        # Perform swap
        h[i] = 0.0
        h[j] = 1.0
        new_val = _overlap_max(h)
        
        # Accept or reject
        if new_val < curr_val:
            curr_val = new_val
            if new_val < best_val:
                best_val = new_val
                best_h = h.copy()
        else:
            # Reject with probability exp(-delta/T)
            delta = new_val - curr_val
            if np.random.random() < np.exp(-delta / T):
                curr_val = new_val
            else:
                # Undo swap
                h[i] = 1.0
                h[j] = 0.0
        
        T *= cooling
    
    # Final local search
    for _ in range(5):
        improved = False
        ones = np.where(best_h == 1)[0]
        zeros = np.where(best_h == 0)[0]
        
        for i in ones:
            for j in zeros:
                best_h[i] = 0.0
                best_h[j] = 1.0
                val = _overlap_max(best_h)
                if val < best_val:
                    best_val = val
                    improved = True
                else:
                    best_h[i] = 1.0
                    best_h[j] = 0.0
        
        if not improved:
            break
    
    return best_h
