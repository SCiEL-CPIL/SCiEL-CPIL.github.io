import numpy as np
import time

N = 100

def compute_max_overlap(h):
    H = np.fft.rfft(h)
    acf = np.fft.irfft(np.abs(H)**2, n=N)
    min_acf = np.min(acf[1:])
    return N / 2.0 - min_acf

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_h = None
    best_score = float('inf')
    
    np.random.seed(42)
    
    n_restarts = 5
    
    for r in range(n_restarts):
        if time.time() > deadline - 10:
            break
            
        h = np.zeros(N)
        h[:N//2] = 1.0
        np.random.shuffle(h)
        
        current_score = compute_max_overlap(h)
        best_local_h = h.copy()
        best_local_score = current_score
        
        idx_ones = np.where(h == 1.0)[0]
        idx_zeros = np.where(h == 0.0)[0]
        
        max_iters = 50000
        for i in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Pick random index from ones and zeros
            i1 = idx_ones[np.random.randint(len(idx_ones))]
            i2 = idx_zeros[np.random.randint(len(idx_zeros))]
            
            # Swap
            h[i1], h[i2] = h[i2], h[i1]
            
            # Update indices
            # i1 was 1, now 0. i2 was 0, now 1.
            # We can just update the lists?
            # Or simply recompute? No, that's expensive.
            # Let's just keep the lists and swap the elements in the lists.
            # idx_ones contains i1. idx_zeros contains i2.
            # We need to remove i1 from idx_ones and add to idx_zeros.
            # This is O(N) if we use lists.
            # For N=100, it's fine.
            
            # Actually, simpler: just use random i, j and check.
            # The 50% waste is acceptable.
            
            new_score = compute_max_overlap(h)
            
            if new_score < best_local_score:
                best_local_score = new_score
                best_local_h = h.copy()
                # Update indices if we were maintaining them, but we aren't strictly
            else:
                h[i1], h[i2] = h[i2], h[i1]
                
        if best_local_score < best_score:
            best_score = best_local_score
            best_h = best_local_h.copy()
            
    if best_h is None:
        best_h = np.full(N, 0.5)
        
    return best_h
