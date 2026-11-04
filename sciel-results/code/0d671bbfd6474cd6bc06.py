"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 100

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def calc_score(h):
        # Linear overlap: sum_{i=0}^{N-1-k} h_i (1-h_{i+k})
        # We want to minimize the max of this over k
        max_overlap = 0.0
        # Optimization: O_k is likely max at small k
        # But to be rigorous, check all k
        # N=100, so 100 iterations of vectorized sum is fast
        for k in range(N):
            if k == 0:
                term = np.sum(h * (1.0 - h))
            else:
                term = np.sum(h[:N-k] * (1.0 - h[k:]))
            if term > max_overlap:
                max_overlap = term
        return max_overlap

    best_h = np.full(N, 0.5)
    best_score = calc_score(best_h)

    # Search space
    # Low frequency cosines
    # h_i = 0.5 + A * cos(2*pi*f*i/N)
    
    # Try a grid
    freqs = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20]
    amps = [0.1, 0.2, 0.3, 0.4, 0.45, 0.49, 0.5]
    
    for f in freqs:
        for A in amps:
            if time.time() > deadline:
                break
            
            i = np.arange(N)
            h = 0.5 + A * np.cos(2.0 * np.pi * f * i / N)
            
            # Enforce mean 0.5
            # Theoretically mean is 0.5, but let's be sure
            h = h - np.mean(h) + 0.5
            
            # Enforce bounds
            # With A <= 0.5, bounds should be [0, 1]
            # But floating point might push slightly out
            h = np.clip(h, 0.0, 1.0)
            
            # Re-enforce mean after clip?
            # If clipped, mean might shift.
            # The constraint is "sum is exactly len/2".
            # If we clip, we might violate the sum.
            # However, for A=0.5, the values are exactly 0 and 1 at peaks.
            # Clipping is a no-op if A<=0.5.
            # Let's assume A<=0.5 is safe.
            
            score = calc_score(h)
            if score < best_score:
                best_score = score
                best_h = h.copy()
        
        if time.time() > deadline:
            break

    return best_h
# EVOLVE-BLOCK-END
