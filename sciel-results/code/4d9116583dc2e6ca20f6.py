import numpy as np
import random

N = 100

def construct():
    # Simulated Annealing approach
    # Initial state: uniform
    h = np.full(N, 0.5)
    
    # Temperature schedule
    T_start = 1.0
    T_end = 1e-5
    iterations = 50000
    
    def calc_max_overlap(h):
        """Calculate the maximum overlap C_k for all k"""
        # C_k = (2/N) * (N/2 - sum(h[k:]*h[:-k])) for k>0
        # C_0 = (2/N) * (N/2 - sum(h*h))
        
        # Efficiently compute all autocorrelations
        # Using FFT is overkill for N=100, direct loops or vectorized slicing is fine.
        # Let's compute all C_k.
        
        # r_k = sum_{j=0}^{N-k-1} h[j+k] * h[j]
        # We want max over k of (2/N) * (N/2 - r_k)
        # This is equivalent to minimizing min_k r_k.
        
        min_r = float('inf')
        
        # k=0
        r0 = np.dot(h, h)
        if r0 < min_r:
            min_r = r0
            
        for k in range(1, N):
            # sum h[k:] * h[:-k]
            r_k = np.dot(h[k:], h[:-k])
            if r_k < min_r:
                min_r = r_k
                
        # The score is the upper bound on C5, which is max C_k.
        # C_k = (2/N) * (N/2 - r_k)
        # Max C_k corresponds to Min r_k.
        # So we return the value of max C_k.
        max_c = (2.0 / N) * (N / 2.0 - min_r)
        return max_c

    current_score = calc_max_overlap(h)
    
    # SA parameters
    T = T_start
    cooling_rate = (T_end / T_start) ** (1.0 / iterations)
    
    best_h = h.copy()
    best_score = current_score
    
    for i in range(iterations):
        T *= cooling_rate
        
        # Generate a move
        # Pick two distinct indices
        idx1, idx2 = random.sample(range(N), 2)
        
        # Determine direction: move mass from idx1 to idx2 or vice versa
        # To maintain sum, h[idx1] -= delta, h[idx2] += delta
        # delta must be such that 0 <= h[idx1]-delta <= 1 and 0 <= h[idx2]+delta <= 1
        # Let's pick a random delta within the valid range
        
        # Max possible delta in direction idx1 -> idx2
        # h[idx1] - delta >= 0 => delta <= h[idx1]
        # h[idx2] + delta <= 1 => delta <= 1 - h[idx2]
        max_delta = min(h[idx1], 1 - h[idx2])
        
        if max_delta <= 1e-9:
            continue
            
        # Pick a random delta
        # Sometimes pick small, sometimes large
        # Uniform distribution in [0, max_delta]
        delta = random.uniform(0, max_delta)
        
        if delta < 1e-9:
            continue
            
        # Apply move
        h[idx1] -= delta
        h[idx2] += delta
        
        new_score = calc_max_overlap(h)
        
        # Metropolis acceptance
        delta_score = new_score - current_score
        if delta_score < 0 or random.random() < np.exp(-delta_score / T):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_h = h.copy()
        else:
            # Revert
            h[idx1] += delta
            h[idx2] -= delta
            
    # Ensure constraints are met strictly
    # Sum should be N/2
    diff = N / 2.0 - np.sum(best_h)
    if abs(diff) > 1e-9:
        mask = (best_h > 0) & (best_h < 1)
        if np.any(mask):
            best_h[mask] += diff / np.sum(mask)
            best_h = np.clip(best_h, 0, 1)
            # Re-adjust if clipping changed sum
            diff2 = N / 2.0 - np.sum(best_h)
            if abs(diff2) > 1e-9:
                # Fallback to uniform if constraint repair fails
                best_h = np.full(N, 0.5)
                
    return best_h
