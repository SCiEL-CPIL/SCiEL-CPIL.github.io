import numpy as np
import random

N = 50

def construct():
    # Find a large Sidon set in {0, ..., N-1}
    # A Sidon set has distinct pairwise sums i+j (i <= j)
    # This minimizes the peak of the autoconvolution for uniform weights,
    # maximizing the ratio R = ||h||_2^2 / (||h||_1 * max(h*h))
    
    best_S = []
    best_size = 0
    
    # Use a greedy heuristic with random restarts to find a large Sidon set.
    # The size of the Sidon set directly determines the ratio for uniform weights on that set.
    
    num_restarts = 2000
    target_size = 10  # Heuristic upper bound for N=50
    
    for _ in range(num_restarts):
        S = []
        used_sums = set()
        
        # Randomize the order of candidates to explore different local optima
        candidates = list(range(N))
        random.shuffle(candidates)
        
        for i in candidates:
            # Check if adding i creates a collision with existing sums
            # New sums will be i+i and i+j for j in S
            conflict = False
            if (i + i) in used_sums:
                conflict = True
            else:
                for j in S:
                    if (i + j) in used_sums:
                        conflict = True
                        break
            
            if not conflict:
                S.append(i)
                # Add new sums to used_sums
                used_sums.add(i + i)
                for j in S[:-1]: # S[:-1] is S before adding i
                    used_sums.add(i + j)
                
                if len(S) >= target_size:
                    break
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = S
            
            if best_size >= target_size:
                break

    if not best_S:
        best_S = [0]
        
    h = np.zeros(N)
    for i in best_S:
        h[i] = 1.0 / len(best_S)
        
    return list(h)

def compute_ratio(h):
    h = np.asarray(h)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    conv = np.convolve(h, h, mode='full')
    max_peak = np.max(conv)
    if max_peak == 0:
        return 0.0
    return l2_sq / (l1 * max_peak)
