import numpy as np
import time

N = 100

def compute_score(h):
    """Compute the maximum overlap of h with shifted 1-h."""
    max_ov = 0.0
    for s in range(1, N):  # s=0 gives overlap 0, skip it
        shifted = np.roll(h, s)
        ov = np.sum(h * (1.0 - shifted))
        if ov > max_ov:
            max_ov = ov
    return max_ov

def construct():
    # Start with uniform
    h = np.full(N, 0.5)
    best_score = compute_score(h)
    
    deadline = time.time() + 140  # Leave 10s margin
    
    # Try alternating pattern
    h_alt = np.tile([1.0, 0.0], N//2)
    score_alt = compute_score(h_alt)
    if score_alt < best_score:
        best_score = score_alt
        h = h_alt.copy()
    
    # Local search on binary sequences with exactly N/2 ones
    np.random.seed(42)
    
    for restart in range(100):
        if time.time() > deadline:
            break
        
        # Generate random binary sequence with N/2 ones
        h_bin = np.zeros(N)
        ones_positions = np.random.choice(N, N//2, replace=False)
        h_bin[ones_positions] = 1.0
        
        current_score = compute_score(h_bin)
        if current_score < best_score:
            best_score = current_score
            h = h_bin.copy()
        
        # Local search: try swapping pairs of bits (one 1 with one 0)
        improved = True
        stall_count = 0
        while improved and time.time() < deadline:
            improved = False
            stall_count = 0
            for _ in range(500):
                if time.time() > deadline:
                    break
                
                # Pick two random positions, one with 1 and one with 0
                ones = np.where(h_bin == 1)[0]
                zeros = np.where(h_bin == 0)[0]
                if len(ones) == 0 or len(zeros) == 0:
                    break
                
                i = np.random.choice(ones)
                j = np.random.choice(zeros)
                
                # Swap: h[i] becomes 0, h[j] becomes 1
                h_bin[i] = 0
                h_bin[j] = 1
                
                new_score = compute_score(h_bin)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    stall_count = 0
                    if current_score < best_score:
                        best_score = current_score
                        h = h_bin.copy()
                else:
                    # Revert
                    h_bin[i] = 1
                    h_bin[j] = 0
                    stall_count += 1
                
                if stall_count > 200:
                    break
    
    return h
