"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time


def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s to be safe
    
    # We want to maximize min_t A(t) where A(t) = sum_i h[i]*h[(i+t)%N]
    # h is binary with N/2 ones
    
    # Initial: try a few heuristics
    candidates = []
    
    # Heuristic 1: first half 1s, second half 0s
    h1 = np.zeros(N)
    h1[:N//2] = 1
    candidates.append(h1)
    
    # Heuristic 2: alternating
    h2 = np.zeros(N)
    h2[::2] = 1
    candidates.append(h2)
    
    # Heuristic 3: random
    for _ in range(5):
        h = np.zeros(N)
        idx = np.random.choice(N, N//2, replace=False)
        h[idx] = 1
        candidates.append(h)
    
    def min_autocorr(h):
        # Compute A(t) for t=1..N-1
        h_int = h.astype(int)
        min_val = float('inf')
        for t in range(1, N):
            A = np.sum(h_int * np.roll(h_int, -t))
            if A < min_val:
                min_val = A
        return min_val
    
    def score_from_min_autocorr(min_A):
        # Overlap = N/2 - min_A, then integral = (N/2 - min_A) * 2/N
        return (N/2 - min_A) * 2.0 / N
    
    best_h = candidates[0]
    best_min_A = min_autocorr(best_h)
    
    for h in candidates[1:]:
        mA = min_autocorr(h)
        if mA > best_min_A:
            best_min_A = mA
            best_h = h.copy()
    
    # Local search: swap a 1 and a 0
    h = best_h.copy()
    min_A = best_min_A
    
    def compute_min_autocorr(h):
        h_int = h.astype(int)
        min_val = float('inf')
        for t in range(1, N):
            A = np.sum(h_int * np.roll(h_int, -t))
            if A < min_val:
                min_val = A
        return min_val
    
    improved = True
    while improved and (time.time() < deadline - 10):
        improved = False
        ones = np.where(h == 1)[0]
        zeros = np.where(h == 0)[0]
        # Try random swaps
        for _ in range(200):
            if time.time() > deadline - 10:
                break
            i = ones[np.random.randint(len(ones))]
            j = zeros[np.random.randint(len(zeros))]
            if i == j:
                continue
            h[i], h[j] = h[j], h[i]
            new_min_A = compute_min_autocorr(h)
            if new_min_A > min_A:
                min_A = new_min_A
                improved = True
            else:
                h[i], h[j] = h[j], h[i]
    
    return h
# EVOLVE-BLOCK-END
