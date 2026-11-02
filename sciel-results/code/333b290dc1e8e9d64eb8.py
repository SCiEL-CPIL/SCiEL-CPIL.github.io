"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_max_overlap(h):
    """Compute the maximum overlap of h with shifted (1-h)."""
    n = len(h)
    # Overlap at shift s: sum_i h[i] * (1 - h[(i+s) % n]) / n
    # = (sum h[i] - sum h[i]*h[(i+s)%n]) / n
    # Since sum h[i] = n/2, this = 0.5 - autocorr[s]/n
    # We want to minimize max over s of this, i.e., maximize min autocorr
    max_ov = 0.0
    for s in range(1, n):
        shifted = np.roll(h, -s)
        ov = np.mean(h * (1.0 - shifted))
        if ov > max_ov:
            max_ov = ov
    return max_ov

def _local_search(h, max_iters=50000):
    """Local search: flip pairs to maintain sum constraint."""
    n = len(h)
    best_h = h.copy()
    best_score = _compute_max_overlap(best_h)
    
    # Precompute for speed
    def score(h):
        max_ov = 0.0
        for s in range(1, n):
            shifted = np.roll(h, -s)
            ov = np.mean(h * (1.0 - shifted))
            if ov > max_ov:
                max_ov = ov
        return max_ov
    
    for it in range(max_iters):
        # Pick a random 1 and a random 0 to swap
        ones = np.where(h == 1.0)[0]
        zeros = np.where(h == 0.0)[0]
        if len(ones) == 0 or len(zeros) == 0:
            break
        i = ones[np.random.randint(len(ones))]
        j = zeros[np.random.randint(len(zeros))]
        if i == j:
            continue
        
        old_i, old_j = h[i], h[j]
        h[i], h[j] = h[j], h[i]
        new_score = score(h)
        if new_score < best_score:
            best_score = new_score
            best_h = h.copy()
        else:
            h[i], h[j] = old_i, old_j
    
    return best_h, best_score

def construct():
    np.random.seed(42)
    deadline = time.time() + 140  # safety margin
    
    # Start with a random balanced binary sequence
    h = np.zeros(N)
    indices = np.random.permutation(N)
    h[indices[:N//2]] = 1.0
    
    best_h = h.copy()
    best_score = _compute_max_overlap(best_h)
    
    # Try local search with multiple restarts
    num_restarts = 20
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        # New random start
        h = np.zeros(N)
        indices = np.random.permutation(N)
        h[indices[:N//2]] = 1.0
        
        h, sc = _local_search(h, max_iters=20000)
        if sc < best_score:
            best_score = sc
            best_h = h.copy()
        
        if time.time() > deadline:
            break
    
    # Also try some structured patterns
    structured_patterns = []
    
    # Alternating
    alt = np.zeros(N)
    alt[::2] = 1.0
    if N % 2 == 0:
        structured_patterns.append(alt)
    
    # Block of 4: 0011
    block4 = np.tile([0, 0, 1, 1], N // 4)
    if len(block4) == N:
        structured_patterns.append(block4)
    
    # Block of 8: 00011110
    block8 = np.tile([0, 0, 0, 1, 1, 1, 1, 0], N // 8)
    if len(block8) == N:
        structured_patterns.append(block8)
    
    for pat in structured_patterns:
        sc = _compute_max_overlap(pat)
        if sc < best_score:
            best_score = sc
            best_h = pat.copy()
    
    return best_h
# EVOLVE-BLOCK-END
