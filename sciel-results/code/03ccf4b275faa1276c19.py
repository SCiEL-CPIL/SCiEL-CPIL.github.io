import numpy as np
import time

def compute_P(support, weights):
    n = len(support)
    # Normalize weights
    w = weights / np.sum(weights)
    
    # Compute all pairwise sums s_i + s_j
    # s_i + s_j < 2*s_l - s_k
    # P = sum_k sum_l w_k w_l ( sum_{i,j: s_i+s_j < 2s_l-s_k} w_i w_j )
    
    # Precompute pair sums and probs
    pair_sums = np.add.outer(support, support)
    pair_probs = np.multiply.outer(w, w)
    
    P = 0.0
    for k in range(n):
        for l in range(n):
            threshold = 2.0 * support[l] - support[k]
            # Find pairs where sum < threshold
            mask = pair_sums < threshold
            prob_sum = np.sum(pair_probs * mask)
            P += w[k] * w[l] * prob_sum
    return P

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    start_time = time.time()
    deadline = start_time + 600.0
    
    best_support = None
    best_weights = None
    best_score = -1.0
    
    # Try small n
    for n in range(2, 8):
        if time.time() > deadline - 60:
            break
            
        # Grid search or random restarts
        # For small n, we can try some structured distributions
        # 1. Uniform
        s = np.arange(n, dtype=np.float64)
        w = np.ones(n, dtype=np.float64) / n
        score = compute_P(s, w)
        if score > best_score:
            best_score = score
            best_support = s
            best_weights = w
            
        # 2. Geometric-like
        s = np.arange(n, dtype=np.float64)
        w = np.array([0.5**i for i in range(n)])
        w /= np.sum(w)
        score = compute_P(s, w)
        if score > best_score:
            best_score = score
            best_support = s
            best_weights = w
            
        # 3. Bimodal
        if n >= 2:
            s = np.zeros(n)
            s[0] = 0
            s[1:] = 10.0
            w = np.zeros(n)
            w[0] = 0.5
            w[1:] = 0.5 / (n-1)
            score = compute_P(s, w)
            if score > best_score:
                best_score = score
                best_support = s
                best_weights = w
                
    return best_support, best_weights
