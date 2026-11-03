"""Rewritten program for the second autocorrelation inequality task."""
import time
import numpy as np

N = 50

# Precomputed 7-element Sidon set within {0, ..., 49}
SIDON_SET = [0, 1, 4, 10, 18, 30, 46]

def _score_from_heights(h):
    """Compute the raw score: (sum h^2) / (L1 * sup |autoconv|)"""
    h = np.asarray(h, dtype=np.float64)
    if len(h) != N:
        return 0.0
    if np.any(h < 0):
        return 0.0
    l1 = np.sum(h)
    if l1 <= 0:
        return 0.0
    sum_sq = np.sum(h ** 2)
    # Autoconvolution (full)
    ac = np.convolve(h, h, mode='full')
    sup_ac = np.max(np.abs(ac))
    if sup_ac <= 0:
        return 0.0
    return sum_sq / (l1 * sup_ac)

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110.0  # safety margin before 120s limit
    best_score = 0.0
    best_heights = [1.0 / 7.0 if i in SIDON_SET else 0.0 for i in range(N)]
    best_score = _score_from_heights(best_heights)
    
    # Try some alternative Sidon-like or structured sets
    # But the 7-element Sidon set with equal weights should give score ~7.0
    # Let's verify and also try skewing weights to see if we can do better
    
    # Try exponential skewing on the Sidon set
    indices = SIDON_SET
    n_idx = len(indices)
    
    # Try various geometric progressions of weights
    for base in [0.5, 0.6, 0.7, 0.8, 0.9, 1.2, 1.5, 2.0]:
        if time.time() > deadline:
            break
        raw_weights = [base ** i for i in range(n_idx)]
        # Reverse order too
        raw_weights_rev = [base ** (n_idx - 1 - i) for i in range(n_idx)]
        for rw in [raw_weights, raw_weights_rev]:
            if time.time() > deadline:
                break
            total = sum(rw)
            h = [0.0] * N
            for idx, w in zip(indices, rw):
                h[idx] = w / total
            s = _score_from_heights(h)
            if s > best_score:
                best_score = s
                best_heights = h[:]
    
    # Try random search with time budget
    np.random.seed(42)
    iterations = 0
    while time.time() < deadline - 5.0:
        # Random support of size 5-8, random positive weights
        k = np.random.randint(5, 9)
        idx = np.random.choice(N, size=k, replace=False)
        # Random weights: mix of uniform and skewed
        if np.random.random() < 0.5:
            w = np.random.exponential(1.0, size=k)
        else:
            w = np.random.uniform(0.1, 1.0, size=k)
        w = w / np.sum(w)
        h = [0.0] * N
        for i, wv in zip(idx, w):
            h[i] = wv
        s = _score_from_heights(h)
        if s > best_score:
            best_score = s
            best_heights = h[:]
        iterations += 1
    
    return best_heights
# EVOLVE-BLOCK-END
