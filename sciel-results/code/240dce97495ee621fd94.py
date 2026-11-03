"""Second autocorrelation inequality - combinatorial Sidon set search."""
import numpy as np
from itertools import combinations
import time

N = 50
DEADLINE = 115.0  # seconds, safety margin
START_TIME = time.time()

def time_remaining():
    return DEADLINE - (time.time() - START_TIME)

def autoconvolution_peak(h):
    """Compute the supremum norm of the autoconvolution of h using FFT."""
    n = len(h)
    size = 2 * n - 1
    H = np.fft.rfft(h, n=size)
    C = np.fft.irfft(H * np.conj(H), n=size)
    return np.max(C)

def compute_score(h):
    """Compute the ratio: ||h||_2^2 / (||h||_1 * ||h*h||_inf)"""
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    peak = autoconvolution_peak(h)
    if peak < 1e-15:
        return 0.0
    l2_sq = np.sum(h ** 2)
    return l2_sq / (l1 * peak)

def count_pairwise_sum_collisions(indices):
    """Count how many times each pairwise sum appears among the given indices.
    Returns the maximum collision count."""
    if len(indices) < 2:
        return 1
    sums = {}
    for i in range(len(indices)):
        for j in range(i, len(indices)):
            s = indices[i] + indices[j]
            sums[s] = sums.get(s, 0) + 1
    return max(sums.values()) if sums else 1

def is_sidon_set(indices):
    """Check if a set of indices forms a Sidon set (all pairwise sums distinct or at most 2)."""
    if len(indices) < 2:
        return True
    sums = set()
    for i in range(len(indices)):
        for j in range(i, len(indices)):
            s = indices[i] + indices[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def find_sidon_subset(max_size, max_val, seed=0):
    """Find a large Sidon subset of {0, 1, ..., max_val} using greedy + random restarts."""
    best_set = set()
    best_score = 0
    rng = np.random.RandomState(seed)
    
    for attempt in range(50):
        if time_remaining() < 2.0:
            break
        
        # Greedy construction with random perturbation
        current = []
        used_sums = set()
        
        # Random ordering with bias toward spread-out values
        candidates = list(range(max_val + 1))
        rng.shuffle(candidates)
        
        for idx in candidates:
            if len(current) >= max_size:
                break
            # Check if adding idx keeps Sidon property
            can_add = True
            for existing in current:
                s = idx + existing
                if s in used_sums:
                    can_add = False
                    break
            if can_add:
                current.append(idx)
                for existing in current[:-1]:
                    used_sums.add(idx + existing)
                used_sums.add(idx + idx)
        
        score = len(current)
        if score > best_score:
            best_score = score
            best_set = set(current)
    
    return list(best_set)

def optimize_weights_on_support(support):
    """Given a support (set of indices), optimize the weights to maximize the ratio.
    For a Sidon set with uniform weights, the ratio is approximately K/2 where K is the size.
    We can try small perturbations to improve."""
    K = len(support)
    if K == 0:
        return None, 0.0
    
    # Start with uniform weights
    h = np.zeros(N)
    for idx in support:
        h[idx] = 1.0 / K
    
    best_h = h.copy()
    best_score = compute_score(h)
    
    # Try simple coordinate descent
    for _ in range(20):
        if time_remaining() < 1.0:
            break
        improved = False
        for i in range(N):
            if h[i] == 0:
                continue
            # Try small perturbation
            for delta in [0.01, -0.01, 0.05, -0.05]:
                h_new = h.copy()
                h_new[i] = max(0, h[i] + delta)
                if np.sum(h_new) < 1e-15:
                    continue
                score = compute_score(h_new)
                if score > best_score:
                    best_score = score
                    best_h = h_new.copy()
                    improved = True
        h = best_h
        if not improved:
            break
    
    return best_h, best_score

def construct():
    """Construct the optimal heights using combinatorial Sidon set search."""
    best_h = np.ones(N) / N
    best_score = compute_score(best_h)
    
    # Strategy 1: Find large Sidon sets and use uniform weights
    for max_size in [20, 25, 30, 35, 40]:
        if time_remaining() < 5.0:
            break
        
        for seed in range(10):
            if time_remaining() < 2.0:
                break
            
            support = find_sidon_subset(max_size, N - 1, seed=seed * 100 + max_size)
            if len(support) >= 10:
                h, score = optimize_weights_on_support(support)
                if h is not None and score > best_score:
                    best_score = score
                    best_h = h.copy()
    
    # Strategy 2: Random sparse supports with optimization
    rng = np.random.RandomState(42)
    for _ in range(30):
        if time_remaining() < 3.0:
            break
        
        # Random sparse support
        K = rng.randint(15, 35)
        support = list(rng.choice(N, size=K, replace=False))
        h, score = optimize_weights_on_support(support)
        if h is not None and score > best_score:
            best_score = score
            best_h = h.copy()
    
    # Strategy 3: Structured patterns (arithmetic progressions, etc.)
    for step in [2, 3, 4, 5]:
        if time_remaining() < 2.0:
            break
        support = list(range(0, N, step))
        h, score = optimize_weights_on_support(support)
        if h is not None and score > best_score:
            best_score = score
            best_h = h.copy()
    
    # Final: ensure non-negative
    best_h = np.maximum(0, best_h)
    if np.sum(best_h) < 1e-15:
        best_h = np.ones(N) / N
    
    return list(best_h)

if __name__ == "__main__":
    h = construct()
    print(f"Score: {compute_score(np.array(h))}")
