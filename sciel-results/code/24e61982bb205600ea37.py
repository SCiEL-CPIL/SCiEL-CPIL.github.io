"""Rewritten program: DP for globally optimal non-decreasing-gap prime subsequence."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    deadline = time.time() + 80.0  # safety margin before 90s kill

    # Sieve of Eratosthenes for primes <= 10000
    limit = 10000
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = 0

    primes = np.array([p for p in range(2, limit + 1) if is_prime[p]], dtype=np.int32)
    n = len(primes)

    if n == 0:
        return []

    # DP arrays
    # best_len[i] = max length of valid subsequence ending at primes[i]
    # best_gap[i] = the last gap in that optimal subsequence (for tie-breaking, prefer smaller)
    # prev[i] = index of predecessor in the optimal subsequence
    best_len = np.ones(n, dtype=np.int32)
    best_gap = np.zeros(n, dtype=np.int32)   # last gap; 0 means "no predecessor"
    prev = np.full(n, -1, dtype=np.int32)

    for i in range(1, n):
        pi = primes[i]
        
        if time.time() > deadline:
            break

        # Vectorized check for all j < i
        # gaps = pi - primes[:i]
        # valid if gap >= best_gap[j]
        
        gaps = pi - primes[:i]
        valid_mask = gaps >= best_gap[:i]
        
        if not np.any(valid_mask):
            continue

        # Among valid indices, we want:
        # 1. Max best_len[j]
        # 2. Tie-break: Min new gap (pi - primes[j]), which means Max primes[j]
        
        valid_indices = np.where(valid_mask)[0]
        lens = best_len[valid_indices]
        max_len_val = np.max(lens)
        
        # Find candidates with max_len_val
        cand_mask = lens == max_len_val
        candidates = valid_indices[cand_mask]
        
        if len(candidates) > 0:
            # Choose candidate with max prime value (min gap)
            # If multiple have same max prime (impossible for distinct primes), pick any
            best_j = candidates[np.argmax(primes[candidates])]
            
            new_len = max_len_val + 1
            new_gap = pi - primes[best_j]
            
            # Update state
            # Since we process i in order, best_len[i] is currently 1 (or potentially updated by previous logic? No, i is unique target)
            # But wait, we might have multiple updates? No, we compute the best predecessor for i in one go.
            # So we just set it.
            best_len[i] = new_len
            best_gap[i] = new_gap
            prev[i] = best_j

    # Find the index with maximum best_len
    # If tie, we don't strictly need a specific tie-break for the final answer length, 
    # but let's pick the one that appears last or first? 
    # The task asks to maximize length. Any max length sequence is valid.
    # However, if we want to be consistent, we can pick the one with max length.
    max_len_global = np.max(best_len)
    best_indices = np.where(best_len == max_len_global)[0]
    
    # Pick the last one? Or the one with smallest last gap? 
    # It doesn't affect the score (length), but let's pick the one with smallest last gap for robustness if multiple exist.
    # Actually, just picking the first max is fine.
    best_idx = best_indices[0]

    # Reconstruct the sequence
    seq = []
    cur = int(best_idx)
    while cur != -1:
        seq.append(int(primes[cur]))
        cur = int(prev[cur])
    seq.reverse()

    return seq
# EVOLVE-BLOCK-END
