"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _compute_max_abs_sum(signs):
    """Compute the maximum absolute signed sum over any arithmetic progression
    contained in [1, n]. signs is a 1-indexed list (index 0 unused) of ±1."""
    n = len(signs) - 1
    best = 0
    # For each step d, consider all starts s such that s, s+d, ... <= n
    for d in range(1, n + 1):
        # For a fixed step d, the APs are determined by start s and length L.
        # We can iterate over all starts s in [1, n] and extend as far as possible.
        # For each start s, the values are signs[s], signs[s+d], signs[s+2d], ...
        # We need the max abs prefix sum over these sequences.
        s = 1
        while s <= n:
            # Get the sequence for this start and step
            seq = []
            idx = s
            while idx <= n:
                seq.append(signs[idx])
                idx += d
            # Compute max abs prefix sum
            cur = 0
            for v in seq:
                cur += v
                if abs(cur) > best:
                    best = abs(cur)
            s += 1
    return best

def _fast_max_abs_sum(signs, n):
    """Faster computation using numpy where possible, but fallback to pure python for correctness.
    Actually, let's use a smarter approach: precompute for each step d, use vectorized ops if possible,
    but given n=256, pure python might be fast enough. Let's estimate:
    Sum over d=1..256 of (n - d + 1) starts * average length ~ sum_d (n-d+1)^2/2 roughly.
    Actually for each d, there are (n-d+1) starts, and each start has length ~ n/d.
    Total operations ~ sum_d (n-d+1) * (n/d) ~ n^2 sum 1/d ~ n^2 log n ~ 256^2 * 6 ~ 400k, which is fine.
    """
    best = 0
    for d in range(1, n + 1):
        # For step d, iterate over all valid starts
        for s in range(1, n - d + 2):
            cur = 0
            idx = s
            while idx <= n:
                cur += signs[idx]
                if cur > best:
                    best = cur
                elif -cur > best:
                    best = -cur
                idx += d
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    n = 256
    
    # Start with a random initialization
    rng = random.Random(42)
    signs = [0] + [rng.choice((-1, 1)) for _ in range(n)]
    
    current_best = _fast_max_abs_sum(signs, n)
    best_signs = signs[:]
    
    # Local search: flip a single position and see if it improves
    # Use simulated annealing or simple hill climbing with restarts
    
    # First, do a quick hill climb
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(1, n + 1):
            if time.time() > deadline:
                break
            # Flip position i
            old_val = signs[i]
            signs[i] = -old_val
            new_best = _fast_max_abs_sum(signs, n)
            if new_best < current_best:
                current_best = new_best
                best_signs = signs[:]
                improved = True
            else:
                signs[i] = old_val
    
    # Now do simulated annealing for the remaining time
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    sa_signs = best_signs[:]
    sa_best = current_best
    sa_best_signs = sa_signs[:]
    
    while time.time() < deadline:
        # Pick a random position to flip
        i = rng.randint(1, n)
        old_val = sa_signs[i]
        sa_signs[i] = -old_val
        
        new_best = _fast_max_abs_sum(sa_signs, n)
        delta = new_best - sa_best
        
        if delta <= 0:
            sa_best = new_best
            if new_best < current_best:
                current_best = new_best
                best_signs = sa_signs[:]
        else:
            # Accept with probability exp(-delta/T)
            if T > T_min and rng.random() < np.exp(-delta / T):
                sa_best = new_best
            else:
                sa_signs[i] = old_val
        
        T *= cooling
        
        # If we've cooled enough, restart with a fresh random init or reheat
        if T < T_min:
            # Check time
            if time.time() > deadline - 5.0:
                break
            # Reheat
            T = 1.0
            # Occasionally restart from best
            if rng.random() < 0.3:
                sa_signs = sa_best_signs[:]
                sa_best = current_best
    
    return best_signs[1:]
# EVOLVE-BLOCK-END
