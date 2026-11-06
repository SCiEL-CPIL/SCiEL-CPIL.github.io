"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np

N = 256

def _max_abs_aps(x):
    """Compute the maximum absolute signed sum over all arithmetic progressions.
    
    x is a list or array of length N with entries in {-1, +1}.
    """
    max_abs = 0
    # For each step d
    for d in range(1, N + 1):
        # For each residue class r in [0, d)
        for r in range(d):
            # The AP is x[r], x[r+d], x[r+2d], ...
            # Build the list of values in this AP
            chain = []
            pos = r
            while pos < N:
                chain.append(x[pos])
                pos += d
            # Now find the max absolute sum of any contiguous subsequence in chain
            # (since any AP contained in this chain starting at some index and of some length is a contiguous subsequence)
            if len(chain) == 0:
                continue
            # Kadane's algorithm for max subarray sum and min subarray sum
            max_so_far = 0
            min_so_far = 0
            current_max = 0
            current_min = 0
            for v in chain:
                current_max = max(0, current_max + v)
                current_min = min(0, current_min + v)
                if current_max > max_so_far:
                    max_so_far = current_max
                if abs(current_min) > abs(min_so_far):
                    min_so_far = current_min
            cand = max(max_so_far, abs(min_so_far))
            if cand > max_abs:
                max_abs = cand
    return max_abs

def _evaluate(x):
    return _max_abs_aps(x)

def _flip(x, i):
    x[i] = -x[i]

def construct():
    start_time = time.time()
    deadline = 55.0  # safety margin before 60s or 90s kill
    
    # Initial random seed
    rng = random.Random(42)
    
    best_x = [rng.choice((-1, 1)) for _ in range(N)]
    best_score = _evaluate(best_x)
    
    # Also try a few structured initializations
    # Alternating
    x_alt = [1 if i % 2 == 0 else -1 for i in range(N)]
    s_alt = _evaluate(x_alt)
    if s_alt < best_score:
        best_score = s_alt
        best_x = x_alt[:]
    
    # Random with different seeds
    for seed in [171, 123, 456, 789, 314, 271]:
        x_try = [random.Random(seed).choice((-1, 1)) for _ in range(N)]
        s_try = _evaluate(x_try)
        if s_try < best_score:
            best_score = s_try
            best_x = x_try[:]
    
    # Simulated Annealing
    current_x = best_x[:]
    current_score = best_score
    
    T_start = 5.0
    T_end = 0.1
    # Number of iterations we might do; we'll use time as the guard
    iter_count = 0
    
    while time.time() - start_time < deadline:
        T = T_start * (T_end / T_start) ** (iter_count / 100000.0) if iter_count < 100000 else T_end
        T = max(T, T_end)
        
        # Pick a random position to flip
        i = rng.randint(0, N - 1)
        _flip(current_x, i)
        new_score = _evaluate(current_x)
        
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_x = current_x[:]
        else:
            # Accept with probability exp(-(delta)/T)
            delta = new_score - current_score
            if delta > 0:
                prob = math.exp(-delta / T) if T > 0 else 0.0
                if rng.random() < prob:
                    current_score = new_score
                else:
                    _flip(current_x, i)  # revert
            else:
                # Shouldn't happen since delta >= 0 when new_score >= current_score
                pass
        
        iter_count += 1
        # Small sleep to avoid burning CPU too hard? No, we need speed.
        # But let's check time frequently.
    
    return best_x
# EVOLVE-BLOCK-END
