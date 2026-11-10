import time
import random
import numpy as np

def _max_abs_ap_sum(signs):
    """
    Compute the maximum absolute sum over any arithmetic progression
    contained in [1, n].
    signs: list or numpy array of length n, values -1 or 1.
    Returns the max absolute sum.
    """
    n = len(signs)
    max_sum = 0
    # We use 1-based indexing for the problem statement, but 0-based for array.
    # AP: start s (0-based), step d, length L.
    # Sum = signs[s] + signs[s+d] + ... + signs[s+(L-1)d]
    
    # For each step d, we look at chains of indices with that step.
    # A chain is determined by start s in [0, d-1].
    # The chain is s, s+d, s+2d, ... < n
    # We compute prefix sums of the chain.
    # The sum of any sub-AP in the chain is P[j] - P[i].
    # We want max |P[j] - P[i]|.
    
    for d in range(1, n + 1):
        for s in range(d):
            # Get the chain
            # Indices: s, s+d, s+2d, ...
            # We can iterate and compute running sum, but we need max diff of prefix sums.
            # Let P[0] = 0.
            # P[k] = sum of first k elements in chain.
            # We want max |P[j] - P[i]|.
            # This is max(P) - min(P) over the prefix sums of the chain.
            
            # Optimization: just iterate and track min and max prefix sum.
            p = 0
            min_p = 0
            max_p = 0
            # Iterate through the chain
            idx = s
            while idx < n:
                p += signs[idx]
                if p < min_p: min_p = p
                if p > max_p: max_p = p
                idx += d
            
            current_max = max_p - min_p
            if current_max > max_sum:
                max_sum = current_max
                # Early exit if we already found a large value? 
                # We want to minimize, so if we are evaluating a candidate, 
                # we can stop if max_sum exceeds the current best known.
                
    return max_sum

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s limit to be safe
    
    n = 256
    rng = random.Random(42)
    
    # Initial guess: random
    signs = [rng.choice((-1, 1)) for _ in range(n)]
    best_signs = signs[:]
    best_score = _max_abs_ap_sum(best_signs)
    
    # Local Search
    # We try flipping a single bit.
    # To speed up, we can just do random restarts or a simple hill climb.
    # Given O(n^2) evaluation is ~10^5 ops, we can do ~10^4 evals/sec.
    # 50s * 10^4 = 500,000 evals.
    
    # Strategy: Hill Climbing with random restarts if stuck.
    # Or just simple local search.
    
    current_signs = best_signs[:]
    current_score = best_score
    
    no_improve_count = 0
    max_no_improve = 1000  # Restart if stuck for this many steps
    
    while time.time() < deadline:
        if no_improve_count > max_no_improve:
            # Restart
            current_signs = [rng.choice((-1, 1)) for _ in range(n)]
            current_score = _max_abs_ap_sum(current_signs)
            if current_score < best_score:
                best_signs = current_signs[:]
                best_score = current_score
            no_improve_count = 0
            continue
        
        # Pick a random index to flip
        idx = rng.randint(0, n - 1)
        old_val = current_signs[idx]
        current_signs[idx] = -old_val
        
        new_score = _max_abs_ap_sum(current_signs)
        
        if new_score < current_score:
            current_score = new_score
            no_improve_count = 0
            if new_score < best_score:
                best_score = new_score
                best_signs = current_signs[:]
        else:
            # Revert
            current_signs[idx] = old_val
            no_improve_count += 1
            
    return best_signs
