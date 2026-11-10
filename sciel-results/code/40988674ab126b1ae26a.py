"""Rewritten program: local search minimizing max |sum| over all APs in [1,256]."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

_N = 256

def _max_abs_ap_sum(signs):
    """
    Compute the maximum absolute signed sum over all arithmetic progressions
    contained in indices 0..n-1.
    
    For each step d >= 1 and each residue r in 0..d-1, we consider the sequence
    signs[r], signs[r+d], signs[r+2d], ... and track the running sum.
    The max absolute value over all such running sums is returned.
    
    This is O(n^2) in the worst case but with n=256 it's fast enough.
    """
    n = len(signs)
    max_disc = 1  # singletons contribute +/-1
    
    # For each step d from 1 to n-1 (d=n gives only singletons which we already have)
    # Actually d can go up to n-1 for progressions of length >= 2
    # For d=1, we have one sequence of length n: the whole array
    # For d=2, we have two sequences: even and odd indices
    # etc.
    
    for d in range(1, n):
        # For each start residue r in 0..d-1
        for r in range(d):
            # Sequence: r, r+d, r+2d, ...
            s = 0
            idx = r
            while idx < n:
                s += signs[idx]
                abs_s = abs(s)
                if abs_s > max_disc:
                    # Early exit if we already know we're worse than current best
                    # But we need the exact max for comparison, so continue
                    # We can't early exit per-sequence because we need the global max
                    pass
                idx += d
            abs_s = abs(s)
            if abs_s > max_disc:
                max_disc = abs_s
    
    return max_disc

def _compute_max_ap_sum_and_worst(signs):
    """
    Return (max_abs_sum, worst_indices) where worst_indices is a list of (start, step, length)
    tuples for progressions achieving the max absolute sum (or close to it).
    """
    n = len(signs)
    max_disc = 1
    worst_list = []
    
    for d in range(1, n):
        for r in range(d):
            s = 0
            idx = r
            length = 0
            max_in_seq = 0
            worst_len_in_seq = 0
            while idx < n:
                s += signs[idx]
                length += 1
                abs_s = abs(s)
                if abs_s > max_in_seq:
                    max_in_seq = abs_s
                    worst_len_in_seq = length
                idx += d
            if max_in_seq > max_disc:
                max_disc = max_in_seq
                worst_list = [(r, d, worst_len_in_seq)]
            elif max_in_seq == max_disc and max_disc > 1:
                worst_list.append((r, d, worst_len_in_seq))
    
    return max_disc, worst_list

def _local_search(signs, max_time, rng):
    """
    Local search: repeatedly flip a sign that reduces the max absolute AP sum.
    Use simulated annealing or hill climbing with random restarts on stuck.
    """
    n = len(signs)
    signs = signs.copy()
    current_max = _max_abs_ap_sum(signs)
    best_max = current_max
    best_signs = signs.copy()
    
    start_time = time.time()
    deadline = start_time + max_time - 5  # 5s safety margin
    
    # Precompute for speed: we'll try flips and evaluate
    # For efficiency, we can do a simple hill-climbing: try each position, flip, check if max decreases
    
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline:
        # Try to find an improving flip
        improved = False
        # Shuffle positions for diversity
        positions = list(range(n))
        rng.shuffle(positions)
        
        for pos in positions:
            if time.time() > deadline:
                break
            
            # Flip position pos
            signs[pos] = -signs[pos]
            new_max = _max_abs_ap_sum(signs)
            
            if new_max < current_max:
                current_max = new_max
                improved = True
                no_improve_count = 0
                if current_max < best_max:
                    best_max = current_max
                    best_signs = signs.copy()
                break  # Restart scan from beginning after improvement
            else:
                # Revert
                signs[pos] = -signs[pos]
        
        if not improved:
            no_improve_count += 1
            if no_improve_count >= max_no_improve:
                # Random restart: perturb the current best
                signs = best_signs.copy()
                # Flip a few random positions
                for _ in range(rng.randint(1, 5)):
                    p = rng.randint(0, n - 1)
                    signs[p] = -signs[p]
                current_max = _max_abs_ap_sum(signs)
                no_improve_count = 0
    
    return best_signs, best_max

def construct():
    """
    Construct 256 signs (-1 or +1) minimizing the maximum absolute signed sum
    over any arithmetic progression in [1, 256].
    """
    start_time = time.time()
    total_budget = 85  # Leave 5s margin for the 90s limit
    
    rng = random.Random(42)
    
    # Initial random solution
    signs = np.array([rng.choice((-1, 1)) for _ in range(_N)], dtype=np.int8)
    
    # Do a few local searches with different initializations
    best_signs = signs.copy()
    best_max = _max_abs_ap_sum(signs)
    
    # Run local search with time budget
    remaining = total_budget - (time.time() - start_time)
    if remaining > 10:
        # Single local search run with most of the budget
        t0 = time.time()
        result_signs, result_max = _local_search(signs, remaining - 5, rng)
        elapsed = time.time() - t0
        if result_max < best_max:
            best_max = result_max
            best_signs = result_signs
        
        # If we have time, try another run from a different random start
        remaining2 = total_budget - (time.time() - start_time)
        if remaining2 > 15:
            signs2 = np.array([rng.choice((-1, 1)) for _ in range(_N)], dtype=np.int8)
            t0 = time.time()
            result_signs2, result_max2 = _local_search(signs2, remaining2 - 5, rng)
            if result_max2 < best_max:
                best_max = result_max2
                best_signs = result_signs2
    
    # Convert to list of Python ints
    return [int(x) for x in best_signs]

# EVOLVE-BLOCK-END
