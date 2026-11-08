import time
from math import log, floor
import random
import numpy as np
from itertools import combinations, product

def score_set(U):
    """Compute the score for a set U."""
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
    
    # Use numpy for faster computation
    U_arr = np.array(U, dtype=np.int64)
    
    # Compute all pairwise sums (i <= j to avoid duplicates in the set)
    # For sums: all pairs (i,j) where i <= j
    i_idx = np.repeat(np.arange(n), n)
    j_idx = np.tile(np.arange(n), n)
    # Only keep i <= j
    mask = i_idx <= j_idx
    i_idx = i_idx[mask]
    j_idx = j_idx[mask]
    
    sums = U_arr[i_idx] + U_arr[j_idx]
    sum_count = len(np.unique(sums))
    
    # Compute all pairwise differences (i < j for positive, then add negatives and 0)
    i_idx2 = np.repeat(np.arange(n), n)
    j_idx2 = np.tile(np.arange(n), n)
    mask2 = i_idx2 < j_idx2
    i_idx2 = i_idx2[mask2]
    j_idx2 = j_idx2[mask2]
    
    diffs_pos = U_arr[j_idx2] - U_arr[i_idx2]
    # Include 0 and all negative differences
    all_diffs = np.concatenate([np.array([0]), diffs_pos, -diffs_pos])
    diff_count = len(np.unique(all_diffs))
    
    if sum_count == 0:
        return -1
        
    ratio = diff_count / sum_count
    if ratio <= 1:
        return -1
        
    max_u = max(U)
    D = 2 * max_u + 1
    if D <= 1:
        return -1
        
    return 1 + log(ratio) / log(D)

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit

    best_set = [0, 1]
    best_score = score_set(best_set)

    # Direct structured constructions based on the task's mathematical structure.
    # The score is 1 + log(|U-U|/|U+U|) / log(2*max(U)+1).
    # We want to maximize |U-U| / |U+U|.
    # For a set U, |U-U| is at most (2*max(U)+1). |U+U| is at least |U|.
    # If U is a Sidon set (all pairwise sums distinct), |U+U| = n(n+1)/2.
    # If U is "additive", |U+U| can be much smaller.
    # But we want the RATIO to be large. So we want |U-U| large and |U+U| small?
    # No, |U-U| is usually larger than |U+U| for typical sets because differences can be negative.
    # Actually, |U-U| >= |U+U| is not always true, but for sets containing 0 and non-negative, 
    # the positive differences are a subset of the sums? No.
    # Let's look at the ratio R = |U-U| / |U+U|.
    # For an arithmetic progression {0, 1, ..., n-1}:
    #   U+U = {0, 1, ..., 2n-2}, size 2n-1.
    #   U-U = {-(n-1), ..., n-1}, size 2n-1.
    #   Ratio = 1. Score = 1 + log(1)/log(...) = 1.
    # For a geometric progression {0, 1, b, b^2, ..., b^{m-1}}:
    #   Sums are distinct for b > 1? If b is large enough, yes.
    #   |U+U| = n(n+1)/2.
    #   |U-U| = n(n+1)/2 - 1 + 1 (including 0) ... actually all differences are distinct too if b is large.
    #   So ratio is close to 1.
    # 
    # We need |U-U| to be significantly LARGER than |U+U|.
    # This happens when the set is "spread out" such that many differences are unique but sums collide?
    # Or many differences are unique and sums are also unique? 
    # Wait, if all sums are distinct, |U+U| = n(n+1)/2.
    # If all differences are distinct, |U-U| = n(n-1) + 1.
    # Ratio = (n^2 - n + 1) / (n^2 + n) / 2 * 2 = (n^2 - n + 1) / (n^2 + n).
    # This is < 1 for n > 1.
    # So we need |U-U| > |U+U|.
    # This requires collisions in sums but not in differences? Or just generally more difference values.
    # Note that |U-U| includes negative numbers. |U+U| does not (if U >= 0).
    # |U-U| is always at least 2*max(U)-min(U) + 1.
    # |U+U| is at most (max(U)+min(U)) - (min(U+min(U)) - min(U)) + 1? No.
    # Max sum = 2*max(U). Min sum = 0.
    # If the set is "dense" in the sense that it covers an interval, |U+U| is about 2*max(U).
    # And |U-U| is about 2*max(U).
    # Ratio ~ 1.
    #
    # Consider U = {0, 1, 3, 7, 15, ...} (Mersenne-like).
    # Sums: 0, 1, 3, 7, 15, 2, 4, 8, 16, 10, 18, ...
    # Let's just run the search.

    # Initial seeds based on different structural hypotheses
    seeds = [
        [0, 1, 4, 9, 16, 25, 36, 49, 64, 81], # Squares
        [0, 1, 2, 4, 8, 16, 32, 64, 128, 256], # Powers of 2
        [0, 1, 3, 7, 15, 31, 63, 127, 255], # Mersenne-like
        [0, 1, 3, 4, 9, 10, 12, 13, 28, 29, 31, 32], # Small Sidon-like
        [0, 1, 4, 10, 20, 35, 56, 84, 120], # Triangular
        [0, 1, 4, 10, 18, 28, 40, 54, 70], # Quadratic-ish
        [0, 1, 3, 9, 27, 81, 243, 729], # Powers of 3
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], # Interval
        [0, 1, 3, 6, 10, 15, 21, 28], # Triangular
        [0, 1, 2, 5, 7, 10, 13, 17, 21], # Fibonacci-ish
    ]

    for cand in seeds:
        s = score_set(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)

    # Exhaustive search for small n to find local optima
    def exhaustive_search(n, max_val):
        best_s = -1
        best_c = None
        range_vals = range(1, max_val + 1)
        for comb in combinations(range_vals, n - 1):
            U = [0] + list(comb)
            s = score_set(U)
            if s > best_s:
                best_s = s
                best_c = U
        return best_s, best_c

    # Run exhaustive search for small n within time budget
    if time.time() < deadline - 60:
        # n=3, max_val=60
        s, c = exhaustive_search(3, 60)
        if s > best_score:
            best_score = s
            best_set = c
            
    if time.time() < deadline - 40:
        # n=4, max_val=40
        s, c = exhaustive_search(4, 40)
        if s > best_score:
            best_score = s
            best_set = c

    if time.time() < deadline - 20:
        # n=5, max_val=25
        s, c = exhaustive_search(5, 25)
        if s > best_score:
            best_score = s
            best_set = c

    # Local Search with Simulated Annealing
    def local_search_sa(U, deadline):
        U = sorted(set(U))
        current_score = score_set(U)
        best_score_local = current_score
        best_set_local = list(U)
        
        T = 10.0
        cooling = 0.995
        max_iterations = 50000
        
        for i in range(max_iterations):
            if time.time() > deadline - 5:
                break
                
            # Choose a move
            move_type = random.random()
            new_U = list(U)
            
            if move_type < 0.4:
                # Replace an element
                if len(new_U) < 2:
                    continue
                idx = random.randint(1, len(new_U) - 1)
                old_val = new_U[idx]
                # Pick a new value: random near old, or random in range
                if random.random() < 0.7:
                    delta = random.randint(-10, 10)
                    new_val = old_val + delta
                else:
                    new_val = random.randint(1, max(10, max(new_U) * 2))
                
                if new_val <= 0:
                    continue
                if new_val in U:
                    continue
                    
                new_U[idx] = new_val
                new_U = sorted(set(new_U))
                
            elif move_type < 0.7:
                # Add an element
                if len(new_U) < 30:
                    if random.random() < 0.5:
                        # Near max
                        new_val = random.randint(max(1, max(new_U)-5), max(new_U)+20)
                    else:
                        new_val = random.randint(1, max(10, max(new_U)))
                    
                    if new_val in U:
                        continue
                    new_U.append(new_val)
                    new_U = sorted(set(new_U))
                    
            else:
                # Remove an element
                if len(new_U) > 2:
                    idx = random.randint(1, len(new_U) - 1)
                    new_U.pop(idx)
                    new_U = sorted(set(new_U))
                    
            if len(new_U) < 2:
                continue
                
            new_score = score_set(new_U)
            
            # Simulated annealing acceptance
            if new_score > current_score:
                U = new_U
                current_score = new_score
                if current_score > best_score_local:
                    best_score_local = current_score
                    best_set_local = list(U)
            else:
                delta = new_score - current_score
                if random.random() < np.exp(delta / T):
                    U = new_U
                    current_score = new_score
                    
            T *= cooling
            
        return best_score_local, best_set_local

    # Run local search from best seed and other good seeds
    search_seeds = [best_set]
    for cand in seeds:
        if cand != best_set:
            s = score_set(cand)
            if s > best_score * 0.9:
                search_seeds.append(cand)
                
    random.shuffle(search_seeds)
    
    for seed in search_seeds:
        if time.time() > deadline - 10:
            break
        local_score, local_set = local_search_sa(seed, deadline)
        if local_score > best_score:
            best_score = local_score
            best_set = local_set

    # Final polish: greedy hill climbing on the best set
    if time.time() < deadline - 2:
        current = list(best_set)
        current_score = best_score
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            n = len(current)
            
            # Try replacing each element with nearby values
            for i in range(1, n):
                if time.time() > deadline - 1:
                    break
                old_val = current[i]
                for delta in range(-15, 16):
                    if delta == 0:
                        continue
                    new_val = old_val + delta
                    if new_val <= 0 or new_val in current:
                        continue
                    new_set = [current[j] if j != i else new_val for j in range(n)]
                    new_set = sorted(set(new_set))
                    if len(new_set) != n:
                        continue
                    new_score = score_set(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        improved = True
                        break
                if improved:
                    break
            
            # Try adding an element
            if not improved and n < 30:
                max_val = max(current)
                for new_val in range(max(1, max_val-5), max_val+15):
                    if new_val in current:
                        continue
                    new_set = sorted(current + [new_val])
                    new_score = score_set(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        improved = True
                        break
            
            # Try removing an element
            if not improved and n > 3:
                for i in range(1, n):
                    new_set = [current[j] for j in range(n) if j != i]
                    new_score = score_set(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        improved = True
                        break

        if current_score > best_score:
            best_score = current_score
            best_set = current

    return best_set

if __name__ == "__main__":
    U = construct()
    print("Result:", U)
    print("Score:", score_set(U))
