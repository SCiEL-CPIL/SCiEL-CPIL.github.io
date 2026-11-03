import math
import time
import random
import numpy as np

def _score_of(U):
    """Calculate the score for a given set U."""
    if not U:
        return 0.0
    U = sorted(set(U))
    if 0 not in U:
        U = [0] + U
    max_u = max(U)
    if max_u == 0:
        return 1.0

    # Use numpy for fast sum/diff computation if U is not too large
    # For very large U, the O(N^2) set construction in pure Python might be slower
    # than numpy vectorization, but memory is a concern.
    # Let's stick to a hybrid: try numpy, fallback to python if memory error or too slow?
    # Actually, for N up to ~100, numpy is much faster.
    # |U-U| and |U+U| can be computed via unique sums/diffs.
    
    U_arr = np.array(U, dtype=np.int64)
    
    # To avoid O(N^2) memory explosion for large N, we might need to be careful.
    # But N is likely small (< 20) for optimal sets in this range.
    # If N is large, max_u is likely large too, making the log denominator large,
    # which might not be optimal anyway.
    
    # Compute all pairwise sums and diffs
    # This creates arrays of size N^2
    if len(U) > 200:
        # Fallback to python for huge sets to avoid memory issues, though unlikely to be optimal
        sums = set()
        diffs = set()
        for a in U:
            for b in U:
                sums.add(a + b)
                diffs.add(a - b)
        size_sum = len(sums)
        size_diff = len(diffs)
    else:
        # Numpy approach
        # Sums: U_i + U_j
        # Diffs: U_i - U_j
        # We can use broadcasting
        # This is O(N^2) memory. For N=100, 10k elements. Fine.
        # For N=1000, 1M elements. ~8MB. Fine.
        
        # To save memory, we can compute in chunks or just rely on N being small.
        # Let's assume N < 500 for numpy path.
        
        # Efficient way:
        # sums = np.unique(U_arr[:, None] + U_arr[None, :])
        # diffs = np.unique(U_arr[:, None] - U_arr[None, :])
        
        # However, creating the full matrix might be slow for N~100+.
        # Alternative: use set comprehension which is often fast in Python for N < 100.
        # Let's benchmark mentally: Python set comp is ~1us per op? 100*100 = 10k ops = 10ms.
        # Numpy: 10k ops + unique sort. Sort is O(M log M).
        
        # Let's use the Python set approach as it's simpler and robust for the expected N range.
        # Optimization: If we suspect N is small, Python is fine.
        
        sums = set()
        diffs = set()
        # Unroll inner loop for speed?
        for a in U:
            # a + b for all b
            for b in U:
                sums.add(a + b)
                diffs.add(a - b)
        
        size_sum = len(sums)
        size_diff = len(diffs)

    if size_sum == 0:
        return 0.0
        
    ratio = size_diff / size_sum
    if ratio <= 0:
        return 0.0
        
    denom = math.log(2 * max_u + 1)
    if denom == 0:
        return 1.0
        
    return 1.0 + math.log(ratio) / denom

def _local_search(U, deadline, best_set, best_score):
    """Perform local search starting from U."""
    U = sorted(set(U))
    if 0 not in U:
        U = [0] + U
    current_score = _score_of(U)
    
    if current_score > best_score:
        best_score = current_score
        best_set = U[:]
        
    N = len(U)
    
    # Try to add, remove, or modify elements
    # We will use a hill-climbing approach with random restarts of the modification strategy
    
    max_val = max(U) if U else 0
    # Search space for new values: up to some multiple of max_val or fixed large number
    # To maximize score, we want high ratio and small max_val.
    # Ratio is |U-U|/|U+U|.
    # For arithmetic progressions, ratio is small.
    # For "random" looking sets, ratio is closer to 1?
    # Actually, for generic sets, |U+U| ~ |U|^2/2 and |U-U| ~ |U|^2/2, so ratio ~ 1.
    # We want ratio > 1?
    # Wait, |U-U| is always >= |U+U|?
    # No. U-U contains negative numbers too.
    # If U is non-negative, U-U contains both positive and negative.
    # U+U contains only non-negative.
    # Symmetry: |U-U| = 1 + 2 * |{u-v : u>v, u,v in U}|
    # |U+U| = 1 + 2 * |{u+v : u>v, u,v in U}| + |{2u : u in U}|
    # Generally, for sets with 0, |U-U| is often larger than |U+U| because differences can be distinct more easily?
    # Or is it the other way around?
    # Let's check small sets.
    # U={0,1,2}. Sums: 0,1,2,3,4 (5). Diffs: -2,-1,0,1,2 (5). Ratio 1.
    # U={0,1,3}. Sums: 0,1,2,3,4,6 (6). Diffs: -3,-2,-1,0,1,2,3 (7). Ratio 7/6 = 1.16.
    # U={0,2,3}. Sums: 0,2,3,4,5,6 (6). Diffs: -3,-2,-1,0,1,2,3 (7). Ratio 1.16.
    # U={0,1,4}. Sums: 0,1,2,4,5,8 (6). Diffs: -4,-3,-2,-1,0,1,2,3,4 (9). Ratio 9/6 = 1.5.
    # U={0,2,5}. Sums: 0,2,4,5,7,10 (6). Diffs: -5,-4,-3,-2,-1,0,1,2,3,4,5 (11). Ratio 11/6 = 1.83.
    # U={0,3,5}. Sums: 0,3,5,6,8,10 (6). Diffs: -5,-4,-3,-2,-1,0,1,2,3,4,5 (11). Ratio 1.83.
    # U={0,4,5}. Sums: 0,4,5,8,9,10 (6). Diffs: -5,-4,-3,-2,-1,0,1,2,3,4,5 (11). Ratio 1.83.
    # It seems ratio > 1 is possible and desirable.
    
    # Strategy:
    # 1. Try to replace an element with a new one.
    # 2. Try to add an element.
    # 3. Try to remove an element.
    
    max_iterations = 500
    it = 0
    
    while it < max_iterations:
        if time.time() > deadline:
            break
            
        # Choose a random move
        move_type = random.choice(['replace', 'add', 'remove'])
        
        new_U = U[:]
        
        if move_type == 'replace':
            if len(U) <= 1:
                continue
            idx = random.randint(1, len(U) - 1) # Don't replace 0
            old_val = U[idx]
            # Choose new value
            # Range: 1 to max(2*max_val, 50)
            upper = max(2 * max_val, 50)
            new_val = random.randint(1, upper)
            if new_val in U:
                continue
            new_U[idx] = new_val
            new_U.sort()
            
        elif move_type == 'add':
            if len(U) > 15: # Limit size to keep computation fast
                continue
            upper = max(2 * max_val, 50)
            new_val = random.randint(1, upper)
            if new_val in U:
                continue
            new_U.append(new_val)
            new_U.sort()
            
        elif move_type == 'remove':
            if len(U) <= 2:
                continue
            idx = random.randint(1, len(U) - 1) # Don't remove 0
            new_U.pop(idx)
            
        new_score = _score_of(new_U)
        
        if new_score > current_score:
            U = new_U
            current_score = new_score
            max_val = max(U)
            if current_score > best_score:
                best_score = current_score
                best_set = U[:]
        else:
            # Simulated annealing style? Or just restart?
            # For now, just try another move.
            pass
            
        it += 1
        
    return best_set, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    best_set = [0, 1, 4, 9, 15]
    best_score = _score_of(best_set)
    
    # 1. Exhaustive search for small sets
    # We found that small sets with "sparse" structure can have high ratio.
    # Let's search N=3 to N=7, M up to 30.
    
    for N in range(3, 8):
        # M limit depends on N to keep C(M, N-1) manageable
        # C(30, 6) = 593775. Too many.
        # C(20, 6) = 38760. Okay.
        # C(25, 5) = 53130. Okay.
        
        if N == 3:
            M_limit = 40
        elif N == 4:
            M_limit = 35
        elif N == 5:
            M_limit = 30
        elif N == 6:
            M_limit = 25
        else:
            M_limit = 20
            
        import itertools
        for M in range(N - 1 + 1, M_limit + 1):
            if time.time() > deadline:
                return best_set
                
            candidates = list(range(1, M + 1))
            # We must include 0. So we choose N-1 from 1..M
            for combo in itertools.combinations(candidates, N - 1):
                if time.time() > deadline:
                    return best_set
                    
                U = [0] + list(combo)
                s = _score_of(U)
                if s > best_score:
                    best_score = s
                    best_set = U[:]

    # 2. Local search from random starts and the best found
    random.seed(42)
    
    # Start from best_set
    best_set, best_score = _local_search(best_set, deadline, best_set, best_score)
    
    # Random restarts
    restarts = 0
    while time.time() < deadline - 5.0:
        restarts += 1
        N = random.choice([4, 5, 6, 7, 8])
        M_max = random.randint(10, 60)
        U = [0] + sorted(random.sample(range(1, M_max + 1), N - 1))
        
        best_set, best_score = _local_search(U, deadline, best_set, best_score)
        
        # Occasionally try a "greedy" construction
        if restarts % 5 == 0:
            # Try to build a set with high difference/sum ratio
            # Heuristic: pick numbers with large gaps?
            # Or numbers that are powers of 2?
            # U = {0, 1, 2, 4, 8...}
            # Let's try geometric progressions
            base = random.randint(2, 4)
            exp = random.randint(3, 6)
            U_geo = [0] + [base**i for i in range(1, exp)]
            best_set, best_score = _local_search(U_geo, deadline, best_set, best_score)

    return best_set
