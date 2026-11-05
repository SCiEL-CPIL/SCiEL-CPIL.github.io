"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution

def construct():
    n = 128
    target_low = 128
    target_high = 256
    
    # We need to find a subset A of [0, 128] such that every integer in [128, 256]
    # is a sum of two elements of A (unordered, equal summands allowed).
    # Minimize the maximum multiplicity of representations in that interval.
    
    # Represent A as a binary vector of length 129 (indices 0..128)
    # Let x[i] = 1 if i is in A, 0 otherwise.
    
    # The number of representations of a sum s by pairs (a, b) with a <= b, a+b=s, a,b in A:
    # For s in [128, 256]:
    #   count[s] = sum over a in [max(0, s-128), min(128, s//2)] of x[a] * x[s-a]
    #   but wait, unordered pairs with a <= b.
    #   Actually, for a fixed s, the pairs (a, s-a) with a <= s-a, i.e., a <= s/2.
    #   Also a >= 0, s-a <= 128 => a >= s-128.
    #   So a ranges from max(0, s-128) to min(128, s//2).
    #   For each such a, if x[a] == 1 and x[s-a] == 1, that's one representation.
    #   Note: if a == s-a (i.e., s is even and a = s/2), it's still one representation (a,a).
    
    # So count[s] = sum_{a = max(0,s-128)}^{min(128, floor(s/2))} x[a] * x[s-a]
    
    # We want to minimize max_{s in [128,256]} count[s], subject to count[s] >= 1 for all s in [128,256].
    
    # This is a discrete optimization problem. We can use differential evolution on a relaxed
    # continuous problem, then round to binary. But DE works on continuous variables.
    # Alternatively, we can use a penalty-based approach: minimize max count + penalty * (number of uncovered sums).
    
    # Since the search space is 2^129, DE on 129 binary variables might be tricky.
    # Let's try DE on continuous [0,1]^129 and then threshold at 0.5.
    
    def objective(x):
        # x is a numpy array of length 129
        x = np.clip(x, 0, 1)
        # Compute counts for each s in [128, 256]
        max_count = 0
        uncovered = 0
        for s in range(target_low, target_high + 1):
            a_min = max(0, s - 128)
            a_max = min(128, s // 2)
            count = 0
            for a in range(a_min, a_max + 1):
                b = s - a
                if x[a] > 0.5 and x[b] > 0.5:
                    count += 1
            if count == 0:
                uncovered += 1
            else:
                if count > max_count:
                    max_count = count
        # Penalty for uncovered sums
        penalty = 100 * uncovered
        return max_count + penalty
    
    # Use differential evolution
    bounds = [(0, 1)] * 129
    
    # DE can be slow for 129 dimensions. Let's use a smaller population and fewer generations.
    # But we have 60 seconds. Let's try to be efficient.
    
    # Actually, 129 dimensions with DE might be too slow. Let's try a different approach:
    # Use a greedy/heuristic method to get a good initial solution, then refine.
    
    # Let's first try a simple heuristic: start with the seed solution A = [64, 128], which covers [128, 256].
    # For s in [128, 256], s = a + b where a, b in [64, 128].
    # a ranges from max(64, s-128) to min(128, s//2).
    # For s in [128, 192]: a from s-128 to s//2.
    # For s in [193, 256]: a from 64 to s//2.
    
    # Let's compute the max multiplicity for the seed solution.
    def compute_max_count(A_set):
        max_count = 0
        for s in range(target_low, target_high + 1):
            count = 0
            a_min = max(0, s - 128)
            a_max = min(128, s // 2)
            for a in range(a_min, a_max + 1):
                if a in A_set and (s - a) in A_set:
                    count += 1
            if count > max_count:
                max_count = count
        return max_count
    
    # Seed solution
    seed_A = set(range(64, 129))
    seed_max = compute_max_count(seed_A)
    
    # Let's try to improve with a local search: start from seed, try to remove elements and see if coverage is maintained.
    # Or add elements to reduce max count.
    
    # Actually, let's try a randomized local search with restarts.
    import random
    import time
    
    start_time = time.time()
    time_limit = 55  # seconds
    
    best_A = set(seed_A)
    best_max = seed_max
    
    def get_max_count_and_uncovered(A_set):
        max_count = 0
        uncovered = 0
        for s in range(target_low, target_high + 1):
            count = 0
            a_min = max(0, s - 128)
            a_max = min(128, s // 2)
            for a in range(a_min, a_max + 1):
                if a in A_set and (s - a) in A_set:
                    count += 1
            if count == 0:
                uncovered += 1
            else:
                if count > max_count:
                    max_count = count
        return max_count, uncovered
    
    # Local search: try to remove an element from A if it doesn't break coverage and doesn't increase max count.
    # Or try to swap: remove one element, add another.
    
    current_A = set(seed_A)
    current_max, current_uncovered = get_max_count_and_uncovered(current_A)
    
    # Try to remove elements
    for _ in range(1000):
        if time.time() - start_time > time_limit:
            break
        
        # Try removing a random element
        if len(current_A) <= 1:
            break
        elem = random.choice(list(current_A))
        new_A = current_A - {elem}
        new_max, new_uncovered = get_max_count_and_uncovered(new_A)
        
        if new_uncovered == 0 and new_max <= current_max:
            current_A = new_A
            current_max = new_max
            if current_max < best_max:
                best_max = current_max
                best_A = set(current_A)
        else:
            # Try swapping: remove elem, add a random element not in A
            candidates = [i for i in range(129) if i not in current_A]
            if candidates:
                add_elem = random.choice(candidates)
                new_A = current_A - {elem} | {add_elem}
                new_max, new_uncovered = get_max_count_and_uncovered(new_A)
                if new_uncovered == 0 and new_max <= current_max:
                    current_A = new_A
                    current_max = new_max
                    if current_max < best_max:
                        best_max = current_max
                        best_A = set(current_A)
    
    # Try to add elements to reduce max count
    for _ in range(1000):
        if time.time() - start_time > time_limit:
            break
        
        # Find the sum with max count
        max_s = None
        max_c = 0
        for s in range(target_low, target_high + 1):
            count = 0
            a_min = max(0, s - 128)
            a_max = min(128, s // 2)
            for a in range(a_min, a_max + 1):
                if a in current_A and (s - a) in current_A:
                    count += 1
            if count > max_c:
                max_c = count
                max_s = s
        
        if max_s is None:
            break
        
        # Try adding a random element
        candidates = [i for i in range(129) if i not in current_A]
        if not candidates:
            break
        add_elem = random.choice(candidates)
        new_A = current_A | {add_elem}
        new_max, new_uncovered = get_max_count_and_uncovered(new_A)
        
        if new_uncovered == 0 and new_max < current_max:
            current_A = new_A
            current_max = new_max
            if current_max < best_max:
                best_max = current_max
                best_A = set(current_A)
    
    return list(best_A)
# EVOLVE-BLOCK-END
