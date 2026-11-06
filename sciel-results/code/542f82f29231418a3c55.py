import itertools
import random
import time

def construct():
    """
    Returns a set of 12 distinct integers in [0, 128] minimizing the size of the largest Sidon subset.
    """
    start_time = time.time()
    timeout = 55.0  # Safety margin before 90s evaluator kill
    
    # Helper: Check if a set/list of integers is a Sidon set (all unordered pair sums unique)
    def is_sidon(S):
        sums = set()
        n = len(S)
        for i in range(n):
            for j in range(i, n):
                s = S[i] + S[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

    # Helper: Find the maximum size of a Sidon subset of S
    def max_sidon_size(S, deadline=None):
        # Check from largest possible size downwards
        for k in range(len(S), 0, -1):
            for subset in itertools.combinations(S, k):
                if deadline is not None and time.time() > deadline:
                    return k + 1  # Conservative: assume larger exists
                if is_sidon(subset):
                    return k
        return 0

    # Use a small range to force sum collisions
    # Range [0, 25] has 26 values, and sums go up to 50
    # This makes it harder to find large Sidon subsets
    SMALL_RANGE = (0, 25)
    
    # Initialize with a random set in the small range
    def random_small_set():
        vals = set()
        while len(vals) < 12:
            vals.add(random.randint(*SMALL_RANGE))
        return list(vals)
    
    # Try multiple initializations and pick the best
    # Start with an Arithmetic Progression, which has many additive collisions
    ap_set = list(range(12))  # 0..11
    best_set = ap_set[:]
    best_score = max_sidon_size(best_set)
    
    # Try a few more initializations (random and shifted APs) to get a good starting point
    for _ in range(20):
        candidate = random_small_set()
        score = max_sidon_size(candidate)
        if score < best_score:
            best_score = score
            best_set = candidate[:]
        
        # Also try shifted/scaled APs
        start = random.randint(0, 10)
        step = random.randint(1, 3)
        ap_candidate = [start + i * step for i in range(12)]
        if all(0 <= x <= SMALL_RANGE[1] for x in ap_candidate) and len(set(ap_candidate)) == 12:
             score = max_sidon_size(ap_candidate)
             if score < best_score:
                 best_score = score
                 best_set = ap_candidate[:]
                 
        if time.time() - start_time > timeout * 0.1:
            break
    
    # Local Search / Randomized Hill Climbing
    # Strategy: Perturb the current best set, staying in the small range
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() - start_time < timeout and no_improve_count < max_no_improve:
        # Generate a candidate
        candidate = best_set[:]
        
        # Randomly select 1-2 indices to mutate (smaller steps for fine-tuning)
        num_mutations = random.randint(1, 2)
        for _ in range(num_mutations):
            idx = random.randint(0, 11)
            # Pick a new value in the small range not in candidate
            while True:
                new_val = random.randint(*SMALL_RANGE)
                if new_val not in candidate:
                    candidate[idx] = new_val
                    break
        
        # Evaluate candidate
        score = max_sidon_size(candidate, deadline=start_time + timeout - 2)
        
        if score < best_score:
            best_score = score
            best_set = candidate[:]
            no_improve_count = 0
        else:
            no_improve_count += 1
    
    # Final polish: try to improve with single-element changes
    for i in range(12):
        if time.time() - start_time > timeout - 5:
            break
        # Iterate through all possible values in the small range
        for val in range(SMALL_RANGE[0], SMALL_RANGE[1] + 1):
            if val in best_set:
                continue
            candidate = best_set[:]
            candidate[i] = val
            score = max_sidon_size(candidate, deadline=start_time + timeout - 2)
            if score < best_score:
                best_score = score
                best_set = candidate[:]
                break
    
    return best_set
