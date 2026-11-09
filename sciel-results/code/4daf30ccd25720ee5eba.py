import random
import time

def is_sidon(subset):
    # Check if all unordered pair sums (including diagonals) are unique
    sums = set()
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def largest_sidon_subset_size(candidate):
    # Find the size of the largest Sidon subset of candidate
    # candidate is a list of 12 distinct integers
    n = len(candidate)
    # Check subsets of size k from n down to 1
    # We can use itertools.combinations, but 2^12 is small enough for brute force over masks
    # Or just combinations
    from itertools import combinations
    
    # Sort candidate for consistency, though not strictly needed
    # for k in range(n, 0, -1):
    #     for comb in combinations(candidate, k):
    #         if is_sidon(comb):
    #             return k
    #     # If no Sidon subset of size k, try k-1
    # return 1
    
    # More efficient: check all subsets
    # But combinations is fine.
    # C(12, 6) = 924. C(12, 7) = 792. Sum is 4096.
    # is_sidon is O(k^2).
    # Total ops ~ 4096 * 144 ~ 600k.
    
    best = 1
    # We can prune: if we found a Sidon subset of size k, we don't need to check smaller?
    # No, we want the LARGEST. So we check from large to small.
    for k in range(n, 0, -1):
        found = False
        for comb in combinations(candidate, k):
            if is_sidon(comb):
                found = True
                break
        if found:
            return k
    return 1

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin
    
    # Initial candidate: random
    # Or start with something structured?
    # Random is good for diversity.
    current = sorted(random.sample(range(129), 12))
    current_score = largest_sidon_subset_size(current)
    
    best_candidate = current[:]
    best_score = current_score
    
    # Local search
    # We can also try to bias towards sets with many collisions.
    # But random perturbation is simple.
    
    while time.time() - start_time < time_limit:
        # Pick a random index to change
        idx = random.randint(0, 11)
        old_val = current[idx]
        
        # Pick a new value
        # To ensure distinctness, we can pick from [0, 128] excluding current values
        # Or just pick random and check
        new_val = random.randint(0, 128)
        
        if new_val in current:
            continue
            
        # Create new candidate
        new_candidate = current[:]
        new_candidate[idx] = new_val
        new_candidate.sort() # Keep sorted for consistency
        
        new_score = largest_sidon_subset_size(new_candidate)
        
        # Acceptance criteria
        if new_score < best_score:
            best_score = new_score
            best_candidate = new_candidate[:]
            current = new_candidate
        elif new_score <= current_score:
            # Hill climbing: accept if equal or better
            # This might get stuck in local minima
            # Maybe add some noise?
            current = new_candidate
        else:
            # Rejection
            pass

    return best_candidate
