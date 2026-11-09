import time
import random
from itertools import combinations

def is_sidon(subset):
    sums = set()
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def largest_sidon_subset_size(S):
    S = sorted(S)
    n = len(S)
    # Try from largest k down
    for k in range(n, 0, -1):
        for combo in combinations(S, k):
            if is_sidon(combo):
                return k
    return 1

def count_sum_collisions(S):
    """Count how many pair sums are duplicated (higher = more collisions = harder for Sidon)"""
    sums = {}
    n = len(S)
    for i in range(n):
        for j in range(i, n):
            s = S[i] + S[j]
            if s in sums:
                sums[s] += 1
            else:
                sums[s] = 1
    # Total collisions = sum of (count-1) for each sum that appears more than once
    collisions = sum(v - 1 for v in sums.values() if v > 1)
    return collisions

def count_sidon_subsets_of_size(S, k):
    """Count the number of subsets of size k that are Sidon."""
    S = sorted(S)
    count = 0
    for combo in combinations(S, k):
        if is_sidon(combo):
            count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    best_set = None
    best_score = float('inf') # We want to minimize the count of 4-Sidon sets
    
    # Initial candidates
    candidates = []
    # 1. Standard AP
    candidates.append(list(range(12)))
    # 2. AP with step 2
    candidates.append([2*i for i in range(12)])
    # 3. Random dense clusters
    for _ in range(50):
        start = random.randint(0, 110)
        # Try to form a set with many small sums
        # Use a random permutation of small offsets
        offsets = random.sample(range(20), 12)
        s = set(start + o for o in offsets)
        if len(s) == 12 and max(s) <= 128:
            candidates.append(sorted(s))
            
    # Evaluate candidates
    for c in candidates:
        if len(c) != 12:
            continue
        if any(x < 0 or x > 128 for x in c):
            continue
        # Primary objective: Minimize count of Sidon subsets of size 4
        # If count is 0, we have a potential value 3 set.
        # Secondary objective: If count > 0, minimize count of size 5? No, that doesn't help if 4 exists.
        # If count_4 > 0, the value is at least 4.
        # To improve on value 4, we MUST get count_4 == 0.
        # So we focus entirely on minimizing count_4.
        
        c4 = count_sidon_subsets_of_size(c, 4)
        if c4 < best_score:
            best_score = c4
            best_set = c[:]
            
    # Local Search
    # We want to drive count_4 to 0.
    if best_set is not None:
        current_set = best_set[:]
        current_c4 = best_score
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Shuffle indices for randomization
            indices = list(range(12))
            random.shuffle(indices)
            for i in indices:
                if time.time() > deadline - 5:
                    break
                old_val = current_set[i]
                # Try swapping with values in a wider range
                # Dense sets are better, so try values close to current set's range
                min_val = min(current_set)
                max_val = max(current_set)
                search_range = range(max(0, min_val - 10), min(129, max_val + 10))
                
                for new_val in search_range:
                    if new_val == old_val or new_val in current_set:
                        continue
                    trial = current_set[:]
                    trial[i] = new_val
                    trial.sort()
                    trial_c4 = count_sidon_subsets_of_size(trial, 4)
                    if trial_c4 < current_c4:
                        current_c4 = trial_c4
                        current_set = trial
                        improved = True
                        break
                    # Allow equal moves with some probability to escape local minima?
                    # Or just strict improvement.
                if improved:
                    break
                    
        best_set = current_set
        best_score = current_c4

    # Fallback
    if best_set is None:
        best_set = list(range(12))
        
    return best_set
