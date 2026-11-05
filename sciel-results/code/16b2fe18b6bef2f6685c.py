"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import Counter
from itertools import combinations

def _count_duplicate_sums(subset):
    """Count how many unordered pair sums (including diagonals) are duplicated."""
    n = len(subset)
    sums = [subset[i] + subset[j] for i in range(n) for j in range(i, n)]
    c = Counter(sums)
    return sum(k * (k - 1) // 2 for k in c.values())

def _max_sidon_subset_size_bruteforce(subset, deadline=None):
    """Find largest subset with all unordered pair sums unique (including diagonals)."""
    n = len(subset)
    best = 0
    # Try all subsets in descending size order
    for size in range(n, best, -1):
        if deadline and time.time() > deadline:
            break
        for combo in combinations(range(n), size):
            if deadline and time.time() > deadline:
                break
            vals = [subset[i] for i in combo]
            s = len(vals)
            pair_sums = set()
            is_sidon = True
            for i in range(s):
                for j in range(i, s):
                    ps = vals[i] + vals[j]
                    if ps in pair_sums:
                        is_sidon = False
                        break
                    pair_sums.add(ps)
                if not is_sidon:
                    break
            if is_sidon:
                best = size
                break
        if best == size:
            break
    return best

def _local_search_exhaustive(subset, deadline, max_passes=10):
    """
    Exhaustive local search: for each element, try all possible values 0..128.
    Accept if duplicate sum count increases.
    """
    subset = list(subset)
    n = len(subset)
    current_dups = _count_duplicate_sums(subset)
    
    for _ in range(max_passes):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            old_val = subset[i]
            best_new_val = old_val
            best_new_dups = current_dups
            
            # Try all possible new values
            for new_val in range(129):
                if new_val == old_val:
                    continue
                # Check uniqueness
                # Since n=12, checking `new_val in subset` is fast
                # But we need to exclude index i
                if any(subset[j] == new_val for j in range(n) if j != i):
                    continue
                
                subset[i] = new_val
                new_dups = _count_duplicate_sums(subset)
                
                if new_dups > best_new_dups:
                    best_new_dups = new_dups
                    best_new_val = new_val
            
            if best_new_val != old_val:
                subset[i] = best_new_val
                current_dups = best_new_dups
                improved = True
        
        if not improved:
            break
            
    return subset

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s limit
    
    best_subset = list(range(12))
    best_score = _max_sidon_subset_size_bruteforce(best_subset, deadline)
    
    random.seed(42)
    
    # List of initial candidates: random + structured
    candidates = []
    
    # 1. Random candidates
    for _ in range(20):
        candidates.append(random.sample(range(129), 12))
        
    # 2. Structured candidates: Arithmetic Progressions
    for start in range(0, 100, 10):
        for d in [1, 2, 3, 4, 5]:
            ap = [start + i * d for i in range(12)]
            if all(0 <= x <= 128 for x in ap):
                candidates.append(ap)
                
    # 3. Structured candidates: Symmetric sets around a center
    # e.g., c-5, c-4, ..., c+5
    for c in range(64, 100, 10):
        sym = [c - 5 + i for i in range(12)] # This is just an AP with d=1
        # Try a more scattered symmetric set
        # c-10, c-8, c-6, c-4, c-2, c, c+2, c+4, c+6, c+8, c+10, c+12
        sym2 = [c-10, c-8, c-6, c-4, c-2, c, c+2, c+4, c+6, c+8, c+10, c+12]
        if all(0 <= x <= 128 for x in sym2):
            candidates.append(sym2)

    # Process candidates
    for candidate in candidates:
        if time.time() > deadline - 5:
            break
            
        # Local search to maximize collisions
        improved_candidate = _local_search_exhaustive(candidate, deadline, max_passes=5)
        
        # Evaluate
        score = _max_sidon_subset_size_bruteforce(improved_candidate, deadline)
        
        if score < best_score:
            best_score = score
            best_subset = improved_candidate
            if best_score <= 4: # Theoretical min is likely 4 or 5
                break
                
    # If we have time, do a final polish on the best subset
    if time.time() < deadline - 10 and best_score > 4:
        # Try swapping each element with every other value to see if score drops
        # This is expensive, so we limit it
        final_subset = list(best_subset)
        n = 12
        for i in range(n):
            if time.time() > deadline - 2:
                break
            old_val = final_subset[i]
            for new_val in range(129):
                if new_val == old_val:
                    continue
                if any(final_subset[j] == new_val for j in range(n) if j != i):
                    continue
                
                final_subset[i] = new_val
                # Quick check: does this reduce the max Sidon size?
                # We can just check if the duplicate count increased significantly
                # But we need the actual score.
                # To save time, only evaluate if duplicate count is high
                new_dups = _count_duplicate_sums(final_subset)
                if new_dups > _count_duplicate_sums(best_subset):
                    score = _max_sidon_subset_size_bruteforce(final_subset, deadline)
                    if score < best_score:
                        best_score = score
                        best_subset = final_subset
                final_subset[i] = old_val

    return sorted(best_subset)
# EVOLVE-BLOCK-END
