"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    """Construct a set of distinct integers in 1..256 such that subset sums
    from different cardinalities are disjoint. Maximize set size."""
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    # Try to find a large set using powers of 2 based construction
    # Powers of 2: 1, 2, 4, 8, ..., 2^k
    # For powers of 2, all subset sums are unique (no repeated sums at same cardinality)
    # and sums from different cardinalities are disjoint because:
    # - Min sum for cardinality k is 1+2+4+...+2^(k-1) = 2^k - 1
    # - Max sum for cardinality k-1 is 1+2+4+...+2^(k-2) = 2^(k-1) - 1
    # So max sum for card k-1 < min sum for card k
    
    # Powers of 2 up to 256: 1, 2, 4, 8, 16, 32, 64, 128 (8 elements)
    # But we can try larger sets.
    
    # Let's try a greedy approach: start with powers of 2, then try to add more elements
    base = [1, 2, 4, 8, 16, 32, 64, 128]
    
    # Check validity of base
    if not _is_valid(base):
        # Should never happen for powers of 2
        return base
    
    # Try to extend by adding elements from 3 to 256
    best_set = list(base)
    
    # Try adding elements in various orders
    candidates = list(range(3, 257))
    random.shuffle(candidates)
    
    for cand in candidates:
        if time.time() > deadline:
            break
        if cand in best_set:
            continue
        new_set = best_set + [cand]
        if _is_valid(new_set):
            best_set = new_set
    
    # Try multiple random restarts to find larger sets
    best_size = len(best_set)
    
    for _ in range(5):
        if time.time() > deadline:
            break
        
        # Start with a random subset of powers of 2 and try to extend
        current = []
        for p in base:
            if random.random() < 0.8:
                current.append(p)
        
        if not current:
            current = [1]
        
        if not _is_valid(current):
            continue
        
        # Try to add elements
        candidates2 = list(range(1, 257))
        random.shuffle(candidates2)
        
        for cand in candidates2:
            if time.time() > deadline:
                break
            if cand in current:
                continue
            new_set = current + [cand]
            if _is_valid(new_set):
                current = new_set
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = current
    
    return best_set

def _is_valid(s):
    """Check if set s satisfies the cross-cardinality subset sum constraint."""
    s = sorted(set(s))
    k = len(s)
    
    # For small sets, do exact check
    if k <= 20:
        sums_by_card = {}
        for c in range(k + 1):
            for comb in combinations(s, c):
                sm = sum(comb)
                sums_by_card.setdefault(c, set()).add(sm)
        
        # Check cross-cardinality disjointness
        all_sums = {}
        for c, sums in sums_by_card.items():
            for sm in sums:
                if sm in all_sums:
                    return False
                all_sums[sm] = c
        return True
    else:
        # For larger sets, use a heuristic check based on range analysis
        # This is approximate but fast
        for c in range(1, k + 1):
            # Min sum for cardinality c
            min_sum_c = sum(sorted(s)[:c])
            # Max sum for cardinality c
            max_sum_c = sum(sorted(s)[-c:])
            
            # For each other cardinality d, check if ranges overlap
            for d in range(c + 1, k + 1):
                min_sum_d = sum(sorted(s)[:d])
                # If max_sum_c >= min_sum_d, potential overlap
                # But this is a necessary condition, not sufficient
                if max_sum_c >= min_sum_d:
                    # Do a more detailed check for this pair
                    if not _check_pair_disjoint(s, c, d):
                        return False
        return True

def _check_pair_disjoint(s, c, d):
    """Check if subset sums of cardinality c and d are disjoint."""
    s = sorted(s)
    k = len(s)
    
    # Get all sums of cardinality c
    sums_c = set()
    for comb in combinations(s, c):
        sums_c.add(sum(comb))
    
    # Check if any sum of cardinality d is in sums_c
    for comb in combinations(s, d):
        if sum(comb) in sums_c:
            return False
    
    return True
# EVOLVE-BLOCK-END
