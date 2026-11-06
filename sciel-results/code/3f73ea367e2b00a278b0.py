"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # Domain
    V_small = list(range(2, 129))
    V_large = list(range(129, 257))
    
    # Precompute reciprocal sums
    base_sum_large = sum(1.0 / b for b in V_large)
    
    # Precompute conflicts for V_large
    # conf_large[a] = set of b in V_large such that lcm(a, b) <= 256
    conf_large = {}
    for a in V_small:
        confs = set()
        for b in V_large:
            if (a * b) // gcd(a, b) <= 256:
                confs.add(b)
        conf_large[a] = confs
        
    # Precompute conflicts within V_small
    # conf_small[a] = set of c in V_small such that lcm(a, c) <= 256
    conf_small = {}
    for a in V_small:
        confs = set()
        for c in V_small:
            if a == c:
                continue
            if (a * c) // gcd(a, c) <= 256:
                confs.add(c)
        conf_small[a] = confs

    # Structural Change:
    # 1. Core: [2, 16] - High constraint, high value. Exhaustive search.
    # 2. Mid: [17, 128] - Greedy fill based on Core selection.
    # 3. Large: [129, 256] - All compatible with Core+Mid are included.
    #    Note: Large numbers are likely mutually compatible.
    
    core_numbers = list(range(2, 17))
    mid_numbers = list(range(17, 129))
    
    # Sort mid by reciprocal descending for greedy
    mid_sorted = sorted(mid_numbers, key=lambda x: -1.0/x)
    
    best_score = 0.0
    best_small_set = set()
    
    # Helper to evaluate a set of small numbers
    def evaluate_small_set(A):
        # A is a set of numbers from V_small
        # Calculate lost large numbers
        lost_large = set()
        for a in A:
            lost_large.update(conf_large[a])
        
        sum_a = sum(1.0 / a for a in A)
        sum_b = base_sum_large - sum(1.0 / b for b in lost_large)
        return sum_a + sum_b

    # Recursive search over core_numbers
    # Use bitmask or index. 15 items -> 2^15 = 32768 subsets. Very fast.
    # We will iterate over all subsets of core_numbers.
    
    def search_core(idx, current_A, current_conflicts):
        nonlocal best_score, best_small_set
        
        if time.time() > deadline:
            return
            
        if idx == len(core_numbers):
            # Extend with mid_numbers greedily
            extended_A = set(current_A)
            extended_conflicts = set(current_conflicts)
            
            # Greedy add mid numbers
            for c in mid_sorted:
                if c not in extended_conflicts:
                    extended_A.add(c)
                    extended_conflicts |= conf_small[c]
                    extended_conflicts.add(c)
            
            score = evaluate_small_set(extended_A)
            if score > best_score:
                best_score = score
                best_small_set = set(extended_A)
            return
        
        a = core_numbers[idx]
        
        # Option 1: Skip a
        search_core(idx + 1, current_A, current_conflicts)
        
        # Option 2: Add a if no conflict
        if a not in current_conflicts:
            new_A = current_A | {a}
            new_conflicts = current_conflicts | conf_small[a] | {a}
            search_core(idx + 1, new_A, new_conflicts)

    search_core(0, set(), set())
    
    # Fallback
    if not best_small_set:
        # Simple greedy on all small
        greedy_A = set()
        greedy_conflicts = set()
        for a in sorted(V_small, key=lambda x: -1.0/x):
            if a not in greedy_conflicts:
                greedy_A.add(a)
                greedy_conflicts |= conf_small[a]
                greedy_conflicts.add(a)
        
        score = evaluate_small_set(greedy_A)
        if score > best_score:
            best_score = score
            best_small_set = greedy_A

    if not best_small_set:
        return [2]
        
    # Construct final set
    final_set = set(best_small_set)
    lost_large = set()
    for a in final_set:
        lost_large.update(conf_large[a])
        
    for b in V_large:
        if b not in lost_large:
            final_set.add(b)
            
    return list(final_set)
# EVOLVE-BLOCK-END
