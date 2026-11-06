"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
from math import gcd
from functools import reduce
from itertools import combinations

def lcm(a, b):
    return a * b // gcd(a, b)

def lcm_list(lst):
    return reduce(lcm, lst, 1)

def is_dissociated(S):
    """Check if all subset sums of reciprocals of S are distinct, using integer scaling."""
    if not S:
        return True
    L = lcm_list(S)
    # Scaled weights
    weights = [L // n for n in S]
    k = len(weights)
    # Generate all 2^k subset sums
    subset_sums = set()
    for mask in range(1 << k):
        s = 0
        for i in range(k):
            if mask & (1 << i):
                s += weights[i]
        if s in subset_sums:
            return False
        subset_sums.add(s)
    return True

def construct():
    # Try to find the largest dissociated subset of [1,22]
    # Use greedy approach: try adding elements in various orders
    
    # First, let's try a systematic search
    # We want to maximize cardinality, so try from largest possible size down
    
    # Precompute: which subsets of size k are dissociated?
    # For efficiency, use greedy construction
    
    # Strategy: try to build a large set by adding elements one at a time
    # Try different starting points and orders
    
    best_set = None
    
    # Try greedy: start with each possible first element, then add others
    # To maximize cardinality, prefer adding smaller numbers (they have larger reciprocals, 
    # but also the LCM grows faster... actually let's just try)
    
    # Let's try a backtracking approach for small sizes
    # First, determine the maximum possible size by trying all subsets of size k
    
    # For k=9, the current answer works. Let's see if k=10 is possible.
    
    # Try all subsets of size 10 from [1,22] - that's C(22,10) = 646646, might be slow
    # But with early termination in is_dissociated, it could work.
    
    # Actually, let's be smarter. Use a greedy heuristic first.
    
    # Greedy: try to add elements in order of increasing value
    # But also try other orders
    
    # Let's try: start with empty set, repeatedly add the element that keeps it dissociated
    # Try multiple strategies
    
    candidates = list(range(1, 23))
    
    # Try greedy with different orderings
    best_cardinality = 0
    best_result = None
    
    # Strategy 1: greedy adding smallest first
    def greedy_add(order):
        S = []
        for n in order:
            test_S = S + [n]
            if is_dissociated(test_S):
                S = test_S
        return S
    
    # Try several orderings
    orderings = [
        list(range(1, 23)),  # increasing
        list(range(22, 0, -1)),  # decreasing
        [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],  # odds first
        [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21],  # evens first
    ]
    
    for order in orderings:
        S = greedy_add(order)
        if len(S) > best_cardinality:
            best_cardinality = len(S)
            best_result = S
    
    # Now try to improve: for the best cardinality found, try local search
    # Also try to find if we can get cardinality best_cardinality + 1
    
    target_size = best_cardinality + 1
    
    # If target_size is reasonable, try to find a subset of that size
    if target_size <= 12:  # 2^12 = 4096 subset sums, manageable
        # Try random subsets of size target_size
        import random
        random.seed(42)
        found = False
        for _ in range(5000):
            S = random.sample(range(1, 23), target_size)
            if is_dissociated(S):
                found = True
                if len(S) > best_cardinality:
                    best_cardinality = len(S)
                    best_result = S
                break
        
        if not found:
            # Try more systematic: try all subsets of size target_size if feasible
            if target_size <= 10:
                for S in combinations(range(1, 23), target_size):
                    S = list(S)
                    if is_dissociated(S):
                        if len(S) > best_cardinality:
                            best_cardinality = len(S)
                            best_result = S
                        break
    
    # Return the best set found
    if best_result is None:
        # Fallback to known good set
        return [1, 2, 3, 5, 7, 11, 13, 17, 19]
    
    return sorted(best_result)

# EVOLVE-BLOCK-END
