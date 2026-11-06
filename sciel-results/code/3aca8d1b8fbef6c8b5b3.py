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
    # Use backtracking with pruning to find the largest dissociated subset of [1,22]
    # A set is dissociated if all 2^k subset sums of reciprocals are distinct
    # Scale by LCM to use integer arithmetic
    
    all_elements = list(range(1, 23))
    n = len(all_elements)
    
    # Precompute: for a given set, compute the LCM-scaled weights
    # and check dissociation efficiently
    
    def check_dissociated_incremental(current_weights, new_weight):
        """Check if adding new_weight to current_weights keeps all subset sums distinct.
        current_weights: list of existing weights
        new_weight: the new weight to add
        Returns True if the combined set is dissociated.
        """
        if not current_weights:
            return True
        # All existing subset sums
        existing_sums = set()
        k = len(current_weights)
        for mask in range(1 << k):
            s = 0
            for i in range(k):
                if mask & (1 << i):
                    s += current_weights[i]
            existing_sums.add(s)
        
        # Check: for each existing subset sum s, s + new_weight must not be in existing_sums
        # (unless it's the same subset, but since we're adding a new element, 
        #  a collision means two different subsets have the same sum)
        # Actually, we need to check that no two distinct subsets of the new set have the same sum.
        # The new subsets are: old subsets, and old subsets + new element.
        # Collision occurs if:
        #   - Two old subsets have same sum (already checked)
        #   - An old subset sum equals an old subset sum + new_weight (impossible since new_weight > 0)
        #   - Two (old subset + new_weight) have same sum iff two old subsets have same sum
        #   - An old subset sum equals another old subset sum + new_weight
        # So we just need: for all s in existing_sums, s + new_weight not in existing_sums
        for s in existing_sums:
            if s + new_weight in existing_sums:
                return False
        return True
    
    def get_weights(S):
        """Get LCM-scaled weights for set S."""
        if not S:
            return []
        L = lcm_list(S)
        return [L // x for x in S]
    
    # Backtracking search
    best_set = None
    best_size = 0
    
    def backtrack(start_idx, current_set, current_weights, remaining_needed):
        nonlocal best_set, best_size
        
        # Update best
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = list(current_set)
        
        # Pruning: if we can't beat best_size even with all remaining elements
        remaining_elements = n - start_idx
        if len(current_set) + remaining_elements <= best_size:
            return
        
        # If we've found a set of the maximum possible size (limited by remaining), we're done
        # Try to add each element from start_idx onwards
        for i in range(start_idx, n):
            elem = all_elements[i]
            # Compute new LCM and weight
            if current_set:
                new_lcm = lcm(lcm_list(current_set), elem)
                # Recompute all weights with new LCM
                new_weights = [new_lcm // x for x in current_set] + [new_lcm // elem]
            else:
                new_lcm = elem
                new_weights = [1]  # LCM//elem = elem//elem = 1
            
            # Check if adding this element keeps dissociation
            # Use the incremental check
            if check_dissociated_incremental(current_weights, new_lcm // elem) if current_set else True:
                backtrack(i + 1, current_set + [elem], new_weights, remaining_needed - 1)
    
    # Start backtracking
    backtrack(0, [], [], n)
    
    # Also try a greedy approach as backup/improvement
    def greedy_add(order):
        S = []
        weights = []
        for elem in order:
            if not S:
                test_S = [elem]
                test_weights = [1]
            else:
                new_lcm = lcm(lcm_list(S), elem)
                test_weights = [new_lcm // x for x in S] + [new_lcm // elem]
                if check_dissociated_incremental([lcm_list(S) // x for x in S], new_lcm // elem):
                    test_S = S + [elem]
                else:
                    continue
            S = test_S
            weights = test_weights
        return S
    
    # Try multiple greedy orderings
    orderings = [
        list(range(1, 23)),
        list(range(22, 0, -1)),
        [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
        [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21],
        [1, 2, 4, 8, 16, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22],
    ]
    
    for order in orderings:
        S = greedy_add(order)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
    
    if best_set is None:
        return [1, 2, 3, 5, 7, 11, 13, 17, 19]
    
    return sorted(best_set)

# EVOLVE-BLOCK-END
