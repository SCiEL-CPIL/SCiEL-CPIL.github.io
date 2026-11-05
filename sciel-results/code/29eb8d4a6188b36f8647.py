"""Rewritten program for reciprocal dissociated set task.
Uses integer-scaled weights with a global LCM, and a fast backtracking search
with strong pruning via a precomputed collision-check using subset-sum tracking.
"""
# EVOLVE-BLOCK-START
import time
from math import gcd, lcm

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 60s hard limit

    N = 22
    # Compute LCM of 1..22
    L = 1
    for i in range(1, N + 1):
        L = lcm(L, i)
    
    # Precompute weights: w[i] = L // i for i in 1..22
    # w is a list indexed 0..21, where w[i] corresponds to element i+1
    w = [L // (i + 1) for i in range(N)]

    def is_dissociated(weights):
        """Check if all subset sums of the given integer weights are distinct.
        Uses a set and early exit on collision."""
        k = len(weights)
        if k == 0:
            return True
        sums = {0}
        for v in weights:
            new_sums = set()
            for s in sums:
                ns = s + v
                if ns in sums or ns in new_sums:
                    return False
                new_sums.add(ns)
            sums = sums | new_sums
        return True

    def is_dissociated_fast(weights):
        """Faster check using sorted weights and iterative list doubling.
        Sort weights in descending order to catch large collisions earlier? 
        Actually, sort ascending so that the number of sums grows predictably.
        We check len(set(sums)) == 2^k at the end, but use early check."""
        k = len(weights)
        if k == 0:
            return True
        # Sort to potentially improve cache behavior
        sorted_w = sorted(weights)
        sums = [0]
        for v in sorted_w:
            sums += [s + v for s in sums]
        return len(set(sums)) == (1 << k)

    def cheap_check(weights):
        """Very fast necessary condition check.
        For any two distinct weights a, b: a != b (trivially true since distinct elements).
        For any three distinct weights a, b, c: a != b + c, b != a + c, c != a + b.
        This is a necessary condition for dissociation."""
        k = len(weights)
        if k <= 2:
            return True
        # Sort for consistent checking
        sw = sorted(weights)
        for i in range(k):
            for j in range(i + 1, k):
                for m in range(j + 1, k):
                    a, b, c = sw[i], sw[j], sw[m]
                    # Check a == b + c (impossible if a < b+c, but check equality)
                    if a == b + c:
                        return False
                    if b == a + c:
                        return False
                    if c == a + b:
                        return False
        return True

    # Precompute for each element index, its weight
    # We'll search over subsets of {0, 1, ..., 21}
    
    # Strategy: try sizes from 22 down to 1
    # For each size k, do a backtracking search
    # To make the search faster, order elements by weight (largest weight first = smallest reciprocal)
    # Elements with larger weights (smaller i) are "heavier" in the subset-sum sense
    
    # Actually, let's think about this differently.
    # A collision occurs when sum_{i in A} w[i] == sum_{j in B} w[j] for distinct subsets A, B.
    # This is equivalent to: there exists a non-empty multiset with signed coefficients in {-1, 0, 1}
    # such that the weighted sum is 0.
    
    # For the search, we want to find the largest subset.
    # Order: try adding elements in an order that maximizes the chance of success.
    # Larger weights (smaller reciprocals) are less likely to cause collisions with each other
    # because their ratios are less likely to be small integers.
    # So sort by weight descending: w[21] (element 22) has the smallest weight, w[0] (element 1) has the largest.
    
    # Let's sort indices by weight descending
    indices_by_weight_desc = sorted(range(N), key=lambda i: w[i], reverse=True)
    
    # For backtracking, we'll build the set incrementally.
    # At each step, we add the next element and check if the current set is still dissociated.
    
    best_result = None
    
    for k in range(N, 0, -1):
        if time.time() > deadline:
            break
        
        # Try to find a dissociated subset of size k
        # Use backtracking with the elements ordered by weight descending
        # We need to choose k elements from N
        
        def backtrack(selected, start_idx, remaining):
            """selected: list of indices already chosen
            start_idx: minimum index in indices_by_weight_desc to consider next
            remaining: how many more to select
            Returns a list of selected indices if successful, else None."""
            if time.time() > deadline:
                return None
            
            if remaining == 0:
                # Check the full set
                weights = [w[i] for i in selected]
                if is_dissociated(weights):
                    return list(selected)
                return None
            
            if len(indices_by_weight_desc) - start_idx < remaining:
                return None
            
            for i in range(start_idx, len(indices_by_weight_desc)):
                idx = indices_by_weight_desc[i]
                new_selected = selected + [idx]
                # Quick check with the new element
                new_weights = [w[j] for j in new_selected]
                # Use the fast check
                if not is_dissociated(new_weights):
                    continue
                # Recurse
                result = backtrack(new_selected, i + 1, remaining - 1)
                if result is not None:
                    return result
            return None
        
        result = backtrack([], 0, k)
        if result is not None:
            # Convert indices back to values (1..22)
            values = sorted([idx + 1 for idx in result])
            return values
    
    # Fallback: known valid set
    return [1, 2, 3, 5, 7, 11, 13, 17, 19, 23] if 23 <= N else [1, 2, 3, 5, 7, 11, 13, 17, 19]

# EVOLVE-BLOCK-END
