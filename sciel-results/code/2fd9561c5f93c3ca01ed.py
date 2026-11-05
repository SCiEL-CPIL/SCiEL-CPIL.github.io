"""Rewritten program: integer-scaled backtracking with strong pruning to maximize
dissociated reciprocal set cardinality within [1,22]."""
import time
from math import gcd
from functools import lru_cache

# Deadline guard: finish well before 90s wall-clock
_START = time.time()
_DEADLINE = 80.0  # safety margin

def _time_ok():
    return (time.time() - _START) < _DEADLINE


def construct():
    """Return a nonempty subset of [1,22] with all reciprocal subset sums distinct,
    maximizing cardinality."""
    # We work with reciprocals 1/k for k in 1..22.
    # A set S is dissociated iff all subset sums of {1/k : k in S} are distinct.
    # Equivalently, no non-trivial integer linear combination of {1/k : k in S}
    # equals 0 with coefficients in {-1,0,1}.
    #
    # Multiply by LCM(1..22) to get integer weights.
    # LCM(1..22) = 2^4 * 3^2 * 5 * 7 * 11 * 13 * 17 * 19 = 232792560
    # weight[k] = LCM // k
    # Then subset sums of weights are distinct iff reciprocal subset sums are distinct.
    
    LCM = 232792560
    # Weights for k=1..22
    weights = [LCM // k for k in range(1, 23)]
    # weights[i] corresponds to element (i+1)
    
    # We want to find a subset of indices {0..21} of maximum size such that
    # all 2^|S| subset sums of the corresponding weights are distinct.
    #
    # Strategy: backtracking search trying to add elements, checking dissociation
    # incrementally. Use the fact that if we have a set A and add element w,
    # the new subset sums are old_sums ∪ (old_sums + w). They are all distinct
    # iff (old_sums + w) ∩ old_sums = ∅.
    #
    # For efficiency, we represent the set of subset sums as a set of integers.
    # But 2^22 is too large. Instead, we use a smarter approach:
    #
    # Key insight: A set of integers is dissociated iff for every subset, the sum
    # is unique. A necessary and sufficient condition can be checked incrementally:
    # when adding w to a dissociated set with subset sums S, we need S ∩ (S+w) = ∅.
    #
    # For small sets this is fine. Let's try greedy + local search, then verify.
    
    # First, try to find a large dissociated set via greedy with backtracking.
    # Order elements by weight (largest first tends to be safer for dissociation
    # since large gaps reduce collisions).
    
    # Actually, let's think differently. The condition is that no non-trivial
    # {-1,0,1}-combination equals 0. For a set of size n, there are 3^n - 1 non-trivial
    # combinations. For n=11, that's ~177k which is checkable. For n=12, ~531k.
    # For n=13, ~1.6M. Still feasible with early termination.
    
    # Better approach: Use the subset-sum set representation for the actual search.
    # Start with empty set, greedily add elements that maintain dissociation.
    # Try multiple orderings and use backtracking to find max size.
    
    def is_dissociated_incremental(subset_sums, w):
        """Check if adding weight w to a set with subset_sums maintains dissociation.
        Returns new subset_sums if valid, None otherwise."""
        new_sums = set()
        for s in subset_sums:
            ns = s + w
            if ns in subset_sums or ns in new_sums:
                return None
            new_sums.add(ns)
        for s in subset_sums:
            new_sums.add(s)
        return new_sums
    
    # Greedy approach: try elements in various orders
    # For a given order, greedily add if possible. Then try to improve.
    
    best_size = 0
    best_set = None
    
    # Try greedy with different orderings
    indices = list(range(22))  # 0..21, corresponding to k=1..22
    
    # Heuristic: sort by weight descending (larger weights first)
    # This tends to create more "spread out" subset sums
    order_desc = sorted(indices, key=lambda i: weights[i], reverse=True)
    order_asc = sorted(indices, key=lambda i: weights[i])
    # Also try: prioritize larger k (smaller reciprocal) first
    order_k_desc = sorted(indices, key=lambda i: i+1, reverse=True)
    order_k_asc = sorted(indices, key=lambda i: i+1)
    
    # Random-ish orders via different seeds
    import random
    rng = random.Random(42)
    random_orders = []
    for seed in range(20):
        r = random.Random(seed)
        idx = list(range(22))
        r.shuffle(idx)
        random_orders.append(idx)
    
    all_orders = [order_desc, order_asc, order_k_desc, order_k_asc] + random_orders
    
    for order in all_orders:
        if not _time_ok():
            break
        subset_sums = {0}
        chosen = []
        for i in order:
            if not _time_ok():
                break
            new_sums = is_dissociated_incremental(subset_sums, weights[i])
            if new_sums is not None:
                subset_sums = new_sums
                chosen.append(i)
        
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_set = chosen[:]
    
    # Now try to improve: for each element not in best_set, try swapping
    # Also try backtracking to fill remaining slots
    
    # Backtracking search to find maximum size
    # We'll do a DFS that tries to build a set of size > best_size
    
    def backtrack(current_sums, chosen, remaining, target_size):
        """Try to build a set of size target_size starting from current state."""
        if not _time_ok():
            return None
        if len(chosen) == target_size:
            return chosen[:]
        
        # Pruning: not enough remaining elements
        if len(chosen) + len(remaining) < target_size:
            return None
        
        for j, i in enumerate(remaining):
            if not _time_ok():
                return None
            new_sums = is_dissociated_incremental(current_sums, weights[i])
            if new_sums is not None:
                result = backtrack(new_sums, chosen + [i], remaining[j+1:], target_size)
                if result is not None:
                    return result
        return None
    
    # Try to find a set of size best_size + 1
    for target in range(best_size + 1, 14):  # up to 13 elements
        if not _time_ok():
            break
        # Try different starting orders for the backtracking
        for order in [order_desc, order_k_desc, order_asc]:
            if not _time_ok():
                break
            result = backtrack({0}, [], order, target)
            if result is not None:
                best_size = target
                best_set = result
                break
    
    # Final verification
    def verify_dissociated(selected_indices):
        """Verify all subset sums are distinct."""
        w = [weights[i] for i in selected_indices]
        n = len(w)
        sums = set()
        for mask in range(1 << n):
            s = 0
            for b in range(n):
                if mask & (1 << b):
                    s += w[b]
            if s in sums:
                return False
            sums.add(s)
        return True
    
    if best_set and not verify_dissociated(best_set):
        # Fallback to a known valid set
        best_set = [0, 1, 2, 4, 6, 10, 12, 16, 18]  # k=1,2,3,5,7,11,13,17,19
    
    # Convert indices back to values (k = index + 1)
    result = sorted([i + 1 for i in best_set])
    
    # Ensure nonempty
    if not result:
        result = [1]
    
    return result
