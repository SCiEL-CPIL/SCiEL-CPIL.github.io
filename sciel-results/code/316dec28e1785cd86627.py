<<<<<<< SEARCH
"""Optimized program: search for maximum-cardinality reciprocal-dissociated subset of [1..22]."""
import time
import itertools
from fractions import Fraction

# Precompute all subsets for each cardinality, verify all 2^n subset sums (as Fractions) are distinct.
# n=22, max subsets per n is C(22,11)=705432. For each subset, checking 2^k sums.
# We'll do a BFS/DFS by cardinality from high to low, verifying candidates.

# To speed up verification: represent subset sums using integer arithmetic with a common denominator.
# The LCM of 1..22 is 232792560. Using this as denominator, each 1/i becomes LCM//i.
# Then subset sums are integers, and we just check for collisions in a set.

_LCM = 232792560

def _verify_recip_dissociated(subset):
    """Return True if all 2^len(subset) reciprocal subset sums are distinct."""
    k = len(subset)
    if k == 0:
        return True
    # Map each element to its scaled reciprocal
    scaled = [_LCM // x for x in subset]
    # Generate all subset sums via bitmasks
    n = 1 << k
    sums = [0] * n
    # Fast subset sum enumeration using DP-like approach
    for mask in range(1, n):
        # lowest set bit
        lb = mask & (-mask)
        idx = lb.bit_length() - 1
        prev = mask ^ lb
        sums[mask] = sums[prev] + scaled[idx]
    # Check for duplicates
    return len(set(sums)) == n

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Leave 10s safety margin
    
    # We want the maximum cardinality subset of [1..22] that is reciprocal-dissociated.
    # Try cardinalities from 22 down to 1.
    universe = list(range(1, 23))
    
    # For efficiency, we can use a greedy/heuristic approach first, then verify.
    # But let's try to be smarter: use the fact that larger numbers are "safer" for distinctness.
    
    # Start with a known good candidate and try to extend it.
    # Known: [1,2,3,5,7,11,13,17,19] has 9 elements.
    
    # Let's try to find larger sets by searching from high cardinality down.
    # For cardinality 12+, the number of combinations is too large to brute force all.
    # Instead, use a greedy approach: start with all 22, remove elements that cause collisions.
    
    # Greedy removal approach:
    # Start with full set, check if dissociated. If not, find a subset sum collision and remove one element involved.
    
    def find_collision(subset):
        """Find two different subsets with the same sum, or None if dissociated."""
        k = len(subset)
        if k == 0:
            return None
        scaled = [_LCM // x for x in subset]
        n = 1 << k
        # Use DP to compute all subset sums efficiently
        sums = [0] * n
        for mask in range(1, n):
            lb = mask & (-mask)
            idx = lb.bit_length() - 1
            prev = mask ^ lb
            sums[mask] = sums[prev] + scaled[idx]
        # Check for duplicates
        seen = {}
        for mask in range(n):
            s = sums[mask]
            if s in seen:
                return mask, seen[s]
            seen[s] = mask
        return None
    
    # Greedy: start with all 22, iteratively remove elements causing collisions
    current = list(range(1, 23))
    
    while True:
        if time.time() > deadline - 5:
            break
        col = find_collision(current)
        if col is None:
            break
        mask1, mask2 = col
        # Find elements in symmetric difference
        k = len(current)
        sym_diff = []
        for i in range(k):
            bit = 1 << i
            if (mask1 & bit) != (mask2 & bit):
                sym_diff.append(i)
        # Heuristic: remove the element with the largest value among those in sym_diff.
        to_remove = max(sym_diff, key=lambda i: current[i])
        current.remove(current[to_remove])
    
    best = list(current)
    
    # Now try to add back elements we removed
    removed = [x for x in range(1, 23) if x not in best]
    for x in removed:
        if time.time() > deadline - 5:
            break
        candidate = best + [x]
        candidate.sort()
        if find_collision(candidate) is None:
            best = candidate
    
    # If we still have time, try more aggressive search: swap elements
    if time.time() < deadline - 10 and len(best) < 22:
        # Try swapping each element in best with each not in best
        in_set = set(best)
        not_in_set = [x for x in range(1, 23) if x not in in_set]
        for i, a in enumerate(best):
            if time.time() > deadline - 5:
                break
            for b in not_in_set:
                candidate = best[:]
                candidate[i] = b
                candidate.sort()
                if find_collision(candidate) is None:
                    best = candidate
                    in_set = set(best)
                    not_in_set = [x for x in range(1, 23) if x not in in_set]
                    break
    
    # Final attempt: try to add remaining elements one more time
    in_set = set(best)
    not_in_set = [x for x in range(1, 23) if x not in in_set]
    for x in not_in_set:
        if time.time() > deadline - 3:
            break
        candidate = best + [x]
        candidate.sort()
        if find_collision(candidate) is None:
            best = candidate
    
    return best
=======
"""Optimized program: search for maximum-cardinality reciprocal-dissociated subset of [1..22]."""
import time
import random

_LCM = 232792560

def _subset_sums(scaled):
    """Compute all 2^k subset sums using DP over bitmasks."""
    k = len(scaled)
    n = 1 << k
    sums = [0] * n
    for mask in range(1, n):
        lb = mask & (-mask)
        idx = lb.bit_length() - 1
        prev = mask ^ lb
        sums[mask] = sums[prev] + scaled[idx]
    return sums

def _count_collisions(scaled):
    """Return number of distinct sums and whether all are distinct."""
    k = len(scaled)
    if k == 0:
        return 1, True
    sums = _subset_sums(scaled)
    n = 1 << k
    distinct = len(set(sums))
    return distinct, distinct == n

def _find_first_collision(subset):
    """Find two different masks with same subset sum, or None if dissociated."""
    k = len(subset)
    if k == 0:
        return None
    scaled = [_LCM // x for x in subset]
    sums = _subset_sums(scaled)
    seen = {}
    for mask in range(1 << k):
        s = sums[mask]
        if s in seen:
            return mask, seen[s]
        seen[s] = mask
    return None

def _sym_diff_indices(mask1, mask2, k):
    """Return indices where mask1 and mask2 differ."""
    return [i for i in range(k) if ((mask1 >> i) & 1) != ((mask2 >> i) & 1)]

def _greedy_remove(current, deadline):
    """Iteratively remove elements causing collisions until dissociated or time runs out."""
    current = list(current)
    while True:
        if time.time() > deadline - 5:
            break
        col = _find_first_collision(current)
        if col is None:
            break
        mask1, mask2 = col
        k = len(current)
        sym_diff = _sym_diff_indices(mask1, mask2, k)
        # Try each candidate removal and pick the one that minimizes collisions
        best_idx = None
        best_coll_count = float('inf')
        for idx in sym_diff:
            candidate = current[:idx] + current[idx+1:]
            scaled = [_LCM // x for x in candidate]
            distinct, ok = _count_collisions(scaled)
            if ok:
                best_idx = idx
                best_coll_count = 0
                break
            # Count collisions: n - distinct gives number of "extra" sums
            coll_count = (1 << len(candidate)) - distinct
            if coll_count < best_coll_count:
                best_coll_count = coll_count
                best_idx = idx
        if best_idx is None:
            # Fallback: remove largest element in sym_diff
            best_idx = max(sym_diff, key=lambda i: current[i])
        current.pop(best_idx)
    return current

def _try_add_back(best, deadline):
    """Try to add back removed elements one by one."""
    in_set = set(best)
    for x in range(1, 23):
        if time.time() > deadline - 3:
            break
        if x not in in_set:
            candidate = sorted(best + [x])
            if _find_first_collision(candidate) is None:
                best = candidate
                in_set.add(x)
    return best

def _try_swaps(best, deadline):
    """Try swapping elements to improve the set."""
    in_set = set(best)
    not_in_set = [x for x in range(1, 23) if x not in in_set]
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for i, a in enumerate(best):
            if time.time() > deadline - 5:
                break
            for b in not_in_set:
                candidate = best[:]
                candidate[i] = b
                candidate.sort()
                if _find_first_collision(candidate) is None:
                    best = candidate
                    in_set = set(best)
                    not_in_set = [x for x in range(1, 23) if x not in in_set]
                    improved = True
                    break
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Leave 10s safety margin
    universe = list(range(1, 23))
    
    best = []
    best_size = 0
    
    # Multiple restarts with different initial orderings
    rng = random.Random(42)
    
    for attempt in range(10):
        if time.time() > deadline - 10:
            break
        
        # Vary the starting set: sometimes start full, sometimes start with a random subset
        if attempt == 0:
            current = list(universe)
        elif attempt == 1:
            # Start with larger numbers first
            current = sorted(universe, reverse=True)
        elif attempt == 2:
            # Start with odd numbers plus some evens
            current = [x for x in universe if x % 2 == 1] + [x for x in universe if x % 2 == 0][:5]
            current = sorted(current)
        else:
            # Random permutation
            current = list(universe)
            rng.shuffle(current)
        
        # Greedy removal
        current = _greedy_remove(current, deadline)
        
        if len(current) > best_size:
            best_size = len(current)
            best = list(current)
        
        # Add back
        current = _try_add_back(current, deadline)
        if len(current) > best_size:
            best_size = len(current)
            best = list(current)
        
        # Swaps
        if time.time() < deadline - 10:
            current = _try_swaps(current, deadline)
            if len(current) > best_size:
                best_size = len(current)
                best = list(current)
        
        # Final add-back pass
        current = _try_add_back(current, deadline)
        if len(current) > best_size:
            best_size = len(current)
            best = list(current)
    
    # Final polish: try adding any remaining elements
    best = _try_add_back(best, deadline)
    
    return best
>>>>>>> REPLACE
