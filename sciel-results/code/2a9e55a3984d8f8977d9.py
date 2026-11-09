"""Optimized program: search for maximum-cardinality reciprocal-dissociated subset of [1..22]."""
import time
import random
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

def _find_collision(subset):
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

def _greedy_remove_initial(current, deadline):
    """Greedy removal approach starting from a given set."""
    while True:
        if time.time() > deadline - 5:
            break
        col = _find_collision(current)
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
    return list(current)

def _random_greedy_add(deadline):
    """Randomized greedy addition approach."""
    universe = list(range(1, 23))
    random.shuffle(universe)
    current = []
    for x in universe:
        if time.time() > deadline - 5:
            break
        candidate = current + [x]
        if _verify_recip_dissociated(candidate):
            current = candidate
    return current

def _polish_set(best, deadline):
    """Attempt to improve the set by adding elements or swapping."""
    best = list(best)
    in_set = set(best)
    not_in_set = [x for x in range(1, 23) if x not in in_set]
    
    # Phase 1: Add any possible element
    improved = True
    while improved:
        improved = False
        for x in not_in_set[:]:
            if time.time() > deadline - 5:
                return best
            candidate = best + [x]
            if _verify_recip_dissociated(candidate):
                best = candidate
                in_set = set(best)
                not_in_set = [x for x in range(1, 23) if x not in in_set]
                improved = True
                break # Restart loop with new state
    
    # Phase 2: Swap and Add
    # Try swapping i in best with j in not_in_set
    # If swap is valid, check if we can add another element.
    # This is O(|best| * |not_in_set| * Verify).
    # |best| ~ 16, |not_in_set| ~ 6. 96 checks. Each check is 2^16 ~ 65k ops.
    # 96 * 65k ~ 6M ops. Should be fast enough in Python if optimized?
    # Python might be slow. Let's limit this.
    
    if time.time() > deadline - 10:
        return best
        
    found_swap = True
    while found_swap:
        found_swap = False
        if time.time() > deadline - 5:
            return best
        in_set = set(best)
        not_in_set = [x for x in range(1, 23) if x not in in_set]
        
        for i_idx, i_val in enumerate(best):
            if time.time() > deadline - 5:
                return best
            for j_val in not_in_set:
                # Try swap
                candidate_swap = best[:]
                candidate_swap[i_idx] = j_val
                # Check if swap itself is valid
                if not _verify_recip_dissociated(candidate_swap):
                    continue
                
                # If swap is valid, try to add any element from (not_in_set - {j_val})
                # Note: j_val is now in set, so we look for k in original not_in_set excluding j_val
                # Also i_val is now out, so we could potentially add i_val back? 
                # If we add i_val back, we are back to original set (plus maybe others).
                # We want to increase cardinality. So we look for k != j_val.
                # Actually, if we swap i->j, size is same. 
                # Then we try to add k. Size becomes +1.
                
                potential_adds = [x for x in not_in_set if x != j_val]
                # Also check if i_val can be added? 
                # If we add i_val, we get {j, i, ...rest}. 
                # This is effectively a 2-swap? No, it's swap + add.
                
                for k_val in potential_adds:
                    candidate_add = candidate_swap + [k_val]
                    if _verify_recip_dissociated(candidate_add):
                        # Accept swap and add
                        best = candidate_add
                        found_swap = True
                        break
                if found_swap:
                    break
            if found_swap:
                break
                
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Leave 10s safety margin
    
    universe = list(range(1, 23))
    
    # Strategy 1: Greedy Remove from Full
    best = _greedy_remove_initial(list(universe), deadline)
    
    # Strategy 2: Random Greedy Add (Multiple Restarts)
    # Run several times to find a better local optimum
    num_restarts = 20
    for i in range(num_restarts):
        if time.time() > deadline - 10:
            break
        candidate = _random_greedy_add(deadline)
        if len(candidate) > len(best):
            best = candidate
    
    # Strategy 3: Polish the best result
    best = _polish_set(best, deadline)
    
    # Final safety check
    if not _verify_recip_dissociated(best):
        # Fallback to a known small set if something went wrong
        return [1, 2, 3, 5, 7, 11, 13, 17, 19]
        
    return best
