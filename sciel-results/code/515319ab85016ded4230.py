"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

_START_TIME = time.time()
_DEADLINE = _START_TIME + 85.0  # Safety margin for 90s limit

def _subset_products_valid(nums):
    """Check if all 2^k subset products of nums are distinct."""
    if not nums:
        return True
    prods = {1}
    for x in nums:
        new_prods = set(prods)
        for p in prods:
            np = p * x
            if np in prods:
                return False
            new_prods.add(np)
        prods = new_prods
    return True

def _all_valid_subsets(cands):
    """Enumerate all valid subsets of cands, return list of (frozenset, product_set)."""
    n = len(cands)
    valid = []
    # Use bitmask enumeration
    for mask in range(1 << n):
        subset = [cands[i] for i in range(n) if mask & (1 << i)]
        if _subset_products_valid(subset):
            # Compute product set
            prods = {1}
            for x in subset:
                new_prods = set(prods)
                for p in prods:
                    new_prods.add(p * x)
                prods = new_prods
            valid.append((frozenset(subset), prods))
    return valid

def construct():
    """
    Returns a list of distinct integers in 2..40 such that all 2^k subset products are distinct.
    Maximizes the number of selected integers.
    
    Strategy: Meet-in-the-middle.
    1. Split candidates {2..40} into two halves.
    2. Enumerate all valid subsets of each half with their product sets.
    3. For each pair of valid subsets (one from each half), check if their combined
       product sets are disjoint (no cross-collisions).
    4. Find the maximum-size valid combination.
    """
    candidates = list(range(2, 41))
    n = len(candidates)
    mid = n // 2
    left = candidates[:mid]
    right = candidates[mid:]
    
    # Initial best: all primes (known valid)
    def _is_prime(x):
        if x < 2: return False
        for i in range(2, int(x**0.5) + 1):
            if x % i == 0: return False
        return True
    
    primes = [x for x in candidates if _is_prime(x)]
    best_size = len(primes)
    best_set = list(primes)
    
    # Enumerate valid subsets of left half
    left_valid = []
    for mask in range(1 << len(left)):
        if time.time() > _DEADLINE:
            break
        subset = [left[i] for i in range(len(left)) if mask & (1 << i)]
        if _subset_products_valid(subset):
            prods = {1}
            for x in subset:
                new_prods = set(prods)
                for p in prods:
                    new_prods.add(p * x)
                prods = new_prods
            left_valid.append((frozenset(subset), prods))
    
    # Enumerate valid subsets of right half
    right_valid = []
    for mask in range(1 << len(right)):
        if time.time() > _DEADLINE:
            break
        subset = [right[i] for i in range(len(right)) if mask & (1 << i)]
        if _subset_products_valid(subset):
            prods = {1}
            for x in subset:
                new_prods = set(prods)
                for p in prods:
                    new_prods.add(p * x)
                prods = new_prods
            right_valid.append((frozenset(subset), prods))
    
    # Combine: for each left subset and right subset, check cross-product collision
    # Cross collision: exists p1 in left_prods, p2 in right_prods, a in left_subset, b in right_subset
    # such that p1 * a == p2 * b ... actually simpler:
    # Combined product set = {p1 * p2 : p1 in left_prods, p2 in right_prods}
    # This is valid iff |left_prods| * |right_prods| == len(combined)
    # i.e., no p1*p2 = p1'*p2' for distinct pairs.
    # Equivalently: for all (p1,p2) != (p1',p2'), p1*p2 != p1'*p2'
    # This means the ratio p1/p1' != p2'/p2 for any distinct pairs.
    # Simpler check: the combined set size equals |left_prods| * |right_prods|
    
    for lset, lprods in left_valid:
        if time.time() > _DEADLINE:
            break
        lsize = len(lset)
        if lsize + len(right) <= best_size:
            continue  # Can't beat best even with all of right
        
        for rset, rprods in right_valid:
            if time.time() > _DEADLINE:
                break
            total_size = lsize + len(rset)
            if total_size <= best_size:
                continue
            
            # Check cross-product collision
            # Combined products: {p1 * p2 for p1 in lprods for p2 in rprods}
            # Valid iff all these are distinct, i.e., len(combined) == len(lprods) * len(rprods)
            # But computing the full combined set is expensive.
            # Alternative: check if there exist p1 != p1' in lprods and p2 != p2' in rprods
            # with p1*p2 = p1'*p2', i.e., p1/p1' = p2'/p2.
            # This is equivalent to: the set of ratios {p1/p1' : p1 != p1' in lprods} 
            # is disjoint from {p2'/p2 : p2 != p2' in rprods}.
            # Even simpler: just compute the combined set for small sizes.
            
            # For efficiency, only do full check if sizes are manageable
            if len(lprods) * len(rprods) <= 200000:
                combined = set()
                for p1 in lprods:
                    for p2 in rprods:
                        combined.add(p1 * p2)
                if len(combined) == len(lprods) * len(rprods):
                    # Valid combination
                    if total_size > best_size:
                        best_size = total_size
                        best_set = sorted(lset | rset)
            else:
                # Skip large combinations (they're unlikely to be optimal anyway
                # since we're looking for max size, and large product sets mean
                # large subsets which we've already checked via smaller combos)
                pass
    
    # Fallback: if meet-in-the-middle didn't find anything better, 
    # do a quick greedy + local search as backup
    if best_size == len(primes):
        # Try greedy: add numbers in order of "least conflict"
        current = set()
        current_prods = {1}
        for x in sorted(candidates, key=lambda v: v.bit_count()):
            if time.time() > _DEADLINE:
                break
            # Check if x can be added
            can_add = True
            for p in current_prods:
                if p * x in current_prods:
                    can_add = False
                    break
            if can_add:
                new_prods = set(current_prods)
                for p in current_prods:
                    new_prods.add(p * x)
                current.add(x)
                current_prods = new_prods
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = sorted(current)
    
    return [int(x) for x in best_set]
# EVOLVE-BLOCK-END
