"""Program for large sets with distinct subset products."""
# EVOLVE-BLOCK-START
import time
import random
from math import isqrt

_DEADLINE = 55.0  # safety margin before 60s kill

def _is_prime(a):
    if a < 2:
        return False
    if a % 2 == 0:
        return a == 2
    d = 3
    while d * d <= a:
        if a % d == 0:
            return False
        d += 2
    return True

# Precompute prime factors and exponent vectors for numbers 2..40
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
NUMBERS = list(range(2, 41))
EXP_VECTORS = {}

for n in NUMBERS:
    vec = [0] * len(PRIMES)
    temp = n
    for i, p in enumerate(PRIMES):
        while temp % p == 0:
            vec[i] += 1
            temp //= p
    # If temp > 1, it's a prime > 37, but max is 40, so no such case.
    EXP_VECTORS[n] = tuple(vec)

def _subset_sums(vecs):
    """Compute set of all subset sums of a list of tuples."""
    sums = {tuple([0] * len(vecs[0]))} # Zero vector
    for v in vecs:
        new_sums = set()
        for s in sums:
            # Tuple addition
            ns = tuple(s[i] + v[i] for i in range(len(v)))
            new_sums.add(ns)
        sums |= new_sums
    return sums

def _can_add(current_vecs, v):
    """
    Check if adding vector v to current_vecs preserves distinct subset sums.
    Returns True if valid.
    current_vecs: list of tuples
    v: tuple
    """
    # Compute subset sums of current_vecs
    # Optimization: We can pass the precomputed sums if we maintain them, 
    # but for simplicity in the search loop, we recompute. 
    # 2^12 = 4096 sums. 2^13 = 8192. This is fast enough.
    sums = _subset_sums(current_vecs)
    
    # Check for collision: exists s1, s2 in sums such that s1 + v == s2?
    # i.e., s2 - s1 == v?
    # Or simply: check if (s + v) is in sums for any s in sums.
    # If s + v is in sums, then s and s+v are two distinct subset sums (unless v=0, which it isn't)
    # that collide? No.
    # If s + v = s2, then the subset corresponding to s2 has sum s+v.
    # The subset corresponding to s has sum s.
    # The new element v forms a subset {v} with sum v.
    # And subset S_u (sum s) + v has sum s+v.
    # Collision occurs if s+v is already a subset sum of the OLD set.
    # Then the new subset (S_u union {v}) has the same product as the old subset S_w (where sum(S_w) = s+v).
    # So we just need to check if any s+v is in sums.
    
    for s in sums:
        ns = tuple(s[i] + v[i] for i in range(len(v)))
        if ns in sums:
            return False
    return True

def construct():
    t0 = time.time()
    
    # Start with all 12 primes
    current_nums = list(PRIMES)
    current_vecs = [EXP_VECTORS[n] for n in current_nums]
    
    composites = [n for n in NUMBERS if not _is_prime(n)]
    
    best_nums = list(current_nums)
    best_vecs = list(current_vecs)
    
    # Greedy pass
    for x in composites:
        if time.time() - t0 > _DEADLINE:
            break
        v = EXP_VECTORS[x]
        if _can_add(current_vecs, v):
            current_vecs.append(v)
            current_nums.append(x)
            if len(current_nums) > len(best_nums):
                best_nums = list(current_nums)
                best_vecs = list(current_vecs)
    
    # Hill Climbing / Local Search
    # Try to add more if possible, or swap to allow more.
    rng = random.Random(12345)
    improved = True
    while improved and (time.time() - t0) < _DEADLINE:
        improved = False
        
        # 1. Try adding any remaining composite
        for x in composites:
            if x in current_nums:
                continue
            v = EXP_VECTORS[x]
            if _can_add(current_vecs, v):
                current_vecs.append(v)
                current_nums.append(x)
                improved = True
                if len(current_nums) > len(best_nums):
                    best_nums = list(current_nums)
                    best_vecs = list(current_vecs)
                break
        
        # 2. Try swapping one element for two others (1-for-2 swap)
        # This can increase size by 1 if successful.
        if not improved and (time.time() - t0) < _DEADLINE:
            # Iterate through current elements to remove
            for i in range(len(current_nums)):
                if time.time() - t0 > _DEADLINE:
                    break
                
                removed_num = current_nums.pop(i)
                removed_vec = current_vecs.pop(i)
                
                # Try to add two composites not in current
                added_count = 0
                added_items = []
                
                for x in composites:
                    if x in current_nums or x == removed_num:
                        continue
                    v = EXP_VECTORS[x]
                    if _can_add(current_vecs, v):
                        current_vecs.append(v)
                        current_nums.append(x)
                        added_items.append(x)
                        added_count += 1
                        if added_count == 2:
                            break
                
                if added_count == 2:
                    # Success: removed 1, added 2. Net +1.
                    if len(current_nums) > len(best_nums):
                        best_nums = list(current_nums)
                        best_vecs = list(current_vecs)
                    improved = True
                    # We can break out of the removal loop as we found an improvement
                    break
                else:
                    # Restore state
                    # Remove the added items
                    for x in added_items:
                        idx = current_nums.index(x)
                        current_nums.pop(idx)
                        current_vecs.pop(idx)
                    # Restore removed item
                    current_nums.append(removed_num)
                    current_vecs.append(removed_vec)
                    
            # If we broke out due to improvement, the loop continues to check for more additions
            # If we didn't break, we tried all removals.

    # Final verification
    # Check if best_nums has distinct subset products
    # We can reuse _can_add logic or just check the final set
    # For safety, we verify the best set.
    if len(best_nums) > 0:
        best_vecs_check = [EXP_VECTORS[n] for n in best_nums]
        # Check distinctness of best_vecs_check
        # We can just call _can_add on an empty set and add all? 
        # Or simpler: check if all subset sums are distinct.
        # 2^16 = 65536 sums. Fast.
        sums = _subset_sums(best_vecs_check)
        if len(sums) != (1 << len(best_vecs_check)):
             # Fallback to primes
             best_nums = list(PRIMES)
             
    return [int(x) for x in best_nums]
# EVOLVE-BLOCK-END
