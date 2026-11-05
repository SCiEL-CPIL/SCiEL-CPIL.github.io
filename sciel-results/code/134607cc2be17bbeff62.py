"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from functools import reduce

def _lcm(a, b):
    return a * b // gcd(a, b)

def _is_valid_fast(S):
    """
    Check if S is a valid reciprocal-subsum-free set.
    Uses integer LCM and bitset subset sum.
    """
    if not S:
        return True
    
    # Calculate LCM of all elements in S
    # LCM of 1..24 is ~ 5e8, which fits in 64-bit int, but Python ints are fine.
    try:
        L = reduce(_lcm, S)
    except OverflowError:
        return False # Should not happen for N=24

    S_list = list(S)
    n = len(S_list)
    
    # Precompute L / x for all x in S
    # target[x] = L / x
    # summands[x] = [L / y for y in S if y != x and L/y <= L/x]
    
    # Optimization: If sum of all other reciprocals < 1/x, it's valid for x.
    # Sum(1/y for y != x) < 1/x  <=>  Sum(L/y for y != x) < L/x
    # We can check this quickly with floats or exact ints.
    
    for x in S_list:
        target = L // x
        
        # Get valid summands: L/y for y in S, y != x, such that L/y <= target
        # If L/y > target, we can't use it in a sum that equals target (since all are positive)
        # Actually, we can't use ANY summand > target.
        
        summands = []
        total_sum = 0
        for y in S_list:
            if y == x:
                continue
            val = L // y
            if val <= target:
                summands.append(val)
                total_sum += val
        
        # If total sum of all possible summands < target, x is safe.
        if total_sum < target:
            continue
        
        # Check if target is in subset sums of summands
        # Use bitset
        # bits[i] = 1 if sum i is reachable
        # Start with 0 reachable
        bits = 1
        
        # Mask to keep bitset size manageable (up to target)
        # mask = (1 << (target + 1)) - 1
        # This might be huge if target is large.
        # But we only need to know if 'target' is reachable.
        # We can prune bits if they exceed target?
        # bits = bits & mask
        
        for a in summands:
            bits |= (bits << a)
            # Prune bits above target to keep integer small
            # if target is ~5e8, mask is 2^(5e8) which is huge.
            # However, Python big ints are sparse in memory? No, they are dense arrays of digits.
            # 5e8 bits is 62.5 MB.
            # Doing this for 24 elements is 1.5 GB.
            # And the shift operation is O(size of int).
            # This might be too slow.
            
            # Alternative: Use a set of reachable sums?
            # If the number of reachable sums is small, set is faster.
            # But it can explode.
            
            # Let's stick to the bitset but be careful.
            # For x=1, target=L. Summands are L/2, L/3...
            # L/2 is ~2.5e8.
            # bits << 2.5e8.
            # This creates a huge integer.
            
            # Is there a better way?
            # Maybe we don't need to check all x?
            # If we construct the set greedily, we only check the new element x against the existing set S_old.
            # When adding x to S_old, we check if 1/x can be formed by S_old.
            # AND we must check if any existing y in S_old can be formed by S_old U {x} \ {y}.
            # The second part is expensive.
            
            # However, if we add elements in increasing order of value (decreasing reciprocal),
            # 1/x is large.
            # If we add small numbers first, 1/1, 1/2...
            # When adding x, we check if 1/x is in sum(S_old).
            # Do we need to re-check old elements?
            # Adding a new term 1/x makes it EASIER to form sums for old elements.
            # So we MUST re-check.
            
            # Unless... we use a property.
            # If S is valid, and we add x such that 1/x > sum(1/y for y in S), then:
            # 1. 1/x cannot be formed by S (sum too small).
            # 2. For any y in S, can 1/y be formed by S U {x} \ {y}?
            #    Sum of (S \ {y}) is < sum(S) < 1/x.
            #    Adding 1/x: Sum((S \ {y}) U {x}) = Sum(S \ {y}) + 1/x.
            #    We need to check if 1/y = Sum(subset of S\{y}) + k * (1/x)? No, subset.
            #    So 1/y = A + B/x where A is sum of subset of S\{y} and B is 0 or 1.
            #    If B=0, 1/y = A. Since S was valid, A != 1/y (as x not in S).
            #    Wait, S was valid, so 1/y is NOT formed by S\{y}.
            #    So B must be 1?
            #    1/y = A + 1/x => 1/x = 1/y - A.
            #    So 1/x must be formed by 1/y - A.
            #    This is a specific condition.
            
            # This suggests that if we add x such that 1/x is "large" compared to the rest,
            # we only need to check a few cases.
            
            # But generally, re-checking is required.
            
            # Given the time limit (90s) and N=24, the LCM bitset approach might be too slow for full re-validation.
            # However, we can optimize the bitset by only keeping bits up to target.
            # And we can early exit if target is found.
            
            # Let's try the bitset with early exit and pruning.
            # If it's too slow, we fall back to Fraction DFS for the final check.
            
            # Actually, let's use the Fraction DFS but with memoization and better pruning?
            # Or use the bitset only for the "add" check (1/x vs S_old), and do a full validation at the end?
            # If we do full validation at the end, we might waste time on invalid intermediate sets.
            
            # Let's try a hybrid:
            # 1. Greedy add with fast check (1/x vs S_old).
            # 2. Full validation of the final set using Fraction DFS (which was working and found size 19).
            # 3. Local search: Try swaps. Validate full set.
            
            # The current program already does this.
            # The improvement should be in the search strategy (more restarts, better heuristics)
            # enabled by a slightly faster validation or just more aggressive local search.
            
            # Let's assume the Fraction DFS is the bottleneck.
            # We can speed up _can_form_target by using a set of reachable sums with pruning.
            
            pass # Placeholder for thought process
            
        # End of bitset consideration
        
        # Let's implement the Fraction DFS optimization:
        # Use a set of reachable sums, but prune sums > target.
        # And use the suffix sum pruning.
        
        reachable = {0}
        # Sort summands descending?
        # No, let's just iterate.
        # For each element y in S\{x}:
        #   new_reachable = {s + 1/y for s in reachable if s + 1/y <= target}
        #   reachable = reachable | new_reachable
        #   if target in reachable: return False (invalid)
        
        # This is O(2^k) worst case, but pruning helps.
        # For k=20, 2^20 is 1M. Might be okay for a few checks.
        
        pass

    return True

def _can_form_target_frac(target, elements):
    """
    Fast check using set of reachable sums with pruning.
    """
    if not elements:
        return False
    
    # Sort elements by value ascending (reciprocals descending)
    # This helps prune s + 1/y > target
    elems = sorted(elements)
    
    reachable = {0}
    
    for y in elems:
        r = Fraction(1, y)
        new_sums = set()
        for s in reachable:
            ns = s + r
            if ns <= target:
                new_sums.add(ns)
            else:
                # Since r is positive and elems are sorted ascending (r descending),
                # if s + r > target, then for any s' > s, s' + r > target.
                # But we iterate all s.
                # We can break if we sort reachable? No, reachable is a set.
                pass
        
        reachable.update(new_sums)
        
        if target in reachable:
            return True
            
    return False

def _is_valid(S):
    S_list = list(S)
    for i, x in enumerate(S_list):
        target = Fraction(1, x)
        others = [y for y in S_list if y != x]
        if _can_form_target_frac(target, others):
            return False
    return True

def _find_best_set():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    best_set = []
    best_size = 0
    
    # Randomized Restart Greedy
    # Try different orders
    
    for _ in range(100): # Max restarts, break on time
        if time.time() > deadline - 5.0:
            break
            
        order = list(range(1, 25))
        random.shuffle(order)
        
        current_set = []
        for x in order:
            # Fast check: Can 1/x be formed by current_set?
            if _can_form_target_frac(Fraction(1, x), current_set):
                continue
            
            # Tentative add
            test_set = current_set + [x]
            
            # Full check: Is test_set valid?
            # This is expensive.
            # Optimization: Only do full check if we think it's likely to be valid?
            # Or just do it.
            
            if _is_valid(test_set):
                current_set = test_set
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            
        # Local Search on best_set
        if time.time() > deadline - 2.0:
            break
            
        current = best_set[:]
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            
            # Try to add any element
            for x in range(1, 25):
                if x in current:
                    continue
                test_set = current + [x]
                if _is_valid(test_set):
                    current = test_set
                    improved = True
                    break
            
            # Try swaps (remove y, add z)
            if not improved:
                for y in current:
                    reduced = [z for z in current if z != y]
                    for z in range(1, 25):
                        if z == y or z in reduced:
                            continue
                        test_set = reduced + [z]
                        if _is_valid(test_set):
                            # Only accept if it leads to a "better" state?
                            # Since size is same, we accept to explore.
                            # Risk of cycling.
                            # Use a visited set?
                            current = test_set
                            improved = True
                            break
                    if improved:
                        break
                        
        if len(current) > best_size:
            best_size = len(current)
            best_set = current[:]

    return best_set

def construct():
    start_time = time.time()
    
    result = _find_best_set()
    
    if not result:
        result = [13]
        
    # Final validation
    if not _is_valid(result):
        # Fallback to known valid set
        result = list(range(13, 25))
        
    return result
# EVOLVE-BLOCK-END
