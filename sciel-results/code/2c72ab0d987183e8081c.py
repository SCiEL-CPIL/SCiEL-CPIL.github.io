"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    def _check_divisibility_optimized(sum_set):
        """
        Check that no two distinct values in sum_set divide each other.
        Uses the multiple checking method which is efficient for sparse sets 
        where elements are large.
        """
        if len(sum_set) <= 1:
            return True
        
        # Sort to help with early termination or just iterate
        # We don't strictly need to sort for the 'in' check, but it might help 
        # if we were doing pairwise. With 'in', order doesn't matter.
        
        max_val = max(sum_set)
        
        # Optimization: If 1 is in the set, it divides everything.
        if 1 in sum_set and len(sum_set) > 1:
            return False
            
        for a in sum_set:
            if a > max_val // 2:
                continue
            # Check multiples
            # a * 2, a * 3, ...
            # We can break early if a * k > max_val
            for k in range(2, max_val // a + 1):
                if (a * k) in sum_set:
                    return False
        return True

    def _can_add(current_sums, x):
        """
        Check if x can be added to the set defined by current_sums.
        Returns (True, new_sums) or (False, None).
        """
        # 1. Check for duplicates
        # New sums are x and s+x for s in current_sums
        # If x is in current_sums, duplicate.
        if x in current_sums:
            return False, None
        
        new_sums = set(current_sums)
        new_sums.add(x)
        
        # Check if s+x creates duplicates
        for s in current_sums:
            ns = s + x
            if ns in new_sums:
                return False, None
            new_sums.add(ns)
            
        # 2. Check divisibility
        # We only need to check if the NEW elements cause divisibility issues.
        # However, existing elements were already valid.
        # So we only need to check:
        # a) Does any new element divide any existing element?
        # b) Does any existing element divide any new element?
        # c) Do any two new elements divide each other?
        
        # New elements are {x} and {s+x for s in current_sums}.
        # Let new_vals = [x] + [s+x for s in current_sums]
        # But checking all pairs is expensive.
        # Let's use the multiple check on the FULL new_sums, but optimized.
        
        max_val = max(new_sums)
        
        # If the set is large, this is the bottleneck.
        # We can skip checking 'a' if 'a' is large.
        # Also, if 'a' is an existing element, we already know it doesn't divide other existing elements.
        # So we only need to check if 'a' divides a NEW element, or if a NEW element divides 'a'.
        
        # Let's identify new elements.
        # new_elements = set(new_sums) - set(current_sums)
        # This is O(N).
        
        # Actually, just running the full check is safer and simpler, 
        # provided we optimize the inner loop.
        
        for a in new_sums:
            if a > max_val // 2:
                continue
            for k in range(2, max_val // a + 1):
                if (a * k) in new_sums:
                    return False, None
        return True, new_sums

    def _greedy_construct(pool, deadline):
        elements = []
        current_sums = set()
        
        for x in pool:
            if time.time() > deadline:
                break
            ok, new_sums = _can_add(current_sums, x)
            if ok:
                elements.append(x)
                current_sums = new_sums
                
        return elements

    best_result = [31, 32, 34]
    best_size = 3
    
    rng = random.Random(12345)
    
    # Strategy 1: Descending (Large numbers first)
    # This tends to keep sums large, making divisibility less likely.
    pool_desc = list(range(256, 0, -1))
    
    # Strategy 2: Random shuffles
    # Strategy 3: Specific pools (e.g. primes, high range)
    
    pools = [
        pool_desc,
        list(range(256, 100, -1)), # High range descending
        list(range(200, 257)),     # High range ascending
    ]
    
    for _ in range(20):
        p = list(range(256, 0, -1))
        rng.shuffle(p)
        pools.append(p)
        
    # Add some heuristic pools
    # Odds
    pools.append([x for x in range(256, 0, -1) if x % 2 != 0])
    # Primes (approx)
    def is_prime(n):
        if n < 2: return False
        for i in range(2, int(n**0.5)+1):
            if n % i == 0: return False
        return True
    pools.append([x for x in range(256, 0, -1) if is_prime(x)])
            
    for pool in pools:
        if time.time() > deadline:
            break
        result = _greedy_construct(pool, deadline)
        if len(result) > best_size:
            best_size = len(result)
            best_result = result
            
    # Local search / Extension
    if best_size < 30 and time.time() < deadline - 5:
        current_elements = list(best_result)
        current_sums = set()
        # Reconstruct sums for the best result
        for mask in range(1, 1 << len(current_elements)):
            s = 0
            for i in range(len(current_elements)):
                if mask & (1 << i):
                    s += current_elements[i]
            current_sums.add(s)
            
        # Try adding any number from 256 down to 1 not in current_elements
        candidates = [x for x in range(256, 0, -1) if x not in current_elements]
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for x in candidates:
                if time.time() > deadline:
                    break
                ok, new_sums = _can_add(current_sums, x)
                if ok:
                    current_elements.append(x)
                    current_sums = new_sums
                    candidates.remove(x)
                    improved = True
                    if len(current_elements) > best_size:
                        best_size = len(current_elements)
                        best_result = list(current_elements)

    # Final validation
    if len(best_result) > 0:
        # Verify the best result
        final_sums = set()
        for mask in range(1, 1 << len(best_result)):
            s = 0
            for i in range(len(best_result)):
                if mask & (1 << i):
                    s += best_result[i]
            final_sums.add(s)
            
        if not _check_divisibility_optimized(final_sums):
            best_result = [31, 32, 34]
            
    return [int(x) for x in best_result]

# EVOLVE-BLOCK-END
