"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85 seconds safety margin before 90s kill

    def _check_divisibility_sieve(sum_set):
        """
        Check that no two distinct values in sum_set divide each other.
        Uses a sieve-like approach: mark all multiples of each element.
        Much faster than pairwise checking for dense sum sets.
        """
        if len(sum_set) <= 1:
            return True
        max_val = max(sum_set)
        # Mark which values are in the set
        present = set(sum_set)
        for si in sum_set:
            if si <= 1:
                if len(sum_set) > 1:
                    return False
                continue
            # Check if any multiple of si (other than si itself) is in the set
            for k in range(2, max_val // si + 1):
                if (si * k) in present:
                    return False
        return True

    def _get_subset_sums(elements):
        """Get all non-empty subset sums as a set using DP."""
        if not elements:
            return set()
        sums = {0}
        for e in elements:
            new_sums = set()
            for s in sums:
                new_sums.add(s + e)
            sums = sums | new_sums
        sums.discard(0)
        return sums

    def _can_add_fast(current_sums, x):
        """
        Fast check: can x be added?
        Returns (True, new_sums) if valid, (False, None) otherwise.
        current_sums is a set of current non-empty subset sums.
        """
        # New sums: current_sums ∪ {x} ∪ {s+x for s in current_sums}
        # Check for duplicates first (efficiency and correctness)
        new_sums = set(current_sums)
        if x in new_sums:
            return False, None
        new_sums.add(x)
        
        for s in current_sums:
            ns = s + x
            if ns in new_sums:
                return False, None
            new_sums.add(ns)
        
        # Check divisibility using sieve approach
        max_val = max(new_sums)
        for si in new_sums:
            if si <= 1:
                if len(new_sums) > 1:
                    return False, None
                continue
            for k in range(2, max_val // si + 1):
                if (si * k) in new_sums:
                    return False, None
        return True, new_sums

    def _greedy_construct(pool, deadline):
        elements = []
        current_sums = set()
        
        for x in pool:
            if time.time() > deadline:
                break
            ok, new_sums = _can_add_fast(current_sums, x)
            if ok:
                elements.append(x)
                current_sums = new_sums
                
        return elements

    def _local_search_swap_add(elements, current_sums, deadline):
        """
        Try to improve by removing one element and adding one or more new elements.
        Returns (improved_elements, improved_sums) if improvement found, else (None, None).
        """
        if not elements:
            return None, None
        
        n = len(elements)
        best_improvement = 0
        best_new_elements = None
        best_new_sums = None
        
        # For each element to remove, try adding candidates
        for remove_idx in range(n):
            if time.time() > deadline:
                break
            
            # Compute sums without this element
            remaining_elements = elements[:remove_idx] + elements[remove_idx+1:]
            remaining_sums = _get_subset_sums(remaining_elements)
            
            # Try adding candidates
            candidates = [x for x in range(2, 257) if x not in remaining_elements]
            # Sort candidates to try promising ones first
            candidates.sort(reverse=True)
            
            added_any = False
            new_elements = list(remaining_elements)
            new_sums = set(remaining_sums)
            
            for x in candidates:
                if time.time() > deadline:
                    break
                ok, updated_sums = _can_add_fast(new_sums, x)
                if ok:
                    new_elements.append(x)
                    new_sums = updated_sums
                    added_any = True
            
            if added_any and len(new_elements) > len(elements):
                if len(new_elements) > len(elements) + best_improvement:
                    best_improvement = len(new_elements) - len(elements)
                    best_new_elements = new_elements
                    best_new_sums = new_sums
        
        return best_new_elements, best_new_sums

    best_result = [31, 32, 34]
    best_size = 3
    
    rng = random.Random(12345)
    
    # Strategy:
    # 1. Try descending order (large numbers first) - often works well.
    # 2. Try random shuffles of various pools.
    # 3. Try specific pools (primes, high range).
    
    pools = [
        list(range(256, 1, -1)),  # Descending
        list(range(2, 257)),      # Ascending
        [x for x in range(2, 257) if x % 2 != 0], # Odds
        [x for x in range(2, 257) if x % 2 == 0], # Evens
        # Numbers in higher ranges where divisibility conflicts are less likely
        [x for x in range(128, 257)],  # Upper half
        [x for x in range(192, 257)],  # Upper quarter
        [x for x in range(224, 257)],  # Top 33
        # Prime-like: numbers with few divisors
        [x for x in range(2, 257) if x % 2 != 0 and x % 3 != 0],  # Not div by 2 or 3
        [x for x in range(2, 257) if x % 2 != 0 and x % 3 != 0 and x % 5 != 0],  # Not div by 2,3,5
    ]
    
    # Add random pools with different strategies - more diverse
    for _ in range(12):
        p = list(range(2, 257))
        rng.shuffle(p)
        pools.append(p)
    
    # Weighted random: favor larger numbers
    for _ in range(6):
        p = list(range(80, 257))
        rng.shuffle(p)
        pools.append(p)
    
    # High range pools with various orderings
    for low in [80, 120, 160, 200]:
        p = list(range(low, 257))
        pools.append(p)
        p2 = p[:]
        rng.shuffle(p2)
        pools.append(p2)
        p3 = list(range(256, low - 1, -1))
        pools.append(p3)
        
    # Mixed: interleave high and medium
    for _ in range(4):
        high = list(range(180, 257))
        med = list(range(80, 180))
        rng.shuffle(high)
        rng.shuffle(med)
        p = []
        hi, mi = 0, 0
        while hi < len(high) or mi < len(med):
            if hi < len(high) and (mi >= len(med) or rng.random() < 0.6):
                p.append(high[hi])
                hi += 1
            else:
                p.append(med[mi])
                mi += 1
        pools.append(p)

    for pool in pools:
        if time.time() > deadline:
            break
        result = _greedy_construct(pool, deadline)
        if len(result) > best_size:
            best_size = len(result)
            best_result = result

    # Local search / Extension
    # Try to extend the best result by adding numbers not in it.
    if best_size < 30 and time.time() < deadline - 10:
        current_elements = list(best_result)
        current_sums = _get_subset_sums(current_elements)
        
        # Try multiple candidate orderings for extension
        for order_strategy in ['desc', 'asc', 'random1', 'random2', 'random3']:
            if time.time() > deadline - 10:
                break
            if order_strategy == 'desc':
                candidates = [x for x in range(256, 1, -1) if x not in current_elements]
            elif order_strategy == 'asc':
                candidates = [x for x in range(2, 257) if x not in current_elements]
            elif order_strategy == 'random1':
                candidates = [x for x in range(2, 257) if x not in current_elements]
                rng.shuffle(candidates)
            elif order_strategy == 'random2':
                # Favor larger numbers in random
                candidates = [x for x in range(2, 257) if x not in current_elements]
                rng.shuffle(candidates)
                candidates.sort(key=lambda x: -x)
                rng.shuffle(candidates)
            else:
                # Random with slight preference for numbers in upper range
                candidates = [x for x in range(2, 257) if x not in current_elements]
                rng.shuffle(candidates)
                candidates.sort(key=lambda x: (-x if x > 128 else x))
            
            improved = True
            while improved and time.time() < deadline - 10:
                improved = False
                for x in candidates[:]:  # Copy to allow removal
                    if time.time() > deadline:
                        break
                    ok, new_sums = _can_add_fast(current_sums, x)
                    if ok:
                        current_elements.append(x)
                        current_sums = new_sums
                        candidates.remove(x)
                        improved = True
                        if len(current_elements) > best_size:
                            best_size = len(current_elements)
                            best_result = list(current_elements)

    # Swap-add local search: try removing one element and adding one or more
    if best_size < 30 and time.time() < deadline - 5:
        current_elements = list(best_result)
        current_sums = _get_subset_sums(current_elements)
        
        for _ in range(8):  # Multiple rounds of swap-add
            if time.time() > deadline - 5:
                break
            new_elements, new_sums = _local_search_swap_add(current_elements, current_sums, deadline)
            if new_elements and len(new_elements) > len(current_elements):
                current_elements = new_elements
                current_sums = new_sums
                if len(current_elements) > best_size:
                    best_size = len(current_elements)
                    best_result = list(current_elements)
            else:
                break  # No improvement, stop

    # Final validation
    if len(best_result) > 0:
        sums = _get_subset_sums(best_result)
        if not _check_divisibility_sieve(sums):
            # Fallback
            best_result = [31, 32, 34]
            
    return [int(x) for x in best_result]

# EVOLVE-BLOCK-END
