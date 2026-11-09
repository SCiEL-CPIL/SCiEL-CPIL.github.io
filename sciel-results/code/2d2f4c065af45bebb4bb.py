"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85 seconds safety margin before 90s kill
    
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
        # 1. Check distinctness
        if x in current_sums:
            return False, None
        
        # Generate new sums from x
        # new_sums = {x} U {s+x for s in current_sums}
        # We need to check if any of these collide with current_sums or each other.
        # Collision with current_sums:
        # (s+x) in current_sums?
        for s in current_sums:
            if (s + x) in current_sums:
                return False, None
        
        # If we pass this, all new sums are distinct from old sums.
        # And x is distinct from old sums.
        # And new sums {s+x} are distinct from each other (since s are distinct).
        # So distinctness is satisfied.
        
        # 2. Check divisibility
        # Let S = current_sums
        # Let N = {x} U {s+x | s in S}
        # We need no u, v in S U N, u != v, such that u | v.
        
        # Check x against S
        for s in current_sums:
            if x % s == 0 or s % x == 0:
                return False, None
        
        # Check x against N (excluding x itself)
        # N contains s+x.
        # x | (s+x) <=> x | s
        # (s+x) | x is impossible for s>0.
        # So we just need x not to divide s and s not to divide x.
        # This is already checked in the loop above.
        
        # Check S against N (excluding x)
        # N contains s+x.
        # For a in S, b+x in N:
        # a | (b+x) or (b+x) | a
        
        # Check N (excluding x) against N (excluding x)
        # For s1+x, s2+x:
        # (s1+x) | (s2+x) <=> (s1+x) | (s2-s1)
        
        # Optimization:
        # We will iterate through S to check against N.
        # And iterate through pairs in S to check N vs N.
        
        # To speed up, we can sort S.
        # But S is a set.
        
        # Check S vs {s+x}
        for a in current_sums:
            for b in current_sums:
                val = b + x
                # Check a | val
                if a <= val:
                    if val % a == 0:
                        return False, None
                # Check val | a
                if val <= a:
                    if a % val == 0:
                        return False, None
        
        # Check {s+x} vs {s+x}
        # (s1+x) | (s2+x) <=> (s1+x) | (s2-s1)
        # We only need to check if s1+x is a divisor of the difference.
        # Note: s1+x is positive.
        # If s1 == s2, diff is 0. 0 is divisible by anything? 
        # But s1+x and s2+x are distinct elements in N if s1 != s2.
        # If s1 == s2, they are the same element, so we don't check.
        # So we iterate s1 != s2.
        
        # This is O(|S|^2).
        # We can optimize by noting that if s1+x > |s2-s1|, then s1+x cannot divide |s2-s1| (unless |s2-s1|=0, which is excluded).
        # So we only need to check if s1+x <= |s2-s1|.
        
        for s1 in current_sums:
            u = s1 + x
            for s2 in current_sums:
                if s1 == s2:
                    continue
                diff = abs(s2 - s1)
                if u <= diff:
                    if diff % u == 0:
                        return False, None
                    # Also need to check if the other way around?
                    # (s2+x) | (s1+x) <=> (s2+x) | (s1-s2)
                    # This is symmetric. If we check all pairs (s1, s2), we cover both.
                    # Wait, the condition "u <= diff" prunes many cases.
                    # If u > diff, then u cannot divide diff.
                    # So we are safe.

        # If we reach here, it's valid.
        new_sums = set(current_sums)
        new_sums.add(x)
        for s in current_sums:
            new_sums.add(s + x)
            
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
            # Sort candidates to try promising ones first (larger numbers often safer)
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
        [x for x in range(224, 257)],  # Top 16
        # Prime-like: numbers with few divisors
        [x for x in range(2, 257) if x % 2 != 0 and x % 3 != 0],  # Not div by 2 or 3
        [x for x in range(2, 257) if x % 2 != 0 and x % 3 != 0 and x % 5 != 0],  # Not div by 2,3,5
    ]
    
    # Add random pools with different strategies
    for _ in range(8):
        p = list(range(2, 257))
        rng.shuffle(p)
        pools.append(p)
    
    # Weighted random: favor larger numbers
    for _ in range(4):
        p = list(range(100, 257))
        rng.shuffle(p)
        pools.append(p)
    
    # High range pools with various orderings
    for low in [100, 150, 200]:
        p = list(range(low, 257))
        pools.append(p)
        p2 = p[:]
        rng.shuffle(p2)
        pools.append(p2)
        p3 = list(range(256, low - 1, -1))
        pools.append(p3)
        
    # Mixed: interleave high and medium
    for _ in range(3):
        high = list(range(200, 257))
        med = list(range(100, 200))
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
        for order_strategy in ['desc', 'asc', 'random1', 'random2']:
            if time.time() > deadline - 10:
                break
            if order_strategy == 'desc':
                candidates = [x for x in range(256, 1, -1) if x not in current_elements]
            elif order_strategy == 'asc':
                candidates = [x for x in range(2, 257) if x not in current_elements]
            elif order_strategy == 'random1':
                candidates = [x for x in range(2, 257) if x not in current_elements]
                rng.shuffle(candidates)
            else:
                # Favor larger numbers in random
                candidates = [x for x in range(2, 257) if x not in current_elements]
                rng.shuffle(candidates)
                candidates.sort(key=lambda x: -x)
                # Re-shuffle but with bias
                rng.shuffle(candidates)
            
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
        
        for _ in range(5):  # Multiple rounds of swap-add
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
        # Check distinctness
        if len(sums) != (1 << len(best_result)) - 1:
            best_result = [31, 32, 34]
        else:
            # Check divisibility
            sums_list = sorted(sums)
            sums_set = set(sums_list)
            max_sum = sums_list[-1]
            valid = True
            for s in sums_list:
                for k in range(2, max_sum // s + 1):
                    if k * s in sums_set:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                best_result = [31, 32, 34]
            
    return [int(x) for x in best_result]

# EVOLVE-BLOCK-END
