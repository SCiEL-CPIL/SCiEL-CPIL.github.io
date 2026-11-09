"""
Rewritten program: Randomized backtracking search for cardinality-separated subset sums.
"""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _check_validity(S):
    """
    Checks if the set S satisfies the condition:
    For any two subsets A, B with |A| != |B|, sum(A) != sum(B).
    This is equivalent to: The sets of sums for each cardinality are pairwise disjoint.
    """
    n = len(S)
    if n == 0:
        return True
    
    # sums_by_card[k] is a set of all possible sums of subsets of S with size k
    sums_by_card = [set() for _ in range(n + 1)]
    sums_by_card[0].add(0)
    
    # We can compute this incrementally or by iterating all subsets.
    # For n <= 20, iterating all 2^n subsets is too slow if done naively in Python loops,
    # but we can use DP or simple recursion.
    # Actually, for n=15, 2^15 = 32768 subsets. This is fast enough to just enumerate.
    
    # Optimization: Just enumerate all subsets
    valid = True
    # We need to check if any sum appears in two different cardinalities.
    # sum_to_cards: dict mapping sum -> set of cardinalities that achieve it
    
    sum_to_cards = {}
    
    # Iterate all masks
    for mask in range(1 << n):
        s = 0
        count = 0
        for i in range(n):
            if mask & (1 << i):
                s += S[i]
                count += 1
        
        if s in sum_to_cards:
            # If this sum was seen before, check if the cardinality is different
            if count not in sum_to_cards[s]:
                return False
        else:
            sum_to_cards[s] = {count}
            
    return True

def _try_find_set(target_size, timeout=5.0):
    """
    Attempts to find a valid set of size target_size.
    Uses randomized backtracking.
    Returns the set if found, else None.
    """
    start_time = time.time()
    elements_pool = list(range(1, 257))
    
    def backtrack(current_set, available_elements):
        if len(current_set) == target_size:
            return current_set
        
        if time.time() - start_time > timeout:
            return None
            
        # Heuristic: Sort available elements. 
        # We want to try smaller elements first to keep sums small, 
        # but randomize to avoid getting stuck in one specific "bad" pattern.
        # Let's pick a random subset of candidates to try, prioritizing smaller ones.
        
        # Filter candidates that might be "too large" immediately? 
        # If current sum of largest k elements + new element > 256*k? No, elements are distinct 1..256.
        
        # To improve search, we can try candidates in a randomized order biased towards smaller values.
        # Let's just shuffle and try.
        
        candidates = available_elements[:]
        random.shuffle(candidates)
        
        # Optional: Sort by value to try smaller ones first, but with some noise?
        # Actually, for subset sums, smaller numbers are generally "safer" for disjointness 
        # because they create denser but lower range sums. Larger numbers create sparser, higher sums.
        # However, mixing them is key.
        
        for x in candidates:
            new_set = current_set + [x]
            # Check validity of new_set
            # Optimization: Only check the new sums introduced by x?
            # Full check is O(2^n). For n=15, 32k ops. 15 steps * 256 candidates * 32k ops is too slow for full check at every step.
            
            # We need a faster incremental check or a smarter heuristic.
            # Let's use the incremental sum set update.
            
            # However, implementing incremental check with backtracking state is complex.
            # Let's try a different approach: 
            # Generate a random permutation of the pool and greedily pick? No, greedy fails.
            
            # Let's stick to backtracking but optimize the check.
            # For small n (up to 12-15), full check is ~32k-300k operations.
            # If we have 256 candidates, that's 8M-76M ops per level. Too slow in Python.
            
            # Alternative: Use the property that if we maintain `sums_by_card` incrementally, 
            # we can check validity in O(N * MaxSum) or similar.
            
            # Let's refactor to pass `sums_by_card` and `sum_to_cards` state.
            
            pass # Placeholder, logic below
            
    # The simple backtracking with full check is too slow.
    # Let's use a different strategy: 
    # 1. Generate a random set of size N.
    # 2. Check validity.
    # 3. If invalid, identify conflicting pairs and remove one, then try to repair?
    
    # Better Strategy: 
    # Randomized Greedy with Local Search / Backtracking on Conflicts.
    
    # Let's try a "Randomized Incremental Construction" with a fast validity check.
    # We maintain `sums_by_card` as a list of sets.
    # When adding x, we update sums_by_card.
    # We check if any new sum intersects with existing sums of different cardinalities.
    
    def fast_add_check(current_sums_by_card, x):
        """
        Check if adding x to a set represented by current_sums_by_card is valid.
        Returns new_sums_by_card if valid, else None.
        """
        n_current = len(current_sums_by_card) - 1
        new_sums_by_card = [set() for _ in range(n_current + 2)]
        
        # Copy existing
        for i in range(n_current + 1):
            new_sums_by_card[i] = set(current_sums_by_card[i])
            
        # Add new sums involving x
        # New sums of size k+1 are {s + x for s in current_sums_by_card[k]}
        # We must ensure these new sums do not conflict with ANY existing sum in new_sums_by_card[j] for j != k+1.
        # Note: new_sums_by_card[k+1] already contains old sums of size k+1 (from previous elements).
        # So we are adding to new_sums_by_card[k+1] the set {s+x | s in current_sums_by_card[k]}.
        
        for k in range(n_current, -1, -1):
            if not current_sums_by_card[k]:
                continue
            new_sums_for_k_plus_1 = {s + x for s in current_sums_by_card[k]}
            
            # Check for conflicts
            # 1. Conflict with existing sums in new_sums_by_card[k+1] (which are sums of size k+1 from previous elements)
            #    Wait, sums of the SAME cardinality can repeat? 
            #    "Repeated sums within the SAME cardinality are allowed."
            #    So we only care if new_sums_for_k_plus_1 intersects with new_sums_by_card[j] for j != k+1.
            
            for j in range(len(new_sums_by_card)):
                if j == k + 1:
                    continue
                if new_sums_by_card[j]:
                    # Intersection check
                    if new_sums_for_k_plus_1 & new_sums_by_card[j]:
                        return None
            
            # If no conflict, add to new_sums_by_card[k+1]
            new_sums_by_card[k+1].update(new_sums_for_k_plus_1)
            
        return new_sums_by_card

    def backtrack_fast(current_set, current_sums_by_card, available_elements, depth):
        if len(current_set) == target_size:
            return current_set
        
        if time.time() - start_time > timeout:
            return None
            
        candidates = available_elements[:]
        random.shuffle(candidates)
        
        # Heuristic: Try smaller elements first? 
        # Let's sort candidates to try smaller ones first, but keep some randomness.
        # Actually, just shuffling is fine for now.
        
        for x in candidates:
            new_sums = fast_add_check(current_sums_by_card, x)
            if new_sums is not None:
                result = backtrack_fast(
                    current_set + [x], 
                    new_sums, 
                    [y for y in available_elements if y != x], 
                    depth + 1
                )
                if result is not None:
                    return result
        return None

    # Initial state
    initial_sums = [set()]
    initial_sums[0].add(0)
    
    result = backtrack_fast([], initial_sums, elements_pool, 0)
    return result

def construct():
    """
    Returns a list of distinct integers in 1..256 satisfying the cardinality-separated subset sum property.
    """
    start_time = time.time()
    deadline = start_time + 80.0 # Safety margin
    
    best_set = []
    
    # Try to find larger sets first
    # Max possible size? 
    # Sum of 1..60 is 1830. Max sum for size 1 is 256.
    # The constraint is tight. Size 15-20 is a good target.
    
    # We will try sizes from high to low, with limited time per size.
    # Or we can just run a search for a fixed high target and return whatever we find if we have to.
    # Better: Try N=20, if fail, N=19, etc.
    
    # However, finding N=20 might be very hard. N=10 is easy.
    # Let's estimate: 
    # Size 10: 2^10 = 1024 subsets. Very fast.
    # Size 20: 2^20 = 1M subsets. Fast check, but search space is huge.
    
    # Let's try a range of target sizes.
    # We'll allocate time based on remaining budget.
    
    current_target = 15
    while current_target > 0 and time.time() < deadline:
        # Allocate remaining time proportionally or fixed chunks
        time_remaining = deadline - time.time()
        if time_remaining < 1.0:
            break
            
        # Give more time to larger targets initially? 
        # Let's just try to find a set of size current_target.
        # If found, we can try to extend it? 
        # Or just return it.
        
        # Actually, if we find size 15, we should try to find size 16.
        # So we iterate downwards only if we fail to find the current target.
        
        # Let's try to find size current_target with a timeout of min(5s, time_remaining/2)
        t_out = min(5.0, time_remaining * 0.5)
        
        found = _try_find_set(current_target, timeout=t_out)
        if found:
            best_set = found
            # Try to go higher?
            # If we found size 15, maybe we can find 16?
            # Let's increment target and continue loop?
            # The loop is `while current_target > 0`. 
            # If we found it, we should try a HIGHER target.
            # So we should iterate UPWARDS until we fail?
            
            # Let's restructure:
            # Try N=10, 11, 12...
            # If N=12 fails, stop and return N=11.
            
            pass # Logic adjustment needed
            
    # Restructured Search:
    # Start with a known good small set (e.g., size 8) to ensure we return something.
    # Then try to improve.
    
    # Baseline: Greedy construction often works for small sizes.
    # Let's generate a baseline first.
    def greedy_construct(target_size):
        S = []
        sums_by_card = [set()]
        sums_by_card[0].add(0)
        pool = list(range(1, 257))
        
        for _ in range(target_size):
            found = False
            # Try smallest first
            for x in pool:
                new_sums = fast_add_check(sums_by_card, x)
                if new_sums is not None:
                    S.append(x)
                    sums_by_card = new_sums
                    pool.remove(x)
                    found = True
                    break
            if not found:
                break
        return S

    # Get a baseline
    baseline = greedy_construct(10)
    if baseline:
        best_set = baseline
    else:
        # Fallback to hardcoded small set if greedy fails (unlikely)
        best_set = [1, 2, 4, 8, 16, 32, 64, 128]
        
    # Now try to improve using randomized search
    # Try sizes 11, 12, 13, 14, 15...
    # Stop if we fail to find a size N within a time budget, then return best found.
    
    current_target = len(best_set) + 1
    max_target = 20 # Don't go too high, search space explodes
    
    while current_target <= max_target and time.time() < deadline:
        time_remaining = deadline - time.time()
        if time_remaining < 2.0:
            break
            
        # Allocate time for this attempt
        # Larger targets need more time, but let's be conservative.
        t_out = min(10.0, time_remaining * 0.8)
        
        found = _try_find_set(current_target, timeout=t_out)
        if found:
            best_set = found
            current_target += 1
        else:
            # Failed to find current_target. 
            # Maybe try a few more times with different seeds?
            # Or just break and return best_set.
            # Let's try one more time with a slightly different random seed state.
            found_retry = _try_find_set(current_target, timeout=min(5.0, time_remaining * 0.5))
            if found_retry:
                best_set = found_retry
                current_target += 1
            else:
                break
                
    # Ensure types are correct
    return [int(x) for x in best_set]

# EVOLVE-BLOCK-END
