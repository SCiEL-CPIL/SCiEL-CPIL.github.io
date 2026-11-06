import time
import numpy as np
import random

# EVOLVE-BLOCK-START
def _is_dissociated_subset(S, recips=None):
    """
    Checks if a set of integers S is dissociated in the reciprocals.
    i.e., all subset sums of {1/s for s in S} are distinct.
    Uses a set-based collision check for O(2^n) time (faster than sort for small n).
    """
    if not S:
        return True

    if recips is None:
        reciprocals = [1.0 / s for s in S]
    else:
        reciprocals = [recips[s] for s in S]

    # Generate all subset sums iteratively
    sums = {0.0}
    for r in reciprocals:
        # For each existing sum, add r. If r already exists as a sum, collision.
        # More efficiently: new_sums = {s + r for s in sums}; check overlap with sums.
        new_sums = set()
        for s in sums:
            ns = s + r
            if ns in sums or ns in new_sums:
                return False
            new_sums.add(ns)
        sums |= new_sums

    return True


def _local_search_swap(S, recips, deadline, max_iters=5000):
    """
    Try to grow set S by swapping elements: remove one, try adding another.
    Also try pure additions first. Returns best set found.
    """
    S = list(S)
    best = S[:]
    used = set(S)
    universe = list(range(1, 23))

    for it in range(max_iters):
        if time.time() > deadline:
            break

        improved = False

        # Phase 1: Try adding any unused element
        for x in universe:
            if x in used:
                continue
            new_set = S + [x]
            if _is_dissociated_subset(new_set, recips):
                S = new_set
                used.add(x)
                improved = True
                if len(S) > len(best):
                    best = S[:]
                break  # restart scan after improvement

        if improved:
            continue

        # Phase 2: Try swap moves (remove one, add one)
        for i, x in enumerate(S):
            if time.time() > deadline:
                break
            S_without = S[:i] + S[i+1:]
            used.discard(x)
            for y in universe:
                if y in used:
                    continue
                new_set = S_without + [y]
                if _is_dissociated_subset(new_set, recips):
                    S = new_set
                    used.discard(x)
                    used.add(y)
                    improved = True
                    if len(S) > len(best):
                        best = S[:]
                    break
            if improved:
                break
            # restore x
            S = S_without + [x]
            used.add(x)

        if not improved:
            # No swap or add possible; try random restarts
            break

    return best

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s safety margin for 90s limit
    
    # Precompute reciprocals
    recips = {i: 1.0/i for i in range(1, 23)}
    universe = list(range(1, 23))
    
    # Known good seed (size 9)
    best_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    best_size = len(best_set)
    
    # Initialize random seeds for reproducibility of the search path (optional but good for debugging)
    random.seed(42)
    np.random.seed(42)
    
    # Strategy 1: Greedy with various orders
    # This is fast and often finds good local optima.
    
    orders = [
        list(range(1, 23)),
        list(range(22, 0, -1)),
    ]
    
    # Add some random permutations
    # We want to spend time on random sampling if greedy doesn't find size 12+
    # So limit greedy random orders to a reasonable number.
    max_greedy_randoms = 100
    
    for _ in range(max_greedy_randoms):
        if time.time() > deadline - 20:  # Leave 20s for random sampling
            break
        cand_order = list(range(1, 23))
        random.shuffle(cand_order)
        orders.append(cand_order)
    
    for order in orders:
        if time.time() > deadline - 20:
            break
            
        current_set = []
        used = set()
        
        for x in order:
            if x in used:
                continue
            # Try adding x
            new_set = current_set + [x]
            if _is_dissociated_subset(new_set, recips):
                current_set = new_set
                used.add(x)
                
        if len(current_set) > best_size:
            best_set = current_set
            best_size = len(current_set)
            
        # Early exit if we reached a very high number
        if best_size >= 15:
            break

    # Strategy 2: Random sampling of specific sizes
    # Try to find sets of size > best_size by random sampling.
    # We iterate from a high target down to best_size + 1.
    
    # If we already found size 15+, we can return early?
    # The task says "Maximize cardinality". 22 is the max possible.
    # But finding 15+ is very hard.
    # Let's try to push for 12, 13, 14.
    
    # Determine max target to try.
    # If best_size is 9, try 12, 11, 10.
    # If best_size is 10, try 12, 11.
    # If best_size is 11, try 12.
    # If best_size is 12, try 13? 14?
    
    # Let's try up to 14.
    max_target = min(22, max(best_size + 1, 14))
    
    # We should try larger sizes first.
    for target_size in range(max_target, best_size, -1):
        if time.time() > deadline - 5:
            break
        
        # Number of attempts to try
        # Larger sizes are rarer, so we might need more attempts, but each check is slower.
        # Check for size K takes O(2^K).
        # Size 12: 4096 sums. Size 13: 8192. Size 14: 16384.
        # 10000 checks of size 14 might take 10000 * 16k * ops.
        # Let's estimate time per check.
        # 16k floats, sort, compare. ~1ms?
        # 10000 checks -> 10s.
        # So we can afford ~5000-10000 checks for size 14 in the remaining time.
        
        attempts = 0
        # Adjust attempts based on size
        if target_size >= 14:
            max_attempts = 2000
        elif target_size >= 12:
            max_attempts = 10000
        else:
            max_attempts = 50000
            
        while attempts < max_attempts:
            if time.time() > deadline - 2:
                break
            
            # Random subset of target_size
            candidate = random.sample(universe, target_size)
            if _is_dissociated_subset(candidate, recips):
                best_set = candidate
                best_size = target_size
                # Found one!
                # Should we continue looking for a larger one?
                # The loop structure `for target_size in range(max_target, best_size, -1)`
                # will move to the next smaller size if we don't break the outer loop.
                # But if we found size 14, best_size becomes 14.
                # The range was fixed at the start of the loop iteration? No, `range` is evaluated once.
                # So if we started with range(14, 9, -1) -> [14, 13, 12, 11, 10].
                # If we find 14, best_size=14.
                # Next iteration is 13. We will try to find 13?
                # But 14 is better. We should stop trying smaller sizes.
                # So we should break the outer loop if we find a valid set for the current target?
                # Actually, if we find a valid set for target_size, that is the max possible
                # for this specific random search strategy (since we are going down).
                # However, we might have skipped a larger size in a previous iteration
                # because we didn't find one in the limited attempts.
                # But if we found 14, we are done (unless we can find 15, 16...).
                # Since we are iterating down, if we find 14, we should check if we can find 15?
                # The loop doesn't go up.
                # To be robust, if we find a valid set of size K, we should ideally
                # try to find K+1, K+2...
                # But that requires restarting the loop or a nested structure.
                # Given the rarity, finding 14 is a huge win.
                # Let's break out of the inner loop and the outer loop?
                # Or just break the inner loop and let the outer loop continue?
                # If the outer loop continues, it will try 13, 12... which are worse.
                # So we should break the outer loop if we found a set of size >= 12?
                # Actually, if we found 10, we should still try 11, 12.
                # The loop goes 14, 13, 12, 11, 10.
                # If we find 10, we break inner. Outer moves to 9? No, range stops at best_size+1.
                # If best_size was 9, range is 14..10.
                # If we find 10, best_size=10.
                # The range object is already created: 14, 13, 12, 11, 10.
                # The loop iterator will continue to the next item in the range.
                # But wait, `range` is fixed.
                # So if we find 10, the loop will finish (no more items).
                # If we find 11, the loop will try 10 next.
                # We don't want to try 10 if we found 11.
                # So we should break the outer loop if we found a valid set?
                # No, because we might have missed 12, 13, 14 in previous iterations
                # due to low probability.
                # But if we found 11, we already tried 14, 13, 12 and failed.
                # So trying 10 is pointless.
                # So yes, break the outer loop if we found a valid set.
                
                # However, what if we found 10, but 11 is possible?
                # We already tried 11 in the previous iteration (if range was 14..10).
                # So if we are at 10, we have already failed to find 11, 12, 13, 14.
                # So breaking is correct.
                
                break
            attempts += 1
        
        # If we found a valid set in this target_size, break the outer loop
        # because we are iterating downwards and larger sizes have already been attempted.
        # Wait, did we attempt larger sizes?
        # Yes, the loop goes from max_target down to best_size+1.
        # So if we are at target_size T, we have already tried T+1, T+2... max_target.
        # If we found a valid set at T, it means T+1..max_target failed.
        # So T is the best we found. We can stop.
        if len(best_set) == target_size:
            break

    # Fallback
    if not best_set:
        return [1, 2, 3, 5, 7, 11, 13, 17, 19]
        
    return best_set
# EVOLVE-BLOCK-END
