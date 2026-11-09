import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 55.0  # safety margin before 60s kill

    def elapsed():
        return time.time() - start_time

    def is_done():
        return elapsed() > DEADLINE

    def try_add(A, candidate, sums_by_k, sum_to_card):
        """
        Try to add candidate to A.
        Returns (success, new_sums_by_k, new_sum_to_card).
        sums_by_k[k] is a set of sums for cardinality k.
        sum_to_card[s] is the cardinality k such that s in sums_by_k[k].
        """
        if candidate in A:
            return False, None, None
        
        n = len(A)
        
        # Check for conflicts
        # For each k, for each sum s in sums_by_k[k], new_sum = s + candidate.
        # This new_sum will belong to cardinality k+1.
        # If new_sum is already in sum_to_card, it must have cardinality k+1.
        # Otherwise, conflict.
        
        # We need to be careful: we are adding a new element, so we generate new sums.
        # We don't modify sums_by_k in place yet, we just check.
        
        # Optimization: Pre-compute the set of new sums for each k?
        # Or just iterate.
        
        for k in range(n + 1):
            if not sums_by_k[k]:
                continue
            for s in sums_by_k[k]:
                ns = s + candidate
                # ns will be in cardinality k+1
                if ns in sum_to_card:
                    if sum_to_card[ns] != k + 1:
                        return False, None, None
                # If ns is not in sum_to_card, it's fine (it will be added).
                # If ns is in sum_to_card and is k+1, it's fine (duplicate sum within same cardinality is allowed).
        
        # If no conflict, create new structures
        new_sums_by_k = [set(s) for s in sums_by_k]
        new_sums_by_k.append(set()) # Index n+1
        
        # Update sums_by_k
        for k in range(n, -1, -1):
            if new_sums_by_k[k]:
                for s in new_sums_by_k[k]:
                    ns = s + candidate
                    new_sums_by_k[k+1].add(ns)
        
        # Update sum_to_card
        new_sum_to_card = dict(sum_to_card)
        for k in range(n + 1):
            # The sums in new_sums_by_k[k+1] that are NEW need to be added to sum_to_card.
            # Actually, we can just iterate all sums in new_sums_by_k[k+1] and set them.
            # But we must be careful not to overwrite existing valid entries?
            # No, if it was valid before, sum_to_card[s] was k.
            # Now s is also in k+1? No, that would be a conflict we already checked.
            # So any sum in new_sums_by_k[k+1] that wasn't there before is new.
            # Any sum that was there before must have cardinality k+1 (from previous elements? No, previous elements only had up to cardinality n).
            # Wait, sums_by_k had indices 0..n.
            # new_sums_by_k has indices 0..n+1.
            # The new sums are in new_sums_by_k[k+1] for k=0..n.
            # Specifically, the new sums are {s + candidate | s in old_sums_by_k[k]}.
            # These sums now belong to cardinality k+1.
            
            # We can just update sum_to_card for all sums in new_sums_by_k[k+1].
            # Is it possible that a sum in new_sums_by_k[k+1] was already in sum_to_card with a DIFFERENT cardinality?
            # We checked for conflicts. If sum_to_card[ns] existed and was not k+1, we returned False.
            # So if we are here, any existing sum_to_card[ns] must be k+1.
            # So setting it to k+1 is safe.
            
            for s in new_sums_by_k[k+1]:
                new_sum_to_card[s] = k + 1
        
        return True, new_sums_by_k, new_sum_to_card

    def greedy_build(order):
        A = []
        sums_by_k = [set() for _ in range(1)]
        sums_by_k[0].add(0)
        sum_to_card = {0: 0}
        
        for candidate in order:
            if is_done():
                break
            if len(A) >= 60:
                break
            success, new_sums, new_card = try_add(A, candidate, sums_by_k, sum_to_card)
            if success:
                A.append(candidate)
                sums_by_k = new_sums
                sum_to_card = new_card
                
        return A, sums_by_k, sum_to_card

    def local_search(A, sums_by_k, sum_to_card):
        A = list(A)
        sums_by_k = [set(s) for s in sums_by_k]
        sum_to_card = dict(sum_to_card)
        improved = True
        
        while improved and not is_done():
            improved = False
            n = len(A)
            
            # 1. Try adding any unused element
            if n < 60:
                for c in range(1, 257):
                    if is_done():
                        break
                    if c in A:
                        continue
                    success, new_sums, new_card = try_add(A, c, sums_by_k, sum_to_card)
                    if success:
                        A.append(c)
                        sums_by_k = new_sums
                        sum_to_card = new_card
                        improved = True
                        if is_done():
                            break
            
            if is_done():
                break
            
            # 2. Try swapping one element with another
            # Remove i, add j.
            # This is expensive to check directly.
            # We can try to remove i, then try to add j to the reduced set.
            # But removing requires recomputing sums or having a data structure that supports removal.
            # Recomputing sums for N=10..20 is fast (2^N subsets).
            # For N=20, 2^20 = 1M operations. Might be too slow for many swaps.
            # Let's limit swaps to small N or just rely on greedy.
            
            # Alternative: Random restarts with different orders might be more effective than swaps.
            
        return A, sums_by_k, sum_to_card

    best_A = []
    best_sums = None
    best_card = None
    
    # Strategy 1: Ascending
    A1, sums1, card1 = greedy_build(range(1, 257))
    if len(A1) > len(best_A):
        best_A = A1
        best_sums = sums1
        best_card = card1
    if best_A:
        best_A, best_sums, best_card = local_search(best_A, best_sums, best_card)

    # Strategy 2: Descending
    if not is_done():
        A2, sums2, card2 = greedy_build(range(256, 0, -1))
        if len(A2) > len(best_A):
            best_A = A2
            best_sums = sums2
            best_card = card2
        if best_A:
            best_A, best_sums, best_card = local_search(best_A, best_sums, best_card)

    # Strategy 3: Random restarts
    random.seed(42)
    for _ in range(50):
        if is_done():
            break
        order = list(range(1, 257))
        random.shuffle(order)
        Ar, sumsr, cardr = greedy_build(order)
        if len(Ar) > len(best_A):
            best_A = Ar
            best_sums = sumsr
            best_card = cardr
            if best_A:
                best_A, best_sums, best_card = local_search(best_A, best_sums, best_card)

    # Strategy 4: Structured sets
    # Powers of 2
    pot2 = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    if len(pot2) > len(best_A):
        best_A = pot2
        # Recompute sums for best_A
        n = len(best_A)
        best_sums = [set() for _ in range(n + 1)]
        best_sums[0].add(0)
        best_card = {0: 0}
        for a in best_A:
            for k in range(n - 1, -1, -1):
                if best_sums[k]:
                    for s in best_sums[k]:
                        ns = s + a
                        best_sums[k+1].add(ns)
                        best_card[ns] = k + 1
        best_A, best_sums, best_card = local_search(best_A, best_sums, best_card)
        
    if not best_A:
        best_A = [1]
        
    return [int(x) for x in best_A]
