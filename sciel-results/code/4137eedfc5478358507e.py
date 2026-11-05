import time
import random

_START = time.time()
_DEADLINE = 85.0 # Safety margin for 90s limit

def _time_ok():
    return (time.time() - _START) < _DEADLINE

def construct():
    # LCM of 1 to 22
    LCM = 232792560
    # weights[i] corresponds to element i+1 (1-indexed)
    # index 0 -> 1, index 1 -> 2, ..., index 21 -> 22
    weights = [LCM // k for k in range(1, 23)]
    n = 22
    
    def get_subset_sums(indices):
        """Compute the set of all subset sums for the given indices."""
        sums = {0}
        for i in indices:
            w = weights[i]
            # Optimization: if we just need to check validity, we can do it incrementally.
            # For full set, we compute all.
            new_sums = set()
            for s in sums:
                ns = s + w
                if ns in sums:
                    # Collision detected, but for this helper we assume valid or just return what we have?
                    # Better to use incremental check in the main logic.
                    pass 
                new_sums.add(ns)
            sums.update(new_sums)
        return sums

    def is_dissociated_incremental(current_sums, w):
        """
        Check if adding weight w to a set with subset sums 'current_sums' 
        maintains dissociation.
        Returns the new set of subset sums if valid, None otherwise.
        """
        new_sums = set()
        for s in current_sums:
            ns = s + w
            if ns in current_sums or ns in new_sums:
                return None
            new_sums.add(ns)
        new_sums.update(current_sums)
        return new_sums

    def greedy(order):
        """
        Greedily construct a dissociated set following the given order of indices.
        """
        chosen = []
        sums = {0}
        for i in order:
            new_sums = is_dissociated_incremental(sums, weights[i])
            if new_sums is not None:
                chosen.append(i)
                sums = new_sums
        return chosen, sums

    # 1. Generate multiple candidate sets using different orderings
    best_set = []
    best_sums = {0}
    best_size = 0
    
    # Deterministic orderings
    orders = [
        list(range(n)),                      # Ascending index (1 to 22)
        list(range(n-1, -1, -1)),            # Descending index (22 to 1)
        sorted(range(n), key=lambda i: weights[i], reverse=True), # Descending weight
        sorted(range(n), key=lambda i: weights[i]),               # Ascending weight
    ]
    
    # Random orderings
    rng = random.Random(42)
    for _ in range(50):
        perm = list(range(n))
        rng.shuffle(perm)
        orders.append(perm)
    
    for order in orders:
        if not _time_ok():
            break
        cand_set, cand_sums = greedy(order)
        if len(cand_set) > best_size:
            best_size = len(cand_set)
            best_set = cand_set
            best_sums = cand_sums
            
    # 2. Local Search: Try to add any remaining element
    # Repeat until no improvement
    improved = True
    while improved and _time_ok():
        improved = False
        remaining = [i for i in range(n) if i not in best_set]
        # Try adding each remaining element
        # To make it deterministic and potentially find better structures, 
        # we can try them in order of weight or index
        for i in remaining:
            if not _time_ok():
                break
            new_sums = is_dissociated_incremental(best_sums, weights[i])
            if new_sums is not None:
                best_set.append(i)
                best_sums = new_sums
                best_size += 1
                improved = True
                # Restart the loop to check new possibilities with the updated set
                # This is a simple hill-climbing step
                break 

    # 3. Backtracking to see if we can beat the current best_size
    # We search for a set of size best_size + 1
    # If found, update and search for best_size + 2, etc.
    
    # To make backtracking efficient, we use the current best_set as a hint?
    # Or just run a full search with strong pruning.
    # Pruning: if len(current) + (n - start) <= best_size: return
    
    # We'll use a recursive search that tries to build a set of size > best_size.
    # We'll pass the current best_size as the threshold to beat.
    
    def backtrack(start, current, current_sums, target_size):
        """
        Recursive search to find a dissociated set of size target_size.
        Returns the set if found, None otherwise.
        """
        if not _time_ok():
            return None
        
        if len(current) == target_size:
            return current[:]
        
        # Pruning: not enough elements left to reach target size
        if len(current) + (n - start) < target_size:
            return None
            
        for i in range(start, n):
            if not _time_ok():
                return None
            w = weights[i]
            new_sums = is_dissociated_incremental(current_sums, w)
            if new_sums is not None:
                current.append(i)
                result = backtrack(i + 1, current, new_sums, target_size)
                if result is not None:
                    return result
                current.pop()
                
        return None

    # Try to find a set of size best_size + 1, then +2, etc.
    # We limit the search depth/time for each target size.
    target = best_size + 1
    while target <= n and _time_ok():
        # To speed up, we can sort the candidates. 
        # Elements with larger weights might be "safer" to pick?
        # Let's try a few orderings for the backtracking start.
        # Actually, the `start` parameter in backtrack enforces increasing index order.
        # This limits the search space significantly (subsets are found in lexicographic order of indices).
        # This is a standard way to avoid duplicates.
        
        found = backtrack(0, [], {0}, target)
        if found is not None:
            best_set = found
            best_size = target
            # Recompute sums for the new best set
            best_sums = get_subset_sums(best_set)
            target += 1
        else:
            # If we can't find a set of size `target`, 
            # it might be because of the specific search order or time limit.
            # However, if we exhaust the search space (or run out of time), 
            # we should probably stop assuming `target` is impossible.
            # Given the small N=22, if we run out of time, we just return what we have.
            # But if we finish the search quickly, it's a proof.
            # We don't easily know if we finished or timed out.
            # Let's just break if we timed out or if we fail to find.
            # A more robust approach: if we fail to find target, try target again with different heuristic?
            # For now, break.
            break

    # Final verification
    if not best_set:
        return [1] # Fallback: {1} is always dissociated
        
    # Double check validity
    final_sums = get_subset_sums(best_set)
    if len(final_sums) != (1 << len(best_set)):
        # Something went wrong, fall back to a known valid small set
        # Powers of 2 indices (0,1,2,3 -> k=1,2,4,8) are dissociated
        # weights: LCM, LCM/2, LCM/4, LCM/8
        # Sums are distinct because they are binary representations scaled.
        best_set = [0, 1, 2, 3]
        
    return [i + 1 for i in best_set]
