def construct():
    import time
    import random
    from functools import lru_cache
    
    start_time = time.time()
    deadline = 80  # safety margin before 90s kill
    
    def check_validity(arr):
        """Check if arr satisfies the cardinality-separated subset sum constraint."""
        n = len(arr)
        # sums_by_card[k] = set of all subset sums using exactly k elements
        sums_by_card = [set() for _ in range(n + 1)]
        sums_by_card[0].add(0)
        
        for x in arr:
            # Process in reverse to avoid using x multiple times
            for k in range(n, 0, -1):
                if sums_by_card[k-1]:
                    for s in sums_by_card[k-1]:
                        sums_by_card[k].add(s + x)
        
        # Check pairwise disjointness across different cardinalities
        for i in range(n + 1):
            for j in range(i + 1, n + 1):
                if sums_by_card[i] & sums_by_card[j]:
                    return False
        return True
    
    def check_validity_bitset(arr):
        """Faster check using bitsets for subset sums by cardinality."""
        n = len(arr)
        max_sum = sum(arr)
        # sums_by_card[k] is a bitset where bit s is 1 if sum s is achievable with k elements
        sums_by_card = [0] * (n + 1)
        sums_by_card[0] = 1  # sum 0 achievable with 0 elements
        
        for x in arr:
            for k in range(n, 0, -1):
                # Shift the bitset for k-1 elements by x
                shifted = (sums_by_card[k-1] << x)
                sums_by_card[k] |= shifted
        
        # Check pairwise disjointness
        for i in range(n + 1):
            for j in range(i + 1, n + 1):
                if sums_by_card[i] & sums_by_card[j]:
                    return False
        return True
    
    # Start with the known valid set of 9
    best = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    best_size = len(best)
    
    # Try to greedily add elements
    # Strategy: try adding elements from 1 to 256, check validity
    
    def try_greedy_addition(base_set, candidates):
        """Try to add elements from candidates to base_set while maintaining validity."""
        current = list(base_set)
        for c in candidates:
            if c in current:
                continue
            trial = current + [c]
            if check_validity_bitset(trial):
                current = trial
        return current
    
    # Greedy approach: start empty, add elements in various orders
    all_candidates = list(range(1, 257))
    
    # Try several different orderings
    random.seed(42)
    
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        if attempt == 0:
            order = list(range(1, 257))
        elif attempt == 1:
            order = list(range(256, 0, -1))
        elif attempt == 2:
            order = sorted(all_candidates, key=lambda x: random.random())
        elif attempt == 3:
            # Start with large numbers
            order = [x for x in all_candidates if x > 128] + [x for x in all_candidates if x <= 128]
        elif attempt == 4:
            # Start with small numbers but skip powers of 2
            order = [x for x in all_candidates if x not in [1,2,4,8,16,32,64,128,256]]
        else:
            order = sorted(all_candidates, key=lambda x: random.random())
        
        current = []
        for c in order:
            if time.time() > deadline:
                break
            trial = current + [c]
            if check_validity_bitset(trial):
                current = trial
        
        if len(current) > best_size:
            best = current
            best_size = len(current)
            # Try to improve further from this base
            remaining = [x for x in all_candidates if x not in best]
            improved = try_greedy_addition(best, remaining)
            if len(improved) > best_size:
                best = improved
                best_size = len(best)
    
    # Final attempt: try to extend the best set
    if time.time() < deadline - 5:
        remaining = [x for x in all_candidates if x not in best]
        best = try_greedy_addition(best, remaining)
        best_size = len(best)
    
    return list(best)
