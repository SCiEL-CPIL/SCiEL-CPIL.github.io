import time
import random
import sys

def _compute_cardinality_sums_bitset(elements):
    """Compute bitsets of achievable sums for each cardinality."""
    n = len(elements)
    if n == 0:
        return [1]  # sums[0] = 1 (bit 0 set)
    
    # sums[k] is an integer where bit s is 1 if sum s is achievable with exactly k elements
    sums = [0] * (n + 1)
    sums[0] = 1  # Sum 0 is achievable with 0 elements
    
    for elem in elements:
        # Update in reverse to avoid using elem multiple times
        for k in range(n, 0, -1):
            sums[k] |= (sums[k-1] << elem)
            
    return sums

def _is_valid(elements):
    """Check if the set satisfies the cardinality-separated subset sum constraint using bitsets."""
    sums = _compute_cardinality_sums_bitset(elements)
    n = len(elements)
    # Check for intersection between any two different cardinality classes
    for k in range(n + 1):
        for j in range(k + 1, n + 1):
            if sums[k] & sums[j]:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s budget, 5s safety margin
    
    # Base solution: 16 consecutive integers
    # We know {226..241} is valid.
    # Let's generate a few candidates and pick the best one via local search.
    
    best_size = 0
    best_set = []
    
    # Strategy 1: Start with 16 consecutive large integers
    # M=226, n=16 -> 226..241
    base_set = list(range(226, 242))
    
    # Verify base set
    if _is_valid(base_set):
        best_set = base_set[:]
        best_size = len(base_set)
    
    # Try to grow the set
    # We will use a loop that tries to increase the size by 1
    # by removing 1 element and adding 2 new elements.
    
    current_set = best_set[:]
    used = set(current_set)
    
    # Pre-compute all possible pairs of new elements?
    # Or just iterate.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(current_set)
        
        # Try to find a valid swap: remove a, add x, y
        # Iterate over removal candidates
        for i in range(n):
            if time.time() > deadline:
                break
            a = current_set[i]
            
            # Try adding pairs from 1..256 not in current_set
            # To speed up, we can pre-filter candidates
            # But 256 is small.
            candidates = [x for x in range(1, 257) if x not in used]
            
            # To avoid checking the same pair multiple times in different orders,
            # we can just iterate x < y.
            for x in candidates:
                if time.time() > deadline:
                    break
                for y in candidates:
                    if y <= x:
                        continue
                    if time.time() > deadline:
                        break
                    
                    # Check validity of new set
                    new_set = current_set[:i] + current_set[i+1:] + [x, y]
                    # Optimization: Sort before check? _is_valid doesn't require sort,
                    # but bitset DP is independent of order.
                    if _is_valid(new_set):
                        current_set = new_set
                        current_set.sort()
                        used.discard(a)
                        used.add(x)
                        used.add(y)
                        best_set = current_set[:]
                        best_size = len(current_set)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        # If no growth was possible, try some random restarts or perturbations?
        # If we are stuck at 16, maybe we can try a different base?
        # But if we can't grow 16 to 17, maybe 17 doesn't exist?
        # Or maybe we need a different configuration.
        
        # To explore other configurations, we can try random restarts if time permits.
        # But the "grow" phase is the most direct way to increase size.
        
        # If we haven't improved, let's try a random swap (1-for-1) to change the set
        # and then try to grow again. This helps escape local optima.
        if not improved and time.time() < deadline:
            # Random 1-for-1 swap
            for _ in range(100):
                if time.time() > deadline:
                    break
                i = random.randint(0, n-1)
                a = current_set[i]
                candidates = [x for x in range(1, 257) if x not in used]
                if not candidates:
                    continue
                b = random.choice(candidates)
                new_set = current_set[:i] + current_set[i+1:] + [b]
                if _is_valid(new_set):
                    current_set = new_set
                    current_set.sort()
                    used.discard(a)
                    used.add(b)
                    # We don't update best_size here, as size is same
                    # But we allow the loop to continue to try growing from this new state
                    improved = True
                    break
            # If we did a swap, we continue the while loop to try growing again.
            # If we didn't, we break.
            if not improved:
                break

    return best_set

# EVOLVE-BLOCK-END
