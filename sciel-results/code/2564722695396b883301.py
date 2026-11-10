import time
import random

def construct():
    deadline = time.time() + 85.0  # safety margin under 90s
    N = 256
    MAX_SIZE = 60

    def compute_sums(vals):
        """Compute sums[k] bitset for each cardinality k."""
        n = len(vals)
        sums = [0] * (n + 1)
        sums[0] = 1
        for v in vals:
            for k in range(n - 1, -1, -1):
                if sums[k]:
                    sums[k + 1] |= (sums[k] << v)
        return sums

    def count_collisions(sums):
        """Count total cross-cardinality collisions."""
        n = len(sums) - 1
        total = 0
        for i in range(n + 1):
            for j in range(i + 1, n + 1):
                total += bin(sums[i] & sums[j]).count('1')
        return total

    def is_valid(sums):
        """Check if there are no cross-cardinality collisions."""
        n = len(sums) - 1
        for i in range(n + 1):
            for j in range(i + 1, n + 1):
                if sums[i] & sums[j]:
                    return False
        return True

    def try_greedy_descending(pool, max_size):
        """Try to build a valid set by greedily adding elements from largest to smallest."""
        result = []
        for v in sorted(pool, reverse=True):
            if len(result) >= max_size:
                break
            candidate = result + [v]
            sums = compute_sums(candidate)
            if is_valid(sums):
                result = candidate
        return result

    def try_global_removal(pool, max_size, start_size):
        """Start with a random set and iteratively remove elements causing most collisions."""
        best = []
        best_collisions = float('inf')
        
        for attempt in range(10):
            if time.time() > deadline - 5:
                break
            
            # Start with a random subset
            random.seed(attempt * 12345 + 67890)
            initial = random.sample(pool, min(start_size, len(pool)))
            current = list(initial)
            
            # Iteratively remove elements
            for _ in range(len(current)):
                if time.time() > deadline - 5:
                    break
                
                sums = compute_sums(current)
                if is_valid(sums):
                    break
                
                n = len(current)
                current_collisions = count_collisions(sums)
                
                # For each element, compute collision reduction if removed
                best_removal = -1
                best_new_collisions = current_collisions
                
                for idx in range(n):
                    if time.time() > deadline - 5:
                        break
                    
                    candidate = current[:idx] + current[idx+1:]
                    cand_sums = compute_sums(candidate)
                    cand_collisions = count_collisions(cand_sums)
                    
                    if cand_collisions < best_new_collisions:
                        best_new_collisions = cand_collisions
                        best_removal = idx
                
                if best_removal == -1:
                    # No improvement possible, break
                    break
                
                current = current[:best_removal] + current[best_removal+1:]
            
            # Check final result
            if current:
                final_sums = compute_sums(current)
                if is_valid(final_sums):
                    if len(current) > len(best):
                        best = list(current)
                        best_collisions = 0
                    if len(best) >= max_size:
                        break
                elif best_collisions > 0:
                    # Track best even if not fully valid? No, we need valid
                    pass
        
        return best

    # Strategy 1: Try descending greedy (large values first tends to separate sums)
    pool = list(range(1, N + 1))
    best = try_greedy_descending(pool, MAX_SIZE)
    
    if time.time() > deadline:
        return best
    
    # Strategy 2: Try global removal approach with different starting sizes
    for start_size in [40, 35, 30, 25, 20]:
        if time.time() > deadline - 10:
            break
        result = try_global_removal(pool, MAX_SIZE, start_size)
        if len(result) > len(best):
            best = result
        if len(best) >= MAX_SIZE:
            break
    
    if time.time() > deadline:
        return best
    
    # Strategy 3: Try with subsets of the pool
    for start in range(1, 30):
        if time.time() > deadline - 5:
            break
        sub_pool = list(range(start, N + 1))
        desc = try_greedy_descending(sub_pool, MAX_SIZE)
        if len(desc) > len(best):
            best = desc
    
    # Ensure we return a list of Python ints
    best = [int(x) for x in best]
    return best
