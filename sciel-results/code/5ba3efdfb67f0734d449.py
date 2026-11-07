import time
import random
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 60s limit
    max_val = 256
    max_size = 60
    
    # sums[k] is a bitset where bit s is set if sum s is achievable with exactly k elements
    # Max possible sum: at most 60 * 256 = 15360
    max_sum = 60 * 256 + 1
    mask = (1 << max_sum) - 1
    
    def check_collision(sums, new_bits_by_card):
        """Check if any new_bits at cardinality k collides with existing sums at cardinality j != k."""
        for k, new_bits in new_bits_by_card.items():
            if new_bits == 0:
                continue
            for j, existing in sums.items():
                if j != k and (new_bits & existing):
                    return False
        return True
    
    def try_add(sums, result, x):
        """Try to add x to the set. Returns (new_sums, success)."""
        new_sums = [0] * (max_size + 1)
        new_sums[0] = sums[0]
        
        new_bits_by_card = {}
        for k in range(1, len(result) + 2):
            shifted = (sums[k-1] << x) & mask
            new_sums[k] = sums[k] | shifted
            new_bits = shifted & ~sums[k]
            if new_bits:
                new_bits_by_card[k] = new_bits
        
        if not check_collision(sums, new_bits_by_card):
            return None, False
        
        return new_sums, True
    
    def greedy_build(candidates, sums_init=None, result_init=None):
        """Greedily build set from candidates."""
        if sums_init is None:
            sums_init = [0] * (max_size + 1)
            sums_init[0] = 1
        if result_init is None:
            result_init = []
        
        sums = sums_init[:]
        result = result_init[:]
        
        for x in candidates:
            if time.time() > deadline:
                break
            if len(result) >= max_size:
                break
            if x in result:
                continue
            
            new_sums, ok = try_add(sums, result, x)
            if ok:
                sums = new_sums
                result.append(x)
        
        return result, sums
    
    def local_search(sums, result, candidates, max_iter=1000):
        """Try to improve by swapping elements."""
        result = result[:]
        sums = sums[:]
        used = set(result)
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            if len(result) >= max_size:
                break
            
            improved = False
            # Try adding a new element
            for x in candidates:
                if x in used:
                    continue
                new_sums, ok = try_add(sums, result, x)
                if ok:
                    sums = new_sums
                    result.append(x)
                    used.add(x)
                    improved = True
                    break
            
            if not improved:
                # Try swapping: remove one, add another
                swapped = False
                for i, old_x in enumerate(result):
                    if time.time() > deadline:
                        break
                    # Temporarily remove old_x
                    # Rebuild sums without old_x is expensive; instead try direct approach
                    # Try adding a candidate that was previously blocked
                    for x in candidates:
                        if x in used:
                            continue
                        # Check if we can replace old_x with x
                        # This requires rebuilding sums, which is expensive
                        # Skip for now; just try direct addition
                        pass
                    break
            
            if not improved and not swapped:
                break
        
        return result, sums
    
    # Multiple strategies
    best_result = []
    best_sums = None
    
    # Strategy 1: Ascending
    r, s = greedy_build(range(1, 257))
    if len(r) > len(best_result):
        best_result, best_sums = r, s
    
    # Strategy 2: Descending
    r, s = greedy_build(range(256, 0, -1))
    if len(r) > len(best_result):
        best_result, best_sums = r, s
    
    # Strategy 3: Large first, then small
    r, s = greedy_build(sorted(range(129, 257), reverse=True) + sorted(range(1, 129)))
    if len(r) > len(best_result):
        best_result, best_sums = r, s
    
    # Strategy 4: Small first, then large
    r, s = greedy_build(sorted(range(1, 129)) + sorted(range(129, 257), reverse=True))
    if len(r) > len(best_result):
        best_result, best_sums = r, s
    
    # Strategy 5: Randomized multiple attempts
    for attempt in range(50):
        if time.time() > deadline:
            break
        random.seed(attempt * 12345 + 67890)
        candidates = list(range(1, 257))
        # Partially shuffle: keep some structure
        for i in range(256):
            if random.random() < 0.3:
                j = random.randint(0, 255)
                candidates[i], candidates[j] = candidates[j], candidates[i]
        r, s = greedy_build(candidates)
        if len(r) > len(best_result):
            best_result, best_sums = r, s
    
    # Strategy 6: Try to extend best_result with remaining elements
    if best_result and best_sums:
        used = set(best_result)
        remaining = [x for x in range(1, 257) if x not in used]
        r, s = greedy_build(remaining, best_sums, best_result)
        if len(r) > len(best_result):
            best_result, best_sums = r, s
    
    # Strategy 7: More randomized attempts with different structures
    for attempt in range(20):
        if time.time() > deadline:
            break
        random.seed(attempt * 99999 + 12345)
        # Mix of large and small
        large = list(range(129, 257))
        small = list(range(1, 129))
        random.shuffle(large)
        random.shuffle(small)
        # Interleave
        candidates = []
        li, si = 0, 0
        while li < len(large) or si < len(small):
            if li < len(large) and (si >= len(small) or random.random() < 0.5):
                candidates.append(large[li])
                li += 1
            else:
                candidates.append(small[si])
                si += 1
        r, s = greedy_build(candidates)
        if len(r) > len(best_result):
            best_result, best_sums = r, s
    
    # Final: try one more extension pass
    if best_result and best_sums and time.time() < deadline - 5:
        used = set(best_result)
        remaining = [x for x in range(1, 257) if x not in used]
        r, s = greedy_build(remaining, best_sums, best_result)
        if len(r) > len(best_result):
            best_result, best_sums = r, s
    
    if not best_result:
        best_result = [1]
    
    # Ensure all are Python ints
    best_result = [int(x) for x in best_result]
    return best_result
