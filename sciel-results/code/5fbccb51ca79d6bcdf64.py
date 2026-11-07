import time
import random

def _is_valid_bitset(arr):
    """Check if subset sums of different cardinalities are disjoint using bitsets."""
    n = len(arr)
    if n == 0:
        return True
    sums_by_card = [0] * (n + 1)
    sums_by_card[0] = 1
    for x in arr:
        for j in range(n - 1, -1, -1):
            if sums_by_card[j]:
                sums_by_card[j + 1] |= (sums_by_card[j] << x)
    for j in range(n + 1):
        for k in range(j + 1, n + 1):
            if sums_by_card[j] & sums_by_card[k]:
                return False
    return True

def _validate(arr):
    if len(arr) != len(set(arr)):
        return False
    if any(v < 1 or v > 256 for v in arr):
        return False
    return _is_valid_bitset(arr)

def _greedy_construct(max_size=60, deadline=None, rng=None):
    """Greedy construction with multiple orderings."""
    if deadline is None:
        deadline = time.time() + 80
    if rng is None:
        rng = random.Random()
    
    best = [1]
    best_size = 1
    
    candidates = list(range(1, 257))
    
    # Try different starting points and orderings
    start_options = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30, 40, 50]
    
    for start in start_options:
        if time.time() > deadline:
            break
        
        current = [start]
        # Try adding in natural order first
        for candidate in candidates:
            if time.time() > deadline:
                break
            if candidate in current:
                continue
            if len(current) >= max_size:
                break
            
            test_set = current + [candidate]
            if _is_valid_bitset(test_set):
                current = test_set
                if len(current) > best_size:
                    best_size = len(current)
                    best = list(current)
        
        # Try a few random orderings
        for _ in range(3):
            if time.time() > deadline:
                break
            current = [start]
            shuffled = candidates[:]
            rng.shuffle(shuffled)
            for candidate in shuffled:
                if time.time() > deadline:
                    break
                if candidate in current:
                    continue
                if len(current) >= max_size:
                    break
                
                test_set = current + [candidate]
                if _is_valid_bitset(test_set):
                    current = test_set
                    if len(current) > best_size:
                        best_size = len(current)
                        best = list(current)
    
    return best

def _local_improve(arr, deadline, rng=None):
    """Try to improve by swapping or adding elements."""
    if rng is None:
        rng = random.Random()
    current = list(arr)
    current_size = len(current)
    current_set = set(current)
    
    # Try adding elements
    for candidate in range(1, 257):
        if time.time() > deadline:
            break
        if candidate in current_set:
            continue
        test_set = current + [candidate]
        if _is_valid_bitset(test_set):
            current = test_set
            current_set.add(candidate)
            current_size += 1
    
    # Try swaps: remove one element, add another
    for _ in range(50):
        if time.time() > deadline:
            break
        if not current:
            break
        idx = rng.randrange(len(current))
        removed = current[idx]
        base = [x for x in current if x != removed]
        base_set = set(base)
        for candidate in range(1, 257):
            if candidate in base_set:
                continue
            test_set = base + [candidate]
            if _is_valid_bitset(test_set):
                current = test_set
                current_set = set(test_set)
                break
    
    return current

def construct():
    deadline = time.time() + 70  # Safety margin
    
    rng = random.Random()
    
    # Try greedy with different strategies
    best = [1]
    best_size = 1
    
    # Run greedy multiple times with different seeds
    for seed in [0, 1, 42, 123, 999]:
        if time.time() > deadline:
            break
        r = random.Random(seed)
        sol = _greedy_construct(max_size=60, deadline=deadline, rng=r)
        if len(sol) > best_size:
            best_size = len(sol)
            best = list(sol)
    
    # Try local improvement on best solution
    if time.time() < deadline - 5:
        improved = _local_improve(best, deadline, rng)
        if len(improved) > best_size:
            best = improved
            best_size = len(best)
    
    # Final greedy pass with remaining time
    if time.time() < deadline - 2:
        sol = _greedy_construct(max_size=60, deadline=deadline, rng=rng)
        if len(sol) > best_size:
            best = list(sol)
            best_size = len(best)
    
    return list(best)
