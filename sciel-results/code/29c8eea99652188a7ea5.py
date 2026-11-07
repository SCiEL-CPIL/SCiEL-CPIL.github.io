import time
import random

_START = time.time()
_TIME_LIMIT = 82.0  # 90s kill, leave 8s margin

def _subset_sums(nums):
    """Compute all nonempty subset sums as a set."""
    sums = {0}
    for x in nums:
        sums = sums | {s + x for s in sums}
    sums.discard(0)
    return sums

def _is_valid(nums):
    """Check if a set of numbers satisfies the divisibility constraint."""
    if not nums:
        return True
    sums = _subset_sums(nums)
    if not sums:
        return True
    sum_set = set(sums)
    max_sum = max(sum_set)
    for s in sum_set:
        multiple = 2 * s
        while multiple <= max_sum:
            if multiple in sum_set:
                return False
            multiple += s
    return True

def _can_add(nums, new_elem):
    """Check if adding new_elem to nums keeps the set valid.
    Uses incremental approach: compute existing sums, new sums, check all pairs."""
    if not nums:
        return True
    existing_sums = _subset_sums(nums)
    new_sums = {new_elem + s for s in existing_sums} | {new_elem}
    all_sums = existing_sums | new_sums
    sum_set = set(all_sums)
    max_sum = max(sum_set)
    for s in sum_set:
        multiple = 2 * s
        while multiple <= max_sum:
            if multiple in sum_set:
                return False
            multiple += s
    return True

def _greedy(pool, seed, bias='random'):
    """Greedy construction with a given ordering."""
    rng = random.Random(seed)
    remaining = list(pool)
    if bias == 'random':
        rng.shuffle(remaining)
    elif bias == 'desc':
        remaining.sort(reverse=True)
    elif bias == 'asc':
        remaining.sort()
    else:
        rng.shuffle(remaining)
    
    result = []
    for elem in remaining:
        if _can_add(result, elem):
            result.append(elem)
            if len(result) >= 30:
                break
        if time.time() - _START > _TIME_LIMIT:
            break
    return result

def _local_search(current, pool, seed, max_iters=2000):
    """Local search with add, swap, and removal moves."""
    rng = random.Random(seed)
    current = list(current)
    best = list(current)
    best_size = len(current)
    current_set = set(current)
    
    # Build candidate pool
    candidates = [x for x in pool if x not in current_set]
    
    for _ in range(max_iters):
        if time.time() - _START > _TIME_LIMIT:
            break
        
        if not candidates and not current:
            break
        
        move_type = rng.random()
        
        if move_type < 0.4 and len(current) < 30 and candidates:
            # Try adding
            new_elem = rng.choice(candidates)
            trial = current + [new_elem]
            if _is_valid(trial):
                current = trial
                current_set.add(new_elem)
                candidates.remove(new_elem)
                if len(current) > best_size:
                    best = list(current)
                    best_size = len(current)
        elif move_type < 0.8 and current and candidates:
            # Try swap
            if not current:
                continue
            remove_idx = rng.randint(0, len(current) - 1)
            new_elem = rng.choice(candidates)
            trial = current[:remove_idx] + current[remove_idx+1:] + [new_elem]
            if _is_valid(trial):
                current = trial
                current_set.discard(trial[remove_idx] if remove_idx < len(trial) else current[remove_idx])
                # Rebuild set properly
                current_set = set(current)
                candidates = [x for x in pool if x not in current_set]
                if len(current) > best_size:
                    best = list(current)
                    best_size = len(current)
        else:
            # Try removal (downhill move to escape local optima)
            if len(current) > 1:
                remove_idx = rng.randint(0, len(current) - 1)
                removed = current[remove_idx]
                trial = current[:remove_idx] + current[remove_idx+1:]
                current = trial
                current_set.discard(removed)
                candidates.append(removed)
    
    return best

def construct():
    """Main entry point. Returns a list of 1..30 distinct integers in 1..256."""
    pool = list(range(1, 257))
    
    best = []
    best_size = 0
    
    # Phase 1: Multiple greedy attempts
    biases = ['random', 'desc', 'asc']
    for attempt in range(3000):
        if time.time() - _START > _TIME_LIMIT:
            break
        bias = biases[attempt % 3]
        result = _greedy(pool, seed=attempt, bias=bias)
        if len(result) > best_size:
            best = result
            best_size = len(result)
            if best_size >= 30:
                break
    
    # Phase 2: Local search on best result
    if best and time.time() - _START < _TIME_LIMIT - 5:
        improved = _local_search(best, pool, seed=42, max_iters=5000)
        if len(improved) > best_size:
            best = improved
            best_size = len(improved)
    
    # Phase 3: More greedy + local search if time allows
    if best_size < 30 and time.time() - _START < _TIME_LIMIT - 10:
        for attempt in range(2000):
            if time.time() - _START > _TIME_LIMIT - 5:
                break
            seed = 10000 + attempt
            bias = biases[attempt % 3]
            result = _greedy(pool, seed=seed, bias=bias)
            if len(result) >= best_size:
                improved = _local_search(result, pool, seed=seed, max_iters=1000)
                if len(improved) > best_size:
                    best = improved
                    best_size = len(improved)
                    if best_size >= 30:
                        break
    
    # Fallback
    if not best:
        best = [31, 32, 34]
    
    return [int(x) for x in best]
