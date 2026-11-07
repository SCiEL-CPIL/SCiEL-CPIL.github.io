import random
import time

def _all_subset_sums(elements):
    n = len(elements)
    sums = set()
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += elements[i]
        sums.add(s)
    return sums

def _check_no_divisibility(sums):
    sl = sorted(sums)
    for i in range(len(sl)):
        for j in range(i + 1, len(sl)):
            if sl[j] % sl[i] == 0:
                return False
    return True

def _is_valid(elements):
    if not elements:
        return True
    if len(elements) != len(set(elements)):
        return False
    for x in elements:
        if not (1 <= x <= 256):
            return False
    sums = _all_subset_sums(elements)
    if len(sums) != (1 << len(elements)) - 1:
        return False
    return _check_no_divisibility(sums)

def _try_grow(elements, candidates, deadline):
    current = list(elements)
    current_sums = _all_subset_sums(current) if current else set()
    remaining = [c for c in candidates if c not in current]
    random.shuffle(remaining)
    
    for x in remaining:
        if time.time() > deadline:
            break
        new_sums = set(current_sums)
        new_sums.add(x)
        for s in current_sums:
            new_sums.add(s + x)
        
        expected = 2 * len(current_sums) + 1
        if len(new_sums) != expected:
            continue
        
        if not _check_no_divisibility(new_sums):
            continue
        
        current.append(x)
        current_sums = new_sums
    
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    best = [31, 32, 34]
    best_len = len(best)
    
    # Multiple candidate pools with different strategies
    pools = [
        list(range(129, 257)),
        list(range(65, 257)),
        list(range(33, 257)),
        list(range(17, 257)),
        list(range(1, 257)),
    ]
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        pool = pools[attempts % len(pools)]
        random.shuffle(pool)
        
        seed_size = random.randint(3, 6)
        if seed_size > len(pool):
            seed_size = len(pool)
        seed = random.sample(pool, seed_size)
        
        grown = _try_grow(seed, pool, deadline)
        
        if len(grown) > best_len:
            best = grown
            best_len = len(grown)
            if best_len >= 10:
                break
    
    if best and _is_valid(best):
        return [int(x) for x in best]
    return [31, 32, 34]
