# EVOLVE-BLOCK-START
import time
import random

def _subset_sums(nums):
    """Return the set of all nonempty subset sums."""
    sums = set()
    n = len(nums)
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += nums[i]
        sums.add(s)
    return sums

def _valid_set(nums):
    """Check divisibility-incomparable subset sums (no two distinct sums divide)."""
    n = len(nums)
    if n == 0:
        return True
    sums = _subset_sums(nums)
    if len(sums) != (1 << n) - 1:
        return False
    sl = sorted(sums)
    m = len(sl)
    # For each sum, check if any other sum is a multiple
    for i in range(m):
        si = sl[i]
        # Check multiples of si
        k = 2
        while k * si <= sl[-1]:
            if k * si in sums:
                return False
            k += 1
    return True

def _is_good_addition_fast(cur_sums, cand):
    """Fast check: can we add cand? Uses bitset-like approach for divisibility."""
    new_sums = {s + cand for s in cur_sums} | {cand}
    if new_sums & cur_sums:
        return False
    combined = cur_sums | new_sums
    max_sum = max(combined)
    # Check divisibility: for each sum s, no other sum should be a multiple
    for s in combined:
        k = 2
        while k * s <= max_sum:
            if k * s in combined:
                return False
            k += 1
    return True

def _greedy_with_order(order, deadline):
    """Greedy: add candidates in given order if valid."""
    current = []
    cur_sums = set()
    for cand in order:
        if time.time() > deadline:
            break
        if len(current) >= 30:
            break
        if _is_good_addition_fast(cur_sums, cand):
            current.append(cand)
            cur_sums = cur_sums | {s + cand for s in cur_sums} | {cand}
    return current

def _local_search(start_set, deadline, max_iter=2000):
    """Try to improve by removing one and adding two, or swapping."""
    current = list(start_set)
    cur_sums = _subset_sums(current)
    best = list(current)
    best_sums = set(cur_sums)
    
    pool = set(range(129, 257)) - set(current)
    pool_list = list(pool)
    random.shuffle(pool_list)
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        # Try: remove one element, add up to 2 new ones
        for i in range(len(current)):
            if time.time() > deadline:
                break
            trial = current[:i] + current[i+1:]
            trial_sums = _subset_sums(trial)
            
            # Try adding one candidate
            for cand in pool_list[:50]:
                if cand in current:
                    continue
                new_sums = {s + cand for s in trial_sums} | {cand}
                if new_sums & trial_sums:
                    continue
                combined = trial_sums | new_sums
                max_s = max(combined)
                valid = True
                for s in combined:
                    k = 2
                    while k * s <= max_s:
                        if k * s in combined:
                            valid = False
                            break
                        k += 1
                    if not valid:
                        break
                if valid and len(trial) + 1 > len(best):
                    best = trial + [cand]
                    best_sums = combined
                    current = list(best)
                    cur_sums = set(combined)
                    pool_list = [c for c in pool_list if c not in current]
                    break
                elif valid and len(trial) + 1 == len(best):
                    # Same size, keep as alternative
                    pass
            
            # Try adding two candidates
            for j in range(min(len(pool_list), 30)):
                c1 = pool_list[j]
                if c1 in current:
                    continue
                for k in range(j+1, min(len(pool_list), 30)):
                    c2 = pool_list[k]
                    if c2 in current:
                        continue
                    new_sums1 = {s + c1 for s in trial_sums} | {c1}
                    if new_sums1 & trial_sums:
                        continue
                    combined1 = trial_sums | new_sums1
                    new_sums2 = {s + c2 for s in combined1} | {c2}
                    if new_sums2 & combined1:
                        continue
                    combined = combined1 | new_sums2
                    max_s = max(combined)
                    valid = True
                    for s in combined:
                        kk = 2
                        while kk * s <= max_s:
                            if kk * s in combined:
                                valid = False
                                break
                            kk += 1
                        if not valid:
                            break
                    if valid and len(trial) + 2 > len(best):
                        best = trial + [c1, c2]
                        best_sums = combined
                        current = list(best)
                        cur_sums = set(combined)
                        pool_list = [c for c in pool_list if c not in current]
                        break
                if len(current) > len(best) - 2:
                    break
    
    return best

def construct():
    start = time.time()
    deadline = start + 75  # safety margin before 90s kill
    
    best = [129, 130, 131]
    
    # Strategy 1: Greedy with different orderings
    pool = list(range(129, 257))
    
    # Descending
    a = _greedy_with_order(sorted(pool, reverse=True), deadline)
    if len(a) > len(best):
        best = list(a)
    
    # Ascending
    if time.time() < deadline:
        b = _greedy_with_order(sorted(pool), deadline)
        if len(b) > len(best):
            best = list(b)
    
    # Random orderings
    if time.time() < deadline:
        for trial in range(20):
            if time.time() > deadline - 10:
                break
            random.shuffle(pool)
            c = _greedy_with_order(pool, deadline)
            if len(c) > len(best):
                best = list(c)
    
    # Strategy 2: Local search from best greedy result
    if time.time() < deadline - 15:
        improved = _local_search(best, deadline - 5)
        if len(improved) > len(best):
            best = improved
    
    # Strategy 3: Try starting from a different base
    if time.time() < deadline - 15:
        # Start with a few large numbers, then fill in
        base_candidates = [256, 255, 254, 253, 252, 251, 250, 249, 248]
        for base_size in [2, 3, 4]:
            if time.time() > deadline - 10:
                break
            base = base_candidates[:base_size]
            cur_sums = _subset_sums(base)
            current = list(base)
            remaining = [c for c in pool if c not in base]
            random.shuffle(remaining)
            for cand in remaining:
                if time.time() > deadline - 5:
                    break
                if len(current) >= 30:
                    break
                if _is_good_addition_fast(cur_sums, cand):
                    current.append(cand)
                    cur_sums = cur_sums | {s + cand for s in cur_sums} | {cand}
            if len(current) > len(best):
                best = list(current)
    
    # Final verification
    if _valid_set(best):
        return list(best)
    
    # Fallback: shrink until valid
    while len(best) > 0 and not _valid_set(best):
        best.pop()
    return list(best)
# EVOLVE-BLOCK-END
