"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _check_divisibility_sorted(sorted_sums):
    """Check if no two distinct values in sorted_sums divide one another."""
    n = len(sorted_sums)
    for i in range(n):
        si = sorted_sums[i]
        for j in range(i + 1, n):
            sj = sorted_sums[j]
            if sj % si == 0:
                return False
    return True

def _valid_set(nums):
    """Check if the set satisfies the divisibility-incomparable subset sum condition."""
    n = len(nums)
    sums_set = set()
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += nums[i]
        if s in sums_set:
            return False
        sums_set.add(s)
    sorted_sums = sorted(sums_set)
    return _check_divisibility_sorted(sorted_sums)

def _valid_set_with_sums(nums, existing_sums=None):
    """Check validity, optionally using precomputed sums for speed.
    Returns (is_valid, all_sums_set) or just is_valid."""
    n = len(nums)
    if existing_sums is not None:
        sums_set = set(existing_sums)
    else:
        sums_set = set()
    
    # Compute new subset sums incrementally
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += nums[i]
        if s in sums_set:
            return False, sums_set
        sums_set.add(s)
    
    sorted_sums = sorted(sums_set)
    if not _check_divisibility_sorted(sorted_sums):
        return False, sums_set
    return True, sums_set

def _try_greedy_extend(current, candidates, deadline, max_size=30):
    """Try to extend current set greedily with candidates."""
    current = list(current)
    used = set(current)
    current_sums = None
    
    for cand in candidates:
        if time.time() > deadline:
            break
        if len(current) >= max_size:
            break
        if cand in used:
            continue
        trial = current + [cand]
        valid, _ = _valid_set_with_sums(trial, current_sums)
        if valid:
            current = trial
            used.add(cand)
            # Recompute sums for next iteration
            current_sums = None
    
    return current

def construct():
    start = time.time()
    deadline = start + 85  # 85 second deadline, 5s safety margin
    
    best = [31, 32, 34]  # Fallback - known valid set of size 3
    rng = random.Random(42)
    
    # Number-theoretic approach: select from [2^k, 2^k + 2^(k-1))
    # For k=8: [256, 384) but capped at 256, so just {256}
    # For k=7: [128, 192) - 64 values
    # For k=6: [64, 96) - 32 values
    # Key: if all elements are in [A, B), then m-element subset sums are in [mA, mB)
    # If mB < (m+1)A, then ranges for different cardinalities don't overlap
    # And within same cardinality, sums are all in same range so divisibility is less likely
    
    # Strategy 1: Pure greedy from largest numbers
    candidates_desc = list(range(256, 127, -1))
    current = []
    for cand in candidates_desc:
        if time.time() > deadline:
            break
        if len(current) >= 30:
            break
        trial = current + [cand]
        valid, _ = _valid_set_with_sums(trial)
        if valid:
            current = trial
    
    if len(current) > len(best):
        best = list(current)
    
    # Strategy 2: Multiple randomized restarts with biased ordering
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        # Biased pool: more weight to larger numbers
        pool = list(range(256, 127, -1))
        rng.shuffle(pool)
        
        current2 = []
        for cand in pool:
            if time.time() > deadline:
                break
            if len(current2) >= 30:
                break
            trial = current2 + [cand]
            valid, _ = _valid_set_with_sums(trial)
            if valid:
                current2 = trial
        
        if len(current2) > len(best):
            best = list(current2)
    
    # Strategy 3: Try adding mid-range numbers to best set
    if len(best) < 30 and time.time() < deadline:
        base = list(best)
        mid_candidates = list(range(128, 63, -1))
        rng.shuffle(mid_candidates)
        
        for cand in mid_candidates:
            if time.time() > deadline:
                break
            if len(base) >= 30:
                break
            if cand in base:
                continue
            trial = base + [cand]
            valid, _ = _valid_set_with_sums(trial)
            if valid:
                base = trial
        
        if len(base) > len(best):
            best = list(base)
    
    # Strategy 4: Local search with swaps and additions
    if time.time() < deadline:
        current_best = list(best)
        
        for _ in range(5000):
            if time.time() > deadline:
                break
            
            action = rng.random()
            
            if action < 0.5 and len(current_best) < 30:
                # Try adding a new element
                if rng.random() < 0.7:
                    new_val = rng.randint(129, 256)
                else:
                    new_val = rng.randint(65, 128)
                
                if new_val not in current_best:
                    trial = current_best + [new_val]
                    valid, _ = _valid_set_with_sums(trial)
                    if valid:
                        current_best = trial
                        if len(current_best) > len(best):
                            best = list(current_best)
            elif action < 0.8:
                # Try replacing an element
                if not current_best:
                    continue
                idx = rng.randint(0, len(current_best) - 1)
                old_val = current_best[idx]
                
                if rng.random() < 0.7:
                    new_val = rng.randint(129, 256)
                else:
                    new_val = rng.randint(65, 128)
                
                if new_val != old_val and new_val not in current_best:
                    trial = current_best.copy()
                    trial[idx] = new_val
                    trial = sorted(trial)
                    valid, _ = _valid_set_with_sums(trial)
                    if valid:
                        current_best = trial
            else:
                # Try removing an element and replacing with two others
                if len(current_best) >= 2:
                    idx = rng.randint(0, len(current_best) - 1)
                    old_val = current_best[idx]
                    
                    new_val1 = rng.randint(129, 256)
                    new_val2 = rng.randint(129, 256)
                    
                    if new_val1 != old_val and new_val2 != old_val and new_val1 != new_val2:
                        if new_val1 not in current_best and new_val2 not in current_best:
                            trial = current_best.copy()
                            del trial[idx]
                            trial.append(new_val1)
                            trial.append(new_val2)
                            trial = sorted(trial)
                            if len(trial) <= 30:
                                valid, _ = _valid_set_with_sums(trial)
                                if valid:
                                    current_best = trial
                                    if len(current_best) > len(best):
                                        best = list(current_best)
    
    # Strategy 5: Systematic search in narrow intervals
    if time.time() < deadline:
        # Try selecting from [128, 192) - 64 values
        # Then try to extend with [64, 128)
        for interval_start in [192, 160, 128, 96, 64]:
            if time.time() > deadline:
                break
            if len(best) >= 30:
                break
            
            interval = list(range(min(interval_start + 64, 257), max(interval_start, 1) - 1, -1))
            rng.shuffle(interval)
            
            current3 = []
            for cand in interval:
                if time.time() > deadline:
                    break
                if len(current3) >= 30:
                    break
                trial = current3 + [cand]
                valid, _ = _valid_set_with_sums(trial)
                if valid:
                    current3 = trial
            
            # Try to extend with smaller numbers
            if len(current3) < 30:
                smaller = list(range(interval_start, max(interval_start - 64, 1) - 1, -1))
                rng.shuffle(smaller)
                for cand in smaller:
                    if time.time() > deadline:
                        break
                    if len(current3) >= 30:
                        break
                    if cand in current3:
                        continue
                    trial = current3 + [cand]
                    valid, _ = _valid_set_with_sums(trial)
                    if valid:
                        current3 = trial
            
            if len(current3) > len(best):
                best = list(current3)
    
    # Final verification
    if _valid_set(best):
        return list(best)
    else:
        return [31, 32, 34]
# EVOLVE-BLOCK-END
