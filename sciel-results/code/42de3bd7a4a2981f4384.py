"""Rewritten program with optimized subset-sum divisibility search."""
# EVOLVE-BLOCK-START
import time
import random

def _subset_sums(nums):
    """Generate all non-empty subset sums as a set."""
    sums = set()
    for x in nums:
        new_sums = set()
        for s in sums:
            new_sums.add(s + x)
        new_sums.add(x)
        sums |= new_sums
    return sums

def _check_validity(nums):
    """Check divisibility-incomparable subset sum constraint."""
    n = len(nums)
    if n == 0:
        return True
    sums = _subset_sums(nums)
    if len(sums) != (1 << n) - 1:
        return False
    sums_list = sorted(sums)
    sums_set = set(sums_list)
    max_sum = sums_list[-1]
    for s in sums_list:
        for k in range(2, max_sum // s + 1):
            if k * s in sums_set:
                return False
    return True

def _check_validity_fast(nums):
    """Faster validity check using the interval property."""
    n = len(nums)
    if n == 0:
        return True
    sums = _subset_sums(nums)
    if len(sums) != (1 << n) - 1:
        return False
    sums_list = sorted(sums)
    min_s = sums_list[0]
    max_s = sums_list[-1]
    # If max < 2*min, no divisibility is possible
    if max_s < 2 * min_s:
        return True
    # Otherwise, need full check
    sums_set = set(sums_list)
    for s in sums_list:
        for k in range(2, max_s // s + 1):
            if k * s in sums_set:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin before 60s limit
    
    # Verified valid set of size 7
    # {100, 101, 102, 104, 106, 108, 110}
    # Subset sums range from 100 to 731
    # 731 < 2*100=200? No, 731 > 200. Need to verify.
    
    # Actually, let me use a different approach.
    # For n elements, we need all 2^n - 1 sums distinct AND no divisibility.
    # If all sums are in (L, 2L), no divisibility.
    # With n elements each in [a, b], sums are in [a, n*b].
    # Need n*b < 2a, i.e., b < 2a/n.
    # For n=7: b < 2a/7. If a=100, b < 28.6. But we need distinct ints in [100, 28]? Impossible.
    
    # Alternative: use numbers where subset sums avoid divisibility.
    # Let me try powers of 2 scaled: {128, 129, 131, 134, 140, 152, 176}
    # Or better: let me just search.
    
    best = [128, 129, 131, 134, 140, 152, 176]
    
    # Verify our initial best
    if not _check_validity(best):
        # Fallback to known valid small set
        best = [31, 32, 34]
    
    # Try to find larger sets using greedy + local search
    # Key insight: use numbers in a range where subset sums are "spread out"
    # to avoid divisibility. Numbers that are close together and large work well.
    
    # Strategy: greedy construction with careful candidate selection
    # For each candidate, check if adding it maintains validity
    
    all_candidates = list(range(1, 257))
    
    # Try multiple strategies
    strategies = [
        # Start from high range
        (list(range(200, 257)), list(range(150, 200)), list(range(100, 150))),
        # Start from mid-high
        (list(range(150, 257)), list(range(100, 150)), list(range(50, 100))),
        # Start from mid
        (list(range(100, 257)), list(range(50, 100)), list(range(25, 50))),
    ]
    
    for primary, secondary, tertiary in strategies:
        if time.time() > deadline - 5:
            break
        
        for trial in range(50):
            if time.time() > deadline - 5:
                break
            
            rng = random.Random(trial * 104729 + len(primary))
            
            # Shuffle candidates
            pool = primary[:] + secondary[:] + tertiary[:]
            rng.shuffle(pool)
            
            current = []
            for num in pool:
                if len(current) >= 10:  # Cap at 10 for now
                    break
                trial_set = current + [num]
                if _check_validity_fast(trial_set):
                    current = trial_set
            
            if len(current) > len(best):
                best = current[:]
    
    # Try to extend the best set
    if time.time() < deadline - 10:
        extended = best[:]
        for num in sorted(all_candidates, reverse=True):
            if num in extended:
                continue
            trial_set = extended + [num]
            if len(trial_set) <= 12:
                if _check_validity_fast(trial_set):
                    extended = trial_set
        if len(extended) > len(best):
            best = extended[:]
    
    # Local search: try replacing elements to enable further growth
    if time.time() < deadline - 5:
        for _ in range(100):
            if time.time() > deadline - 5:
                break
            new_best = best[:]
            improved = False
            for i in range(len(new_best)):
                # Try replacing element i with something else
                for replacement in random.sample(all_candidates, 40):
                    if replacement in new_best:
                        continue
                    trial = new_best[:]
                    trial[i] = replacement
                    if _check_validity_fast(trial):
                        new_best = trial
                        improved = True
                        break
                if improved:
                    break
            if len(new_best) > len(best):
                best = new_best[:]
    
    # Final validation
    if not _check_validity(best):
        best = [31, 32, 34]
    
    return list(best)
# EVOLVE-BLOCK-END
