"""Divisibility-incomparable subset sums: stronger search.

Key insight: the constraint is that no two distinct nonempty subset sums divide one another.
This is extremely restrictive. We use:
1. Fast incremental validity checking with early termination.
2. Multiple construction strategies (greedy, simulated annealing, genetic-style).
3. Careful time management with a hard deadline.
"""

import time
import random
import sys

START_TIME = time.time()
TIME_LIMIT = 85.0  # 90s kill, leave 5s margin


def _check_divisibility(sums_set):
    """Check if any two distinct sums in the set have a divisibility relation.
    Returns True if valid (no divisibility), False otherwise."""
    sums_list = sorted(sums_set)
    n = len(sums_list)
    for i in range(n):
        si = sums_list[i]
        if si <= 0:
            continue
        for j in range(i + 1, n):
            if sums_list[j] % si == 0:
                return False
    return True


def _compute_all_sums(nums):
    """Generate all nonempty subset sums of nums as a set."""
    sums = set()
    n = len(nums)
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += nums[i]
        sums.add(s)
    return sums


def _can_add_fast(nums, new_elem, existing_sums):
    """Check if adding new_elem to nums (with known existing_sums) keeps the set valid.
    
    Returns (valid, new_sums) where new_sums is the updated sum set if valid.
    """
    if not nums:
        # Single element: only one sum, always valid
        return True, {new_elem}
    
    # New sums = existing_sums ∪ {s + new_elem for s in existing_sums} ∪ {new_elem}
    new_sums = set(existing_sums)
    for s in existing_sums:
        new_sums.add(s + new_elem)
    new_sums.add(new_elem)
    
    # Check divisibility
    if not _check_divisibility(new_sums):
        return False, None
    return True, new_sums


def _is_valid(nums):
    """Full validity check."""
    if not nums:
        return True
    sums = _compute_all_sums(nums)
    return _check_divisibility(sums)


def _greedy_construct(pool, rng, max_size=30):
    """Greedy construction with random order."""
    result = []
    result_sums = set()
    remaining = list(pool)
    rng.shuffle(remaining)
    
    for elem in remaining:
        if time.time() - START_TIME > TIME_LIMIT:
            break
        valid, new_sums = _can_add_fast(result, elem, result_sums)
        if valid:
            result.append(elem)
            result_sums = new_sums
            if len(result) >= max_size:
                break
    return result


def _simulated_annealing(pool, rng, start_size=5, max_iter=500):
    """Simulated annealing to build a valid set."""
    # Start with a small valid set
    initial_pool = list(pool)
    rng.shuffle(initial_pool)
    current = []
    current_sums = set()
    
    for elem in initial_pool[:start_size]:
        valid, new_sums = _can_add_fast(current, elem, current_sums)
        if valid:
            current.append(elem)
            current_sums = new_sums
    
    if not current:
        return []
    
    best = list(current)
    best_sums = set(current_sums)
    best_size = len(current)
    
    available = [x for x in pool if x not in set(current)]
    temp = 1.0
    cooling = 0.995
    
    for it in range(max_iter):
        if time.time() - START_TIME > TIME_LIMIT:
            break
        
        # Random move: add, remove, or swap
        move_type = rng.random()
        
        if move_type < 0.3 and available:
            # Try adding
            elem = rng.choice(available)
            valid, new_sums = _can_add_fast(current, elem, current_sums)
            if valid:
                current.append(elem)
                current_sums = new_sums
                available.remove(elem)
                if len(current) > best_size:
                    best = list(current)
                    best_sums = set(current_sums)
                    best_size = len(current)
        elif move_type < 0.5 and len(current) > 1:
            # Try removing
            idx = rng.randrange(len(current))
            removed = current.pop(idx)
            available.append(removed)
            # Recompute sums
            current_sums = _compute_all_sums(current) if current else set()
        else:
            # Try swap
            if available and current:
                idx = rng.randrange(len(current))
                old_elem = current[idx]
                new_elem = rng.choice(available)
                trial = current[:idx] + current[idx+1:] + [new_elem]
                if _is_valid(trial):
                    current = trial
                    current_sums = _compute_all_sums(current)
                    available.remove(new_elem)
                    available.append(old_elem)
                    if len(current) > best_size:
                        best = list(current)
                        best_sums = set(current_sums)
                        best_size = len(current)
        
        temp *= cooling
    
    return best


def _local_search_improve(nums, pool, rng, max_rounds=20):
    """Improve a set by trying swaps and additions."""
    current = list(nums)
    current_set = set(current)
    available = [x for x in pool if x not in current_set]
    
    for _ in range(max_rounds):
        if time.time() - START_TIME > TIME_LIMIT:
            break
        
        improved = False
        
        # Try adding elements
        rng.shuffle(available)
        for elem in available:
            valid, new_sums = _can_add_fast(current, elem, _compute_all_sums(current))
            if valid:
                current.append(elem)
                current_set.add(elem)
                available.remove(elem)
                improved = True
                break
        
        # Try swaps
        if not improved and available:
            for i, old_elem in enumerate(current):
                if time.time() - START_TIME > TIME_LIMIT:
                    break
                for new_elem in available:
                    if new_elem in current_set:
                        continue
                    trial = current[:i] + current[i+1:] + [new_elem]
                    if _is_valid(trial):
                        current = trial
                        current_set = set(current)
                        available.remove(new_elem)
                        available.append(old_elem)
                        improved = True
                        break
                if improved:
                    break
        
        if not improved:
            break
    
    return current


def construct():
    """Main entry point. Returns a list of 1..30 distinct integers in 1..256."""
    pool = list(range(1, 257))
    
    best = []
    best_size = 0
    
    # Strategy 1: Multiple greedy passes with different seeds
    for seed in range(15):
        if time.time() - START_TIME > TIME_LIMIT:
            break
        rng = random.Random(seed)
        result = _greedy_construct(pool, rng)
        if len(result) > best_size:
            best = result
            best_size = len(result)
            if best_size >= 30:
                break
    
    # Strategy 2: Local search improvement on best found
    if best and best_size < 30:
        if time.time() - START_TIME < TIME_LIMIT - 10:
            rng = random.Random(999)
            improved = _local_search_improve(best, pool, rng, max_rounds=10)
            if len(improved) > best_size:
                best = improved
                best_size = len(best)
    
    # Strategy 3: Simulated annealing with different starts
    for seed in range(5):
        if time.time() - START_TIME > TIME_LIMIT:
            break
        rng = random.Random(1000 + seed)
        result = _simulated_annealing(pool, rng, start_size=4 + seed, max_iter=300)
        if len(result) > best_size:
            best = result
            best_size = len(result)
            if best_size >= 30:
                break
    
    # Strategy 4: Try biased pools (larger numbers tend to work better)
    for seed in range(5):
        if time.time() - START_TIME > TIME_LIMIT:
            break
        # Use upper half of the range
        upper_pool = list(range(100, 257))
        rng = random.Random(2000 + seed)
        result = _greedy_construct(upper_pool, rng)
        # Then try to add from full pool
        if result and len(result) < 30:
            remaining_pool = [x for x in pool if x not in set(result)]
            rng2 = random.Random(3000 + seed)
            result2 = _local_search_improve(result, pool, rng2, max_rounds=5)
            if len(result2) > best_size:
                best = result2
                best_size = len(best)
                if best_size >= 30:
                    break
    
    # Strategy 5: Try to build from scratch with a focus on spread-out values
    if time.time() - START_TIME < TIME_LIMIT - 5:
        # Use values that are far apart to reduce divisibility risk
        spread_pool = [x for x in range(1, 257) if x % 7 != 0]  # Avoid multiples of 7
        rng = random.Random(4000)
        result = _greedy_construct(spread_pool, rng)
        if result and len(result) < 30:
            result = _local_search_improve(result, pool, rng, max_rounds=5)
        if len(result) > best_size:
            best = result
            best_size = len(best)
    
    # Final fallback
    if not best:
        best = [31, 32, 34]
    
    # Ensure Python ints in a list
    return [int(x) for x in best]


# EVOLVE-BLOCK-START
# (The above code replaces the evolve block)
# EVOLVE-BLOCK-END
