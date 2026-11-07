"""Search-based approach to maximize set size under divisibility constraint on subset sums."""
import time
import random

# EVOLVE-BLOCK-START

DEADLINE_SECONDS = 85.0


def _subset_sums_incr(sums_set, x):
    """Given existing subset sums set (nonempty), add x to all and to itself.
    Returns the new set of subset sums for the augmented list."""
    return sums_set | {s + x for s in sums_set} | {x}


def _divisibility_conflict(sums_set):
    """Check if any two distinct values in sums_set divide one another.
    Returns True if conflict exists (invalid), False otherwise."""
    if len(sums_set) < 2:
        return False
    vals = sorted(sums_set)
    max_val = vals[-1]
    for vi in vals:
        if vi > max_val // 2:
            continue
        multiple = vi * 2
        while multiple <= max_val:
            if multiple in sums_set:
                return True
            multiple += vi
    return False


def _is_valid_set(elements):
    """Full validation: all nonempty subset sums, no two distinct divide each other."""
    if not elements:
        return True
    sums_set = set()
    for x in elements:
        sums_set = _subset_sums_incr(sums_set, x)
    return not _divisibility_conflict(sums_set)


def _greedy_randomized(elements_pool, rng, max_size=30, time_limit=None):
    """Randomized greedy: repeatedly pick a random element that keeps validity."""
    current = []
    sums_set = set()
    remaining = list(elements_pool)
    rng.shuffle(remaining)

    if time_limit is not None:
        deadline = time.time() + time_limit
    else:
        deadline = None

    while len(current) < max_size and remaining:
        found = False
        for i in range(len(remaining)):
            x = remaining[i]
            new_sums = _subset_sums_incr(sums_set, x)
            if not _divisibility_conflict(new_sums):
                current.append(x)
                sums_set = new_sums
                remaining.pop(i)
                found = True
                break
            if deadline is not None and time.time() > deadline:
                break
        if not found:
            break
        if deadline is not None and time.time() > deadline:
            break

    return current


def _local_search(current, elements_pool, rng, time_limit=None):
    """Try to improve current set by swapping or adding elements."""
    current = list(current)
    current_set = set(current)
    sums_set = set()
    for x in current:
        sums_set = _subset_sums_incr(sums_set, x)

    if time_limit is not None:
        deadline = time.time() + time_limit
    else:
        deadline = None

    improved = True
    while improved:
        improved = False
        remaining = [x for x in elements_pool if x not in current_set]
        rng.shuffle(remaining)

        # Try adding
        for x in remaining:
            if len(current) >= 30:
                break
            if deadline is not None and time.time() > deadline:
                break
            new_sums = _subset_sums_incr(sums_set, x)
            if not _divisibility_conflict(new_sums):
                current.append(x)
                current_set.add(x)
                sums_set = new_sums
                improved = True
                break

        if deadline is not None and time.time() > deadline:
            break

        # Try swaps: remove one, add another
        if not improved and len(current) > 0:
            for i in range(len(current)):
                if deadline is not None and time.time() > deadline:
                    break
                removed = current[i]
                new_current = current[:i] + current[i+1:]
                new_sums = set()
                for x in new_current:
                    new_sums = _subset_sums_incr(new_sums, x)
                
                for x in remaining:
                    if x in current_set:
                        continue
                    test_sums = _subset_sums_incr(new_sums, x)
                    if not _divisibility_conflict(test_sums):
                        current = new_current + [x]
                        current_set = set(current)
                        sums_set = test_sums
                        improved = True
                        break
                if improved:
                    break

    return current


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS

    best = []
    best_size = 0
    rng = random.Random(42)

    # Define pools based on number-theoretic insights and full range
    pool_full = list(range(1, 257))
    pools = [
        pool_full,
        list(range(128, 257)),       # Upper half
        list(range(128, 192)),       # Narrow [2^7, 1.5*2^7)
        list(range(64, 96)),         # Narrow [2^6, 1.5*2^6)
        list(range(96, 128)),        # Mid-high
        list(range(192, 257)),       # Top quarter
    ]

    # Strategy 1: Greedy on various pools with randomization
    for pool in pools:
        if time.time() > deadline - 15.0:
            break
        
        # Run multiple short greedy attempts
        for _ in range(5):
            if time.time() > deadline - 15.0:
                break
            t_rem = deadline - time.time() - 15.0
            if t_rem < 0.5:
                break
            res = _greedy_randomized(pool, rng, max_size=30, time_limit=min(t_rem, 2.0))
            if len(res) > best_size:
                best = res
                best_size = len(res)

    # Strategy 2: Local Search on best found
    if best and time.time() < deadline - 5.0:
        t_rem = deadline - time.time() - 5.0
        improved = _local_search(best, pool_full, rng, time_limit=t_rem)
        if len(improved) > best_size:
            best = improved
            best_size = len(best)

    # Final verification
    if best:
        if not _is_valid_set(best):
            best = [31, 32, 34]

    return [int(x) for x in best]

# EVOLVE-BLOCK-END
