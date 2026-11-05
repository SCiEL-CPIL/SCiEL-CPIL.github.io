"""Maximize cardinality of a dissociated reciprocal subset of [1,22]."""
import time
import random
import sys
from functools import lru_cache

_START = time.time()
_DEADLINE = 80.0  # safety margin before 90s wall-clock

def _time_ok():
    return (time.time() - _START) < _DEADLINE

# LCM(1..22) = 232792560
LCM = 232792560
# weights[i] corresponds to reciprocal 1/(i+1), scaled by LCM
weights = [LCM // (i + 1) for i in range(22)]


def _subset_sums_set(w_list):
    """Compute all subset sums of w_list as a set."""
    sums = {0}
    for w in w_list:
        sums = sums | {s + w for s in sums}
    return sums


def _is_dissociated(selected_indices):
    """Check if selected indices form a dissociated set (all subset sums distinct)."""
    w = [weights[i] for i in selected_indices]
    n = len(w)
    sums = set()
    for mask in range(1 << n):
        s = 0
        for b in range(n):
            if mask & (1 << b):
                s += w[b]
        if s in sums:
            return False
        sums.add(s)
    return True


def _check_add(current_sums, w):
    """Check if adding weight w to a set with subset sums `current_sums` keeps dissociation.
    Returns new_sums if valid, None otherwise."""
    shifted = {s + w for s in current_sums}
    if shifted & current_sums:
        return None
    return current_sums | shifted


def _greedy(order):
    """Greedily build a dissociated set following the given order."""
    sums = {0}
    chosen = []
    for i in order:
        new_sums = _check_add(sums, weights[i])
        if new_sums is not None:
            sums = new_sums
            chosen.append(i)
    return chosen


def _local_improve(chosen, all_indices):
    """Try to improve the set by swaps and additions."""
    chosen_set = set(chosen)
    sums = _subset_sums_set([weights[i] for i in chosen])
    
    # Try adding elements
    for i in all_indices:
        if i in chosen_set:
            continue
        new_sums = _check_add(sums, weights[i])
        if new_sums is not None:
            chosen.append(i)
            chosen_set.add(i)
            sums = new_sums
    
    # Try swaps: remove one, add one
    improved = True
    while improved:
        improved = False
        for rem in list(chosen):
            # Try removing rem and adding something else
            remaining_sums = _subset_sums_set([weights[i] for i in chosen if i != rem])
            for add in all_indices:
                if add in chosen_set or add == rem:
                    continue
                new_sums = _check_add(remaining_sums, weights[add])
                if new_sums is not None:
                    # Swap
                    chosen.remove(rem)
                    chosen_set.discard(rem)
                    chosen.append(add)
                    chosen_set.add(add)
                    sums = new_sums
                    improved = True
                    break
            if improved:
                break
    return chosen


def construct():
    """Return a nonempty subset of [1,22] with all reciprocal subset sums distinct,
    maximizing cardinality."""
    all_indices = list(range(22))
    
    best_size = 0
    best_set = None
    
    # Generate many different orderings
    orderings = []
    
    # Deterministic orderings
    orderings.append(sorted(all_indices, key=lambda i: weights[i], reverse=True))  # weight desc
    orderings.append(sorted(all_indices, key=lambda i: weights[i]))  # weight asc
    orderings.append(sorted(all_indices, reverse=True))  # k desc
    orderings.append(sorted(all_indices))  # k asc
    orderings.append(sorted(all_indices, key=lambda i: -weights[i] / (i + 1)))  # ratio
    
    # Random orderings
    for seed in range(200):
        rng = random.Random(seed)
        idx = all_indices[:]
        rng.shuffle(idx)
        orderings.append(idx)
    
    # Run greedy on all orderings
    for order in orderings:
        if not _time_ok():
            break
        result = _greedy(order)
        if len(result) > best_size:
            best_size = len(result)
            best_set = result[:]
    
    # Local improvement on best found
    if best_set:
        improved = _local_improve(best_set[:], all_indices)
        if len(improved) > best_size:
            best_size = len(improved)
            best_set = improved
    
    # Backtracking search to try to beat current best
    def _backtrack(current_sums, chosen, remaining, target_size):
        if not _time_ok():
            return None
        if len(chosen) == target_size:
            return chosen[:]
        if len(chosen) + len(remaining) < target_size:
            return None
        
        for j, i in enumerate(remaining):
            if not _time_ok():
                return None
            new_sums = _check_add(current_sums, weights[i])
            if new_sums is not None:
                result = _backtrack(new_sums, chosen + [i], remaining[j + 1:], target_size)
                if result is not None:
                    return result
        return None
    
    # Try to find larger sets
    for target in range(best_size + 1, 14):
        if not _time_ok():
            break
        # Try multiple starting orders for backtracking
        start_orders = [
            sorted(all_indices, key=lambda i: weights[i], reverse=True),
            sorted(all_indices, key=lambda i: weights[i]),
            sorted(all_indices, reverse=True),
        ]
        for seed in range(50):
            rng = random.Random(seed + 1000)
            idx = all_indices[:]
            rng.shuffle(idx)
            start_orders.append(idx)
        
        for order in start_orders:
            if not _time_ok():
                break
            result = _backtrack({0}, [], order, target)
            if result is not None:
                best_size = target
                best_set = result
                break
        if best_size == target:
            break
    
    # Final verification
    if best_set and not _is_dissociated(best_set):
        # Fallback: try a few known-good patterns
        fallbacks = [
            [0, 1, 2, 4, 6, 10, 12, 16, 18],  # k=1,2,3,5,7,11,13,17,19
            [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],  # even k
        ]
        for fb in fallbacks:
            if _is_dissociated(fb):
                best_set = fb
                best_size = len(fb)
                break
        else:
            best_set = [0]  # just {1}
            best_size = 1
    
    # Convert to values
    result = sorted([i + 1 for i in best_set])
    if not result:
        result = [1]
    return result
