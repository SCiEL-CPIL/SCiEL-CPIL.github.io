"""
Program for large reciprocal-subsum-free sets.
Combines:
  - Integer arithmetic with LCM scaling for fast exact subset-sum checks
  - Meet-in-the-middle for larger sets (fast), DFS with pruning for small sets
  - Greedy construction with multiple orderings + random restarts
  - Local-search swap improvement (remove one, try to add a better one)
"""
import time
import math
import random
from math import gcd

_START_TIME = time.time()
_DEADLINE = 80.0  # Safety margin under the 90s wall-clock limit


def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)


def _lcm(a, b):
    return a * b // gcd(a, b)


def _lcm_list(lst):
    if not lst:
        return 1
    result = 1
    for x in lst:
        result = _lcm(result, x)
    return result

_lcm_cache = {}

def _lcm_list_cached(lst):
    key = tuple(sorted(lst))
    if key in _lcm_cache:
        return _lcm_cache[key]
    result = 1
    for x in key:
        result = _lcm(result, x)
    _lcm_cache[key] = result
    return result


def _can_form_sum_mitm(target_int, items_int):
    """
    Check if target_int can be formed as a sum of a non-empty subset of items_int
    using meet-in-the-middle. All values are positive integers.
    Returns True if a non-empty subset sums to target_int.
    """
    n = len(items_int)
    if n == 0:
        return target_int == 0
    if target_int <= 0:
        return False
    if sum(items_int) < target_int:
        return False

    # Quick check: if target equals any single item
    if target_int in items_int:
        return True

    mid = n // 2
    left_items = items_int[:mid]
    right_items = items_int[mid:]

    # Generate all subset sums for left half (include 0 for empty subset)
    left_sums = {0}
    for item in left_items:
        new_sums = {s + item for s in left_sums}
        left_sums.update(new_sums)

    # Check if target is in left_sums
    if target_int in left_sums:
        return True

    # Generate all subset sums for right half
    right_sums = {0}
    for item in right_items:
        new_sums = {s + item for s in right_sums}
        right_sums.update(new_sums)

    # Check if there's a pair (l, r) such that l + r = target_int
    for l in left_sums:
        r_needed = target_int - l
        if r_needed in right_sums:
            return True

    return False


def _can_form_sum_dfs(target_int, items_int):
    """
    Check if target_int can be formed as a sum of a non-empty subset of items_int.
    All values are positive integers.
    Uses DFS with pruning based on suffix sums.
    """
    if not items_int:
        return target_int == 0
    if target_int <= 0:
        return False
    if sum(items_int) < target_int:
        return False

    # Quick check: if target equals any single item
    if target_int in items_int:
        return True

    n = len(items_int)
    items_sorted = sorted(items_int, reverse=True)

    suffix_sums = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i + 1] + items_sorted[i]

    found = [False]

    def dfs(i, current_sum, is_non_empty):
        if found[0]:
            return
        if i == n:
            if current_sum == target_int and is_non_empty:
                found[0] = True
            return
        if current_sum + suffix_sums[i] < target_int:
            return
        if current_sum > target_int:
            return
        # Try including item
        new_sum = current_sum + items_sorted[i]
        if new_sum <= target_int:
            dfs(i + 1, new_sum, True)
            if found[0]:
                return
        # Try excluding item
        dfs(i + 1, current_sum, is_non_empty)

    dfs(0, 0, False)
    return found[0]


def _can_form_sum(target_int, items_int):
    """
    Check if target_int can be formed as a sum of a non-empty subset of items_int.
    Uses meet-in-the-middle for larger item lists, DFS for smaller ones.
    """
    if not items_int:
        return target_int == 0
    if target_int <= 0:
        return False
    if sum(items_int) < target_int:
        return False

    n = len(items_int)
    # Meet-in-the-middle is O(2^(n/2)) space/time, DFS is O(2^n) worst case
    # For n <= 12, DFS with pruning is often fast; for n > 12, MITM is better
    if n <= 12:
        return _can_form_sum_dfs(target_int, items_int)
    else:
        return _can_form_sum_mitm(target_int, items_int)


def _is_subsum_free(selected, candidate):
    """
    Check if adding candidate to selected keeps the set subsum-free.
    selected: list of ints
    candidate: int
    """
    # Check for candidate: 1/candidate should not equal sum of reciprocals of any non-empty subset of selected
    if selected:
        all_items = selected + [candidate]
        lcm_val = _lcm_list_cached(all_items)
        target_int = lcm_val // candidate
        items_int = [lcm_val // x for x in selected]
        if _can_form_sum(target_int, items_int):
            return False

    # Check for each element in selected: 1/x should not equal sum of reciprocals
    # of any non-empty subset of (selected - {x}) + {candidate}
    for x in selected:
        others = [y for y in selected if y != x]
        others.append(candidate)
        lcm_val = _lcm_list_cached(others)
        target_int = lcm_val // x
        items_int = [lcm_val // y for y in others]
        if _can_form_sum(target_int, items_int):
            return False

    return True


def _is_valid_set(S):
    """
    Verify that S is a valid reciprocal-subsum-free set.
    For each element x in S, check that 1/x is not the sum of reciprocals of
    any non-empty subset of S \ {x}.
    """
    for x in S:
        others = [y for y in S if y != x]
        if not others:
            continue
        lcm_val = _lcm_list_cached(others)
        target_int = lcm_val // x
        items_int = [lcm_val // y for y in others]
        if _can_form_sum(target_int, items_int):
            return False
    return True


def _greedy_construct(order):
    """Greedy construction given an ordering of candidates."""
    trial_set = []
    for candidate in order:
        if _time_remaining() < 3.0:
            break
        if _is_subsum_free(trial_set, candidate):
            trial_set.append(candidate)
    return trial_set


def _local_search_improve(S):
    """
    Try to improve set S by removing one element and adding a different one
    that keeps the set valid. Returns the best set found.
    """
    best = list(S)
    best_len = len(best)
    all_elements = set(range(1, 25))
    current = set(best)

    # Try all single removals and see if we can add a better element
    for removed in list(best):
        if _time_remaining() < 5.0:
            break
        remaining = [x for x in best if x != removed]
        remaining_set = set(remaining)

        # Try to add any element not in current set
        for candidate in range(24, 0, -1):
            if candidate in current:
                continue
            if _time_remaining() < 3.0:
                break
            new_set = remaining + [candidate]
            if _is_valid_set(new_set):
                if len(new_set) > best_len:
                    best = new_set
                    best_len = len(best)
                    current = set(best)
                    break

    # Try all single additions (if there's room)
    for candidate in range(24, 0, -1):
        if candidate in current:
            continue
        if _time_remaining() < 3.0:
            break
        new_set = best + [candidate]
        if _is_valid_set(new_set):
            if len(new_set) > best_len:
                best = new_set
                best_len = len(best)
                current = set(best)

    return best


def _branch_and_bound_max_cardinality():
    """
    Branch and bound search to maximize cardinality of a reciprocal-subsum-free set.
    Uses the integer Diophantine formulation for exact checking.
    """
    all_elements = list(range(24, 0, -1))  # Try large elements first
    best_set = []
    best_len = 0

    def backtrack(start_idx, current_set):
        nonlocal best_set, best_len

        if len(current_set) > best_len:
            best_set = list(current_set)
            best_len = len(best_set)

        # Pruning: even if we add all remaining elements, can we beat best?
        remaining = 24 - start_idx
        if len(current_set) + remaining <= best_len:
            return

        if _time_remaining() < 5.0:
            return

        for i in range(start_idx, 24):
            candidate = all_elements[i]
            if _is_subsum_free(current_set, candidate):
                current_set.append(candidate)
                backtrack(i + 1, current_set)
                current_set.pop()

                # If we found a set of size 24, we're done
                if best_len == 24:
                    return

    backtrack(0, [])
    return best_set


def construct():
    """
    Construct a large reciprocal-subsum-free subset of [1, 24].
    Uses greedy construction with multiple orderings, random restarts,
    local search improvement, and branch-and-bound.
    """
    all_elements = list(range(1, 25))
    best_set = []

    # 1. Deterministic greedy: Large to Small
    if _time_remaining() > 10.0:
        s = _greedy_construct(sorted(all_elements, reverse=True))
        if len(s) > len(best_set):
            best_set = s

    # 2. Deterministic greedy: Small to Large
    if _time_remaining() > 10.0:
        s = _greedy_construct(sorted(all_elements))
        if len(s) > len(best_set):
            best_set = s

    # 3. Deterministic greedy: Alternating large/small
    if _time_remaining() > 10.0:
        order = []
        large = sorted(all_elements, reverse=True)
        small = sorted(all_elements)
        for i in range(12):
            if i < len(large):
                order.append(large[i])
            if i < len(small):
                order.append(small[i])
        s = _greedy_construct(order)
        if len(s) > len(best_set):
            best_set = s

    # 4. Local search improvement on the best so far
    if _time_remaining() > 15.0 and best_set:
        improved = _local_search_improve(best_set)
        if len(improved) > len(best_set):
            best_set = improved

    # 5. Random restarts
    random.seed(42)
    for _ in range(200):
        if _time_remaining() < 20.0:
            break

        candidates = list(all_elements)
        random.shuffle(candidates)
        trial_set = _greedy_construct(candidates)

        if len(trial_set) > len(best_set):
            best_set = trial_set
            if _time_remaining() > 20.0:
                improved = _local_search_improve(best_set)
                if len(improved) > len(best_set):
                    best_set = improved

    # 6. Biased random restarts: favor large elements
    for _ in range(100):
        if _time_remaining() < 20.0:
            break

        candidates = list(all_elements)
        random.shuffle(candidates)
        large_elems = [x for x in candidates if x > 12]
        small_elems = [x for x in candidates if x <= 12]
        random.shuffle(large_elems)
        random.shuffle(small_elems)
        candidates = large_elems + small_elems

        trial_set = _greedy_construct(candidates)

        if len(trial_set) > len(best_set):
            best_set = trial_set
            if _time_remaining() > 20.0:
                improved = _local_search_improve(best_set)
                if len(improved) > len(best_set):
                    best_set = improved

    # 7. Branch and bound for maximum cardinality
    if _time_remaining() > 30.0:
        bnb_set = _branch_and_bound_max_cardinality()
        if len(bnb_set) > len(best_set):
            best_set = bnb_set

    # 8. Final local search pass on the best set
    if _time_remaining() > 10.0 and best_set:
        improved = _local_search_improve(best_set)
        if len(improved) > len(best_set):
            best_set = improved

    # 9. Try to add any remaining elements
    if _time_remaining() > 5.0 and best_set:
        current_set = set(best_set)
        for candidate in range(24, 0, -1):
            if candidate in current_set:
                continue
            if _time_remaining() < 3.0:
                break
            new_set = best_set + [candidate]
            if _is_valid_set(new_set):
                best_set = new_set
                current_set.add(candidate)

    # Ensure nonempty
    if not best_set:
        best_set = [24]

    return sorted(best_set)


if __name__ == "__main__":
    s = construct()
    print(s)
    print(len(s))
