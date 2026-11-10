"""Reciprocal-weight 3-AP-free set on [1,256]."""
# EVOLVE-BLOCK-START
import time

N = 256
DEADLINE_MARGIN = 5.0  # seconds before the 60s limit

def _has_3ap_with(S, x):
    """Check if adding x to set S creates any 3-term AP."""
    S_set = set(S)
    for a in S_set:
        # x could be the middle: a, x, c => c = 2x - a
        c = 2 * x - a
        if c != x and c in S_set:
            return True
        # x could be the end: a, b, x => b = (a+x)/2
        s = a + x
        if s % 2 == 0:
            b = s // 2
            if b != a and b != x and b in S_set:
                return True
    return False

def _reciprocal_sum(S):
    return sum(1.0 / x for x in S)

def _greedy_descending(S, start_val):
    """Try to add numbers from start_val downward, skipping those that break 3-AP."""
    S_list = list(S)
    S_set = set(S_list)
    for x in range(start_val, 0, -1):
        if x in S_set:
            continue
        if not _has_3ap_with(S_list, x):
            S_list.append(x)
            S_set.add(x)
    return S_list

def _greedy_ascending(S, start_val):
    """Try to add numbers from start_val upward."""
    S_list = list(S)
    S_set = set(S_list)
    for x in range(start_val, N + 1):
        if x in S_set:
            continue
        if not _has_3ap_with(S_list, x):
            S_list.append(x)
            S_set.add(x)
    return S_list

def _local_search(S, deadline):
    """Try single additions and swaps to improve reciprocal sum."""
    S_list = list(S)
    S_set = set(S_list)
    current_sum = _reciprocal_sum(S_list)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try adding any missing number
        best_add = None
        best_add_gain = 0
        for x in range(1, N + 1):
            if x in S_set:
                continue
            if not _has_3ap_with(S_list, x):
                gain = 1.0 / x
                if gain > best_add_gain:
                    best_add_gain = gain
                    best_add = x
        if best_add is not None:
            S_list.append(best_add)
            S_set.add(best_add)
            current_sum += best_add_gain
            improved = True
            continue

        # Try single removal + two additions (net +1 element)
        best_move = None
        best_move_gain = 0
        for rem in S_list:
            S_without = [y for y in S_list if y != rem]
            S_without_set = set(S_without)
            # Find best pair to add
            candidates = [x for x in range(1, N + 1) if x not in S_without_set]
            # Sort by reciprocal value descending
            candidates.sort(key=lambda x: -1.0 / x)
            # Try to add the top candidates that fit
            added = []
            for x in candidates:
                if len(added) >= 2:
                    break
                if not _has_3ap_with(S_without, x):
                    # Check it doesn't conflict with already-added in this trial
                    trial = S_without + added
                    if not _has_3ap_with(trial, x):
                        added.append(x)
            if len(added) >= 2:
                gain = sum(1.0 / x for x in added) - 1.0 / rem
                if gain > best_move_gain:
                    best_move_gain = gain
                    best_move = (rem, added)
        if best_move is not None:
            rem, added = best_move
            S_list = [y for y in S_list if y != rem]
            S_list.extend(added)
            S_set = set(S_list)
            current_sum += best_move_gain
            improved = True
            continue

        # Try single removal + one addition with higher value (swap)
        best_swap = None
        best_swap_gain = 0
        for rem in S_list:
            S_without = [y for y in S_list if y != rem]
            S_without_set = set(S_without)
            for x in range(1, N + 1):
                if x in S_without_set or x == rem:
                    continue
                if not _has_3ap_with(S_without, x):
                    gain = 1.0 / x - 1.0 / rem
                    if gain > best_swap_gain:
                        best_swap_gain = gain
                        best_swap = (rem, x)
        if best_swap is not None:
            rem, x = best_swap
            S_list = [y for y in S_list if y != rem]
            S_list.append(x)
            S_set = set(S_list)
            current_sum += best_swap_gain
            improved = True
            continue
    return S_list

def construct():
    start_time = time.time()
    deadline = start_time + (60.0 - DEADLINE_MARGIN)

    # Start with the highest-value (smallest) numbers, greedily
    # We want small numbers. Greedy from 1 upward, only adding if no 3-AP.
    S = []
    S_set = set()
    for x in range(1, N + 1):
        if not _has_3ap_with(S, x):
            S.append(x)
            S_set.add(x)

    best_S = list(S)
    best_sum = _reciprocal_sum(best_S)

    # Try local search
    if time.time() < deadline:
        S2 = _local_search(S, deadline)
        s2 = _reciprocal_sum(S2)
        if s2 > best_sum:
            best_S = S2
            best_sum = s2

    # Try alternative greedy: start with small numbers but skip some to allow more
    # Heuristic: greedy descending from high values first to fill gaps, then add small
    if time.time() < deadline:
        S3 = _greedy_descending([], N)
        s3 = _reciprocal_sum(S3)
        if s3 > best_sum:
            best_S = S3
            best_sum = s3

    # Another strategy: prioritize very small numbers, use local search
    if time.time() < deadline:
        S4 = []
        for x in range(1, 65):  # focus on small numbers first
            if not _has_3ap_with(S4, x):
                S4.append(x)
        S4 = _local_search(S4, deadline)
        s4 = _reciprocal_sum(S4)
        if s4 > best_sum:
            best_S = S4
            best_sum = s4

    # Ensure nonempty
    if not best_S:
        best_S = [1]

    return best_S
# EVOLVE-BLOCK-END
