"""Combined solver: maximize |S| ⊆ [1,192], |S|≤48, no a∈S divides sum of any 3 distinct others in S."""

import time
import random
from collections import defaultdict

_DEADLINE = 85.0
_START = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _START)

def _is_safe_to_add(selected, x):
    """Check if adding x to valid set 'selected' keeps it valid.
    Uses remainder counting for efficiency."""
    n = len(selected)
    if n < 2:
        return True

    # Check 1: Does x divide sum of any 3 distinct elements in selected?
    if n >= 3:
        rem_count = defaultdict(int)
        for v in selected:
            rem_count[v % x] += 1
        for i in range(n):
            u = selected[i]
            for j in range(i + 1, n):
                v = selected[j]
                target = (-u - v) % x
                cnt = rem_count.get(target, 0)
                if u % x == target:
                    cnt -= 1
                if v % x == target:
                    cnt -= 1
                if cnt > 0:
                    return False

    # Check 2: For each a in selected, does a divide x + u + v for distinct u,v in selected\{a}?
    for i, a in enumerate(selected):
        others = [selected[j] for j in range(n) if j != i]
        if len(others) < 2:
            continue
        target_rem = (-x) % a
        rem_count = defaultdict(int)
        for v in others:
            rem_count[v % a] += 1
        for idx, u in enumerate(others):
            needed = (target_rem - u % a) % a
            cnt = rem_count.get(needed, 0)
            if u % a == needed:
                cnt -= 1
            if cnt > 0:
                return False

    return True

def _check_validity(selected):
    """Full validation check."""
    n = len(selected)
    for i, a in enumerate(selected):
        others = [selected[j] for j in range(n) if j != i]
        if len(others) < 3:
            continue
        rem_count = defaultdict(int)
        for v in others:
            rem_count[v % a] += 1
        for pi in range(len(others)):
            u = others[pi]
            for pj in range(pi + 1, len(others)):
                v = others[pj]
                target = (-u - v) % a
                cnt = rem_count.get(target, 0)
                if u % a == target:
                    cnt -= 1
                if v % a == target:
                    cnt -= 1
                if cnt > 0:
                    return False
    return True

def _greedy(order):
    """Greedy: add numbers in given order."""
    selected = []
    for x in order:
        if _time_left() < 1.0:
            break
        if _is_safe_to_add(selected, x):
            selected.append(x)
    return selected

def _local_search_add(initial, max_iter=100):
    """Try to add more elements to initial set."""
    selected = list(initial)
    sel_set = set(selected)
    for _ in range(max_iter):
        if _time_left() < 2.0:
            break
        improved = False
        for x in range(1, 193):
            if x in sel_set:
                continue
            if _is_safe_to_add(selected, x):
                selected.append(x)
                sel_set.add(x)
                improved = True
                break
        if not improved:
            break
    return selected

def _swap_search(initial, max_rounds=50):
    """Try to grow set by removing 1 element and adding 2 others."""
    S = list(initial)
    S_set = set(S)
    improved = True
    rounds = 0
    while improved and len(S) < 48 and _time_left() > 3.0 and rounds < max_rounds:
        improved = False
        rounds += 1
        for i in range(len(S)):
            if _time_left() < 3.0:
                return S
            S_temp = [S[j] for j in range(len(S)) if j != i]
            S_set_temp = set(S_temp)
            # Try adding first element
            for x1 in range(192, 0, -1):
                if x1 in S_set_temp:
                    continue
                if _is_safe_to_add(S_temp, x1):
                    S_temp2 = S_temp + [x1]
                    S_set_temp2 = S_set_temp | {x1}
                    # Try adding second element
                    for x2 in range(192, 0, -1):
                        if x2 in S_set_temp2:
                            continue
                        if _is_safe_to_add(S_temp2, x2):
                            S_new = S_temp2 + [x2]
                            if len(S_new) > len(S):
                                S = S_new
                                S_set = set(S)
                                improved = True
                                break
                    if improved:
                        break
            if improved:
                break
    return S

def construct():
    global _START
    _START = time.time()

    best = []
    best_size = 0

    # Multiple deterministic greedy passes
    for order in [range(192, 0, -1), range(1, 193), range(96, 0, -1), range(97, 193)]:
        if _time_left() < 2.0:
            break
        cand = _greedy(order)
        if len(cand) > best_size:
            best = cand
            best_size = len(cand)

    # Local search additions
    if _time_left() > 3.0:
        improved = _local_search_add(best, max_iter=50)
        if len(improved) > best_size:
            best = improved
            best_size = len(improved)

    # Swap search to grow further
    if _time_left() > 5.0:
        swapped = _swap_search(best, max_rounds=30)
        if len(swapped) > best_size:
            best = swapped
            best_size = len(swapped)

    # Randomized restarts
    random.seed(42)
    for restart in range(300):
        if _time_left() < 5.0:
            break

        nums = list(range(1, 193))
        random.shuffle(nums)

        selected = _greedy(nums)

        if len(selected) > best_size:
            best = selected
            best_size = len(selected)

        # Local search on promising candidates
        if _time_left() > 5.0 and len(selected) >= best_size - 1:
            improved = _local_search_add(selected, max_iter=30)
            if len(improved) > best_size:
                best = improved
                best_size = len(improved)

        # Swap search on best candidate if time allows
        if _time_left() > 10.0 and len(selected) >= best_size:
            swapped = _swap_search(selected, max_rounds=15)
            if len(swapped) > best_size:
                best = swapped
                best_size = len(swapped)

    return best

if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
    print(f"Valid: {_check_validity(res)}")
    print(res)
