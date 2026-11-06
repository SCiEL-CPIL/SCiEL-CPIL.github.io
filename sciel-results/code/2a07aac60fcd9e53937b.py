import time
import random

def construct():
    """
    Construct a set of up to 48 distinct positive integers <= 512 such that no element
    is the exact arithmetic mean of any subset of at least two OTHER elements.
    Maximize cardinality.
    """
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    
    best_set = []
    
    # Strategy 1: Greedy Descending (Large numbers are safer)
    if time.time() < deadline:
        s = _greedy_construct(range(512, 0, -1), deadline)
        if len(s) > len(best_set):
            best_set = s
            if len(best_set) == 48:
                return best_set

    # Strategy 2: Greedy Ascending
    if time.time() < deadline:
        s = _greedy_construct(range(1, 513), deadline)
        if len(s) > len(best_set):
            best_set = s
            if len(best_set) == 48:
                return best_set

    # Strategy 3: Random Shuffles
    if len(best_set) < 48 and time.time() < deadline:
        for _ in range(20): 
            if time.time() > deadline:
                break
            candidates = list(range(1, 513))
            random.shuffle(candidates)
            s = _greedy_construct(candidates, deadline)
            if len(s) > len(best_set):
                best_set = s
                if len(best_set) == 48:
                    break
                    
    return best_set

def _greedy_construct(candidates, deadline):
    """
    Greedily add elements from candidates if valid.
    """
    selected = []
    selected_set = set()
    
    for x in candidates:
        if time.time() > deadline:
            break
        if len(selected) >= 48:
            break
        if x in selected_set:
            continue
            
        if _is_valid_addition(selected, x):
            selected.append(x)
            selected_set.add(x)
            
    return selected

def _is_valid_addition(selected, x):
    """
    Check if adding x to selected maintains the no-mean property.
    """
    n = len(selected)
    
    # Check 1: Is x the mean of a subset of selected?
    # Sum of k elements = k * x, k >= 2.
    if n >= 2:
        if _subset_mean_exists(selected, x):
            return False

    # Check 2: For each a in selected, is a the mean of a subset including x?
    # Subset T from selected \ {a}, size k >= 1.
    # Sum(T) + x = (k+1) * a
    # Sum(T) = (k+1) * a - x
    # We need to check if Sum(T) is achievable by k elements from selected \ {a}.

    for a in selected:
        others = [y for y in selected if y != a]
        if not others:
            continue

        if _subset_sum_with_target_exists(others, a, x):
            return False

    return True

def _subset_mean_exists(selected, x):
    """
    Check if there is a subset of 'selected' of size k >= 2
    such that sum(subset) == k * x.
    """
    n = len(selected)
    if n < 2:
        return False

    # Fast path: Check k=2
    # Need pair (u, v) such that u + v = 2x
    target2 = 2 * x
    sel_set = set(selected)
    for u in selected:
        v = target2 - u
        if v != u and v in sel_set:
            return True
        # Note: if v == u, we need two copies of u, but elements are distinct.
        # So v != u is required.

    # Fast path: Check k=3
    # Need triple (u, v, w) such that u + v + w = 3x
    target3 = 3 * x
    # We can do this in O(N^2)
    for i in range(n):
        for j in range(i + 1, n):
            w = target3 - selected[i] - selected[j]
            if w > selected[j] and w in sel_set:
                return True
            # Also check if w is in the set and distinct from i, j
            # Since we iterate i<j, and w > selected[j] ensures w is distinct and not i or j (if sorted)
            # But selected is not necessarily sorted.
            # Let's just check membership and distinctness.
            if w in sel_set and w != selected[i] and w != selected[j]:
                 # Need to ensure we don't count same element twice if w == selected[i] etc?
                 # The loop i<j ensures selected[i] != selected[j].
                 # If w == selected[i], then w is used twice? No, w is a value.
                 # If w == selected[i], then we have selected[i], selected[j], selected[i]. Not distinct.
                 # So we must ensure w is not selected[i] or selected[j].
                 # The condition `w != selected[i] and w != selected[j]` handles this.
                 # But we also need to ensure w is not some other element used?
                 # No, just need 3 distinct elements.
                 # If w is in set, and w != u, w != v, then {u, v, w} are distinct.
                 pass

    # Full DP for k >= 4
    max_sum = sum(selected)
    
    dp = [0] * (n + 1)
    dp[0] = 1  # sum 0 with 0 elements

    for elem in selected:
        for k in range(n - 1, -1, -1):
            if dp[k]:
                dp[k + 1] |= (dp[k] << elem)

    # Check if for any k >= 4, bit (k*x) is set in dp[k]
    for k in range(4, n + 1):
        target = k * x
        if target > max_sum:
            break
        if (dp[k] >> target) & 1:
            return True

    return False

def _subset_sum_with_target_exists(others, a, x):
    """
    Check if there exists k >= 1 such that sum of k elements from 'others'
    equals (k+1)*a - x.
    """
    n = len(others)
    if n == 0:
        return False

    # Fast path: Check k=1
    # Need 1 element y such that y = 2a - x
    target1 = 2 * a - x
    if target1 in set(others):
        return True

    # Fast path: Check k=2
    # Need 2 elements u, v such that u + v = 3a - x
    target2 = 3 * a - x
    others_set = set(others)
    for u in others:
        v = target2 - u
        if v != u and v in others_set:
            return True

    # Full DP for k >= 3
    max_sum = sum(others)

    dp = [0] * (n + 1)
    dp[0] = 1

    for elem in others:
        for k in range(n - 1, -1, -1):
            if dp[k]:
                dp[k + 1] |= (dp[k] << elem)

    for k in range(3, n + 1):
        target = (k + 1) * a - x
        if target < 0:
            continue
        if target > max_sum:
            continue
        if (dp[k] >> target) & 1:
            return True

    return False
