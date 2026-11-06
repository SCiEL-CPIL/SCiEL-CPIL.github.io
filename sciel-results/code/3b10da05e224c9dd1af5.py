import time

def construct():
    """
    Construct a set of up to 48 distinct positive integers <= 512 such that no element
    is the exact arithmetic mean of any subset of at least two OTHER elements.
    Maximize cardinality.
    """
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit

    selected = []
    
    # Maintain global subset-sum DP: dp[k] is bitset of achievable sums with k elements
    max_n = 48
    max_sum = 512 * max_n  # 24576
    dp = [0] * (max_n + 1)
    dp[0] = 1  # sum 0 with 0 elements

    # Strategic ordering: powers of 2 have good separation properties
    powers_of_2 = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

    # First pass: add powers of 2 (fast, low risk)
    for x in powers_of_2:
        if time.time() > deadline:
            break
        if len(selected) >= 48:
            break
        if _is_valid_addition(selected, x, dp):
            selected.append(x)
            _update_dp(dp, x, len(selected))

    # Second pass: try remaining numbers in increasing order
    for x in range(1, 513):
        if time.time() > deadline:
            break
        if len(selected) >= 48:
            break
        if x in selected:
            continue
        if _is_valid_addition(selected, x, dp):
            selected.append(x)
            _update_dp(dp, x, len(selected))

    # Third pass: try in decreasing order for diversification
    if time.time() < deadline and len(selected) < 48:
        remaining = [x for x in range(512, 0, -1) if x not in selected]
        for x in remaining:
            if time.time() > deadline:
                break
            if len(selected) >= 48:
                break
            if _is_valid_addition(selected, x, dp):
                selected.append(x)
                _update_dp(dp, x, len(selected))

    return selected


def _update_dp(dp, x, current_n):
    """Update the global DP by adding element x. current_n is the size BEFORE adding x."""
    for k in range(current_n, -1, -1):
        if dp[k]:
            dp[k + 1] |= (dp[k] << x)


def _is_valid_addition(selected, x, dp):
    """
    Check if adding x to selected maintains the no-mean property.
    Uses the precomputed dp for efficiency.
    1. x is not the mean of any subset of selected (size >= 2).
    2. No a in selected is the mean of any subset of selected + {x} (size >= 2) that includes x.
    """
    n = len(selected)

    # Check 1: Is x the mean of a subset of selected?
    # Sum of k elements = k * x, k >= 2.
    if n >= 2:
        for k in range(2, n + 1):
            target = k * x
            if target > 512 * n:
                break
            if (dp[k] >> target) & 1:
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

        if _subset_sum_excluding_exists(selected, a, x, dp):
            return False

    return True


def _subset_sum_excluding_exists(selected, a, x, dp):
    """
    Check if there exists k >= 1 such that sum of k elements from 'selected \ {a}'
    equals (k+1)*a - x.
    
    Strategy: First check if the target sum is achievable at all with k elements from selected.
    If yes, verify it doesn't require a.
    """
    n = len(selected)
    if n == 0:
        return False

    for k in range(1, n):
        target = (k + 1) * a - x
        if target < 0 or target > 512 * (n - 1):
            continue

        # Check if target is achievable with k elements from selected
        if not ((dp[k] >> target) & 1):
            continue

        # Now we need to verify that target is achievable with k elements from selected \ {a}
        # Build DP for selected \ {a}
        others = [y for y in selected if y != a]
        if len(others) < k:
            continue

        # Quick check: if target is achievable only using a, we need to verify
        # Build small DP for others
        m = len(others)
        max_sum = sum(others)
        if target > max_sum:
            continue

        dp_small = [0] * (m + 1)
        dp_small[0] = 1
        for elem in others:
            for kk in range(m - 1, -1, -1):
                if dp_small[kk]:
                    dp_small[kk + 1] |= (dp_small[kk] << elem)

        if (dp_small[k] >> target) & 1:
            return True

    return False
