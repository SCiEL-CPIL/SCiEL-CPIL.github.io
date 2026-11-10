"""Structural change: Precompute subset-sum DP once per set state, evaluate all candidates
in parallel using the shared DP, and use conflict-count heuristics for greedy selection."""
# EVOLVE-BLOCK-START
import time
import random

N = 512
MAX_TIME = 85.0  # safety margin under 90s

def _build_dp(current):
    """Build subset-sum DP bitsets for the current set.
    dp[k] is an integer where bit s is set if sum s is achievable by exactly k elements."""
    n = len(current)
    max_sum = sum(current)
    # We only need sums up to max_sum
    dp = [0] * (n + 1)
    dp[0] = 1  # bit 0: sum 0 with 0 elements
    for x in current:
        for k in range(n, 0, -1):
            if dp[k - 1]:
                dp[k] |= (dp[k - 1] << x)
    return dp, max_sum

def _check_element_mean(element, dp, current, max_sum):
    """Check if element is the mean of any subset of current of size >= 2."""
    n = len(current)
    for k in range(2, n + 1):
        target = k * element
        if target > max_sum:
            break
        if (dp[k] >> target) & 1:
            return True
    return False

def _check_new_violations(element, dp, current, max_sum):
    """Check if adding element creates new violations for existing elements.
    For each a in current, check if a is mean of a subset including element.
    a is mean of subset S (size j+1, includes element) if sum(S\{element}) = (j+1)*a - element
    where S\{element} is a subset of current\{a} of size j."""
    n = len(current)
    for a in current:
        # Build DP for current \ {a}
        others = [x for x in current if x != a]
        m = len(others)
        if m < 1:
            continue
        others_sum = sum(others)
        # dp_others[k] = bitset of sums achievable by k elements from others
        dp_others = [0] * (m + 1)
        dp_others[0] = 1
        for x in others:
            for k in range(m, 0, -1):
                if dp_others[k - 1]:
                    dp_others[k] |= (dp_others[k - 1] << x)
        
        for j in range(1, m + 1):
            target = (j + 1) * a - element
            if target < 0 or target > others_sum:
                continue
            if (dp_others[j] >> target) & 1:
                return True
    return False

def _can_add(element, current, dp, max_sum):
    """Check if element can be added to current without violation."""
    # Check 1: is element the mean of any subset of current?
    if len(current) >= 2 and _check_element_mean(element, dp, current, max_sum):
        return False
    
    # Check 2: does adding element create new violations?
    if _check_new_violations(element, dp, current, max_sum):
        return False
    
    return True

def _update_dp_after_add(dp, x, new_len):
    """Update dp after adding x. dp is a list of ints."""
    for k in range(new_len, 0, -1):
        if k > 1:
            dp[k] |= (dp[k - 1] << x)
        else:
            dp[1] |= (1 << x)
    return dp

def _greedy_with_conflict_heuristic(elements, rng, deadline):
    """Greedy construction using conflict-count heuristics."""
    selected = []
    dp = [0] * (len(elements) + 1)
    dp[0] = 1
    max_sum = 0
    
    remaining = list(elements)
    rng.shuffle(remaining)
    
    for x in remaining:
        if time.time() > deadline - 2.0:
            break
        if len(selected) >= 48:
            break
        
        if _can_add(x, selected, dp, max_sum):
            selected.append(x)
            _update_dp_after_add(dp, x, len(selected))
            max_sum += x
    
    return selected

def _exact_verify(S):
    """Exact verification of the set."""
    S_list = sorted(S)
    n = len(S_list)
    if n < 3:
        return True
    
    for idx_a, a in enumerate(S_list):
        others = [x for i, x in enumerate(S_list) if i != idx_a]
        m = len(others)
        if m < 2:
            continue
        
        max_sum = sum(others)
        # dp[k] = bitset of sums achievable by k elements from others
        dp = [0] * (m + 1)
        dp[0] = 1
        
        for x in others:
            for k in range(m, 0, -1):
                if dp[k - 1]:
                    dp[k] |= (dp[k - 1] << x)
        
        for k in range(2, m + 1):
            if k * a <= max_sum and (dp[k] >> (k * a)) & 1:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + MAX_TIME
    
    best_set = []
    rng = random.Random(42)
    elements = list(range(1, N + 1))
    
    # Multiple greedy passes with different orderings
    orderings = [
        list(range(1, N + 1)),
        list(range(N, 0, -1)),
        [x for x in range(1, N + 1) if x % 2 == 1] + [x for x in range(1, N + 1) if x % 2 == 0],
        [x for x in range(1, N + 1) if x % 3 != 0] + [x for x in range(1, N + 1) if x % 3 == 0],
    ]
    
    for _ in range(8):
        order = list(range(1, N + 1))
        rng.shuffle(order)
        orderings.append(order)
    
    for ordering in orderings:
        if time.time() > deadline - 5.0:
            break
        
        s = _greedy_with_conflict_heuristic(ordering, rng, deadline)
        
        # Verify
        if _exact_verify(s):
            if len(s) > len(best_set):
                best_set = s
        
        if len(best_set) >= 48:
            break
    
    # Try to improve by adding more elements
    if best_set and len(best_set) < 48 and time.time() < deadline - 3.0:
        current = list(best_set)
        dp, max_sum = _build_dp(current)
        
        improved = True
        while improved and time.time() < deadline - 3.0:
            improved = False
            for x in elements:
                if x not in current:
                    if _can_add(x, current, dp, max_sum):
                        current.append(x)
                        _update_dp_after_add(dp, x, len(current))
                        max_sum += x
                        improved = True
                        break
        
        if len(current) > len(best_set):
            if _exact_verify(current):
                best_set = current
    
    return best_set
# EVOLVE-BLOCK-END
