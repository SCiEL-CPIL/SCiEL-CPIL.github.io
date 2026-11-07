import time

def _is_valid_bitset(arr):
    """Check if subset sums of different cardinalities are disjoint using bitsets."""
    n = len(arr)
    if n == 0:
        return True
    sums_by_card = [0] * (n + 1)
    sums_by_card[0] = 1
    for x in arr:
        for j in range(n - 1, -1, -1):
            if sums_by_card[j]:
                sums_by_card[j + 1] |= (sums_by_card[j] << x)
    for j in range(n + 1):
        for k in range(j + 1, n + 1):
            if sums_by_card[j] & sums_by_card[k]:
                return False
    return True

def _validate(arr):
    if len(arr) != len(set(arr)):
        return False
    if any(v < 1 or v > 256 for v in arr):
        return False
    return _is_valid_bitset(arr)

def _ap_solution(k):
    """Return AP of size k if it fits in 1..256 and is valid, else None."""
    if k == 1:
        return [1]
    max_j_val = max(j * (k - 1 - j) for j in range(1, k))
    required_x_1 = max_j_val + 1
    if required_x_1 <= 256 - k + 1:
        x_1 = required_x_1
        set_vals = list(range(x_1, x_1 + k))
        if len(set(set_vals)) == k and all(1 <= v <= 256 for v in set_vals):
            if _is_valid_bitset(set_vals):
                return set_vals
    return None

def construct():
    deadline = time.time() + 75
    
    # Find the largest k for which an AP solution exists and is valid
    for k in range(60, 0, -1):
        sol = _ap_solution(k)
        if sol is not None:
            return list(sol)
    
    return [1]
