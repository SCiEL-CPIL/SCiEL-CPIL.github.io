import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80

    LIMIT = 1009
    MAX_VAL = 1008

    # Precompute modular inverses
    inv = [0] * (LIMIT + 1)
    for i in range(1, LIMIT):
        inv[i] = pow(i, -1, LIMIT)

    # Valid range is [2, 1007] since 1 and 1008 are self-inverse
    # 1*1 % 1009 = 1 and 1008*1008 % 1009 = 1
    MIN_VAL = 2
    MAX_VALID = 1007

    # Build conflict map: for each x in [MIN_VAL, MAX_VALID], inv[x] is its conflict partner
    # We need to find the longest interval [L, R] such that for all x in [L, R], inv[x] is NOT in [L, R]
    
    # Approach: For each possible left endpoint L, find the maximum valid right endpoint R
    # An interval [L, R] is valid iff for all x in [L, R], inv[x] < L or inv[x] > R
    
    # Key insight: For a fixed L, the maximum valid R is determined by:
    # R_max(L) = min over all x in [L, R] such that inv[x] >= L of (inv[x] - 1)
    # But we can compute this more efficiently.
    
    # Alternative: For each x, if inv[x] > x, then x conflicts with inv[x].
    # If we include x in the interval, we cannot include inv[x].
    # So for a fixed L, we scan R from L upward, and if at some R, inv[R] is in [L, R], we must stop before R.
    # Actually, we need: for all x in [L, R], inv[x] not in [L, R].
    
    # Efficient approach: Precompute for each position x, the "blocking" value.
    # For a fixed L, as we extend R, the constraint is that no x in [L, R] has inv[x] in [L, R].
    # Equivalently: for all x in [L, R], either inv[x] < L or inv[x] > R.
    # The tightest constraint on R from a given x is: if inv[x] >= L, then R must be < inv[x].
    # So R_max(L) = min(over x in [L, R_max]) of (inv[x] - 1) where inv[x] >= L, but also bounded by MAX_VALID.
    
    # Let's use a different approach: for each L, binary search or linear scan for max valid R.
    # Since N is only 1006, even O(N^2) is fine.
    
    best_len = 0
    best_start = MIN_VAL

    for L in range(MIN_VAL, MAX_VALID + 1):
        if time.time() > deadline:
            break
        # Find max R >= L such that [L, R] is valid
        # [L, R] is valid iff for all x in [L, R], inv[x] < L or inv[x] > R
        # Start with R = L, extend as far as possible
        R = L
        valid = True
        # Check current [L, R]
        # Actually, let's extend R step by step
        for R in range(L, MAX_VALID + 1):
            # Check if adding R keeps the interval valid
            # We need inv[R] not in [L, R]
            if L <= inv[R] <= R:
                # Adding R creates a conflict
                break
            # Also need to check: does any existing x in [L, R-1] now have inv[x] == R?
            # No, because if inv[x] == R, then x == inv[R], and we already checked inv[R] not in [L, R].
            # Wait, that's not quite right. We need ALL pairs to be valid.
            # The condition is: for all x, y in [L, R], x*y % 1009 != 1.
            # This is equivalent to: for all x in [L, R], inv[x] not in [L, R].
            # When we extend from [L, R-1] to [L, R], we only need to check:
            # 1. inv[R] not in [L, R] (new element R's partner)
            # 2. For all x in [L, R-1], inv[x] != R (but this is the same as inv[R] != x for x in [L, R-1])
            # Since inv is an involution (inv[inv[x]] = x), condition 2 is equivalent to inv[R] not in [L, R-1].
            # So condition 1 (inv[R] not in [L, R]) covers it.
            valid = True
            # But wait, we broke out if inv[R] is in [L, R], so if we're here, inv[R] is not in [L, R].
            # But we also need to make sure that previous elements are still valid.
            # Actually, if [L, R-1] was valid, and inv[R] is not in [L, R], then [L, R] is valid.
            # Because the only new pairs are (R, x) for x in [L, R-1], and R*x % 1009 = 1 iff x = inv[R].
            # Since inv[R] is not in [L, R], no such x exists.
            pass
        # After the inner loop, R is one past the max valid endpoint, or we went through all
        # If we broke, R is the first invalid endpoint, so max valid is R-1
        # If we didn't break, max valid is MAX_VALID
        # Let me redo this more carefully
        pass

    # Let me rewrite the inner logic properly
    best_len = 0
    best_start = MIN_VAL

    for L in range(MIN_VAL, MAX_VALID + 1):
        if time.time() > deadline:
            break
        max_R = L - 1
        for R in range(L, MAX_VALID + 1):
            # Check if [L, R] is valid
            # It's valid iff for all x in [L, R], inv[x] not in [L, R]
            # We can check incrementally: [L, R] valid iff [L, R-1] valid AND inv[R] not in [L, R]
            if R == L:
                # Single element, always valid unless self-inverse (but we excluded 1 and 1008)
                max_R = R
            else:
                # Check if inv[R] is in [L, R]
                if L <= inv[R] <= R:
                    break
                max_R = R
        
        length = max_R - L + 1
        if length > best_len:
            best_len = length
            best_start = L

    if best_len == 0:
        best_start = MIN_VAL
        best_len = 1

    return [best_start, best_len]
