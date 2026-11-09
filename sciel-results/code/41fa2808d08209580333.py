"""Optimized program for finding the longest interval with no modular inverse pairs."""
# EVOLVE-BLOCK-START

def construct():
    MOD = 1009
    N = 1008  # values are 1..1008

    # Compute modular inverses in O(N)
    inv = [0] * (MOD + 1)
    inv[1] = 1
    for i in range(2, MOD):
        inv[i] = (MOD - MOD // i) * inv[MOD % i] % MOD

    # For each x in 1..1008, determine the pair (a, b) = (min(x, inv[x]), max(x, inv[x]))
    # We need to find the longest interval [s, e] such that no pair (a, b) is fully contained in [s, e].
    # Condition for [s, e] to be valid: for all x, NOT (s <= a_x AND b_x <= e).
    # This is equivalent to: for all x with a_x >= s, we must have b_x > e.
    # So e < min{b_x | a_x >= s}.
    # Let M[s] = min{b_x | a_x >= s}. Then max valid e for start s is M[s] - 1.
    
    # We want to maximize length L(s) = (M[s] - 1) - s + 1 = M[s] - s, subject to M[s] - 1 >= s => M[s] >= s + 1.
    
    # Initialize min_b_for_a[s] = infinity
    INF = 10**9
    min_b_for_a = [INF] * (MOD + 2) # indices 1..1008
    
    for x in range(1, MOD):
        ix = inv[x]
        a = min(x, ix)
        b = max(x, ix)
        if b < min_b_for_a[a]:
            min_b_for_a[a] = b

    # Compute suffix minimums M[s]
    M = [INF] * (MOD + 2)
    current_min = INF
    for s in range(MOD, 0, -1):
        if min_b_for_a[s] < current_min:
            current_min = min_b_for_a[s]
        M[s] = current_min

    best_len = 0
    best_start = 1
    
    # Check all possible starts s in 1..1008
    for s in range(1, MOD):
        e_max = M[s] - 1
        if e_max >= s:
            length = e_max - s + 1
            if length > best_len:
                best_len = length
                best_start = s

    if best_len > 0:
        return [best_start, best_len]
    else:
        # Fallback: return a valid single element if possible
        return [2, 1]

# EVOLVE-BLOCK-END
