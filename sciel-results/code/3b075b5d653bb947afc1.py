"""Find longest interval in [1,1008] with no modular inverse pair (including self)."""
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s wall-clock limit

    MOD = 1009
    N = 1008

    # Precompute modular inverses: inv[i] = i^(-1) mod MOD
    inv = [0] * (N + 1)
    for i in range(1, N + 1):
        inv[i] = pow(i, MOD - 2, MOD)

    # Two-pointer sweep to find the longest valid interval [s, e]
    # A window [s, e] is valid iff for all x in [s, e], inv[x] is not in [s, e].
    # When we extend e to e+1, the new element e+1 conflicts with inv[e+1].
    # We need to ensure inv[e+1] is not in [s, e+1].
    # Since inv[e+1] is a specific value, if inv[e+1] is in [s, e+1],
    # we must move s past inv[e+1] (i.e., s = inv[e+1] + 1).
    # But we also need to ensure no OTHER element in the window conflicts.
    # Key property: the window was valid for [s, e]. Adding e+1 only creates
    # a new potential conflict between e+1 and inv[e+1].
    # So the only thing we need to check is: is inv[e+1] in [s, e+1]?
    # If inv[e+1] < s, no problem.
    # If s <= inv[e+1] <= e, we need to move s to inv[e+1]+1.
    # If inv[e+1] == e+1, that's a self-inverse, so we must move s to e+2.
    # Wait, if inv[e+1] == e+1, then (e+1)^2 % MOD == 1, meaning e+1 is self-inverse.
    # In that case, the element e+1 itself conflicts with itself, so we can't include it.
    # We must move s to e+2 (empty window, then next step).

    best_start = 1
    best_len = 0
    s = 1

    for e in range(1, N + 1):
        ie = inv[e]
        # Check if ie is in [s, e]
        if s <= ie <= e:
            # Need to move s past ie
            s = ie + 1
        # Now check if e itself is self-inverse: e*e % MOD == 1
        # This means inv[e] == e, which was handled above (ie == e, so s = e+1)
        # After adjustment, window is [s, e]
        if s <= e:
            length = e - s + 1
            if length > best_len:
                best_len = length
                best_start = s

        # Check deadline
        if time.time() > deadline:
            break

    if best_len == 0:
        # Should not happen, but fallback
        return [2, 1]

    return [best_start, best_len]
