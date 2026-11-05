"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    MOD = 1009
    N = 1008

    # Precompute modular inverses for 1..1008
    inv = [0] * (N + 1)
    for x in range(1, N + 1):
        inv[x] = pow(x, MOD - 2, MOD)

    best = [2, 1]  # fallback: [2,1] is valid (2*2=4 != 1)

    # For each start L, find max R such that [L, R] is valid
    # [L, R] valid iff for all x in [L, R], inv[x] < L or inv[x] > R
    for L in range(1, N + 1):
        if time.time() > deadline - 2:
            break
        # If L is self-inverse, it can't be in any valid interval
        if inv[L] == L:
            continue
        
        # Extend R from L upward
        max_R = L - 1
        for R in range(L, N + 1):
            # Check if adding R creates a violation
            # Need: inv[R] not in [L, R]
            if L <= inv[R] <= R:
                break
            max_R = R
        
        length = max_R - L + 1
        if length > best[1]:
            best = [L, length]

    return best
# EVOLVE-BLOCK-END
