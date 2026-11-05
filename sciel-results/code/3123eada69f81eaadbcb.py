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

    # Identify self-inverse residues (x where x*x % MOD == 1)
    self_inverse = set(x for x in range(1, N + 1) if inv[x] == x)

    # For each position, find the nearest "conflict" to the right
    # A conflict at position R for interval starting at L exists if inv[R] >= L
    # We want to find for each L the maximum R such that for all x in [L,R], inv[x] < L or inv[x] > R

    # Strategy: For each L, the constraint is that no x in [L, R] has inv[x] in [L, R]
    # This means: for all x in [L, R], either inv[x] < L or inv[x] > R

    # Precompute: for each possible L, what's the first R >= L where inv[R] >= L?
    # Actually, we need: for all x in [L,R], inv[x] is NOT in [L,R]
    # So if inv[R] is in [L, R], we have a violation

    # Better approach: for each L, scan R from L upward, and stop when we find R such that
    # inv[R] is in [L, R]. But we can optimize by noting that once we pass a certain point,
    # we might be able to skip ahead.

    # Let's precompute for each R, the minimum L such that inv[R] < L (i.e., L > inv[R])
    # Or L such that inv[R] > R (which is impossible since inv[R] <= N and R <= N)

    # Actually, the condition for R to be valid in interval [L, R] is:
    # inv[R] < L OR inv[R] > R
    # Since inv[R] is the inverse of R, and R*inv[R] % MOD = 1
    # inv[R] > R is possible when R is small

    # For a given L, R is valid if inv[R] < L or inv[R] > R
    # The interval breaks when we find R where L <= inv[R] <= R

    best = [2, 1]  # fallback

    # For each L, find the maximum valid R
    for L in range(1, N + 1):
        if time.time() > deadline - 2:
            break

        # If L is self-inverse, skip
        if L in self_inverse:
            continue

        # Find max R such that for all x in [L, R], inv[x] < L or inv[x] > R
        # Start with R = L, check if inv[L] is in [L, L] i.e., inv[L] == L
        # We already skipped self-inverse L, so inv[L] != L
        # But we need inv[L] < L or inv[L] > L (which is always true if inv[L] != L)
        # Wait, the condition is inv[x] not in [L, R]
        # For x = L: need inv[L] not in [L, R], i.e., inv[L] < L or inv[L] > R

        # Let's just scan R from L upward
        max_R = L - 1
        for R in range(L, N + 1):
            # Check if R can be added: need inv[R] not in [L, R]
            if L <= inv[R] <= R:
                break
            max_R = R

        length = max_R - L + 1
        if length > best[1]:
            best = [L, length]

    return best
# EVOLVE-BLOCK-END
