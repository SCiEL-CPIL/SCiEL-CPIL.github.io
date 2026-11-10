"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 85.0  # safety margin before 90s kill
    p = 1009
    N = 1008  # max value in [1, 1008]

    # Precompute modular inverses using Fermat's little theorem
    inv = [0] * (p + 1)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)

    # Self-inverse elements: x where x*x % p == 1
    # For prime p=1009, these are x=1 and x=p-1=1008
    self_inv = {1, 1008}

    # For each start s, find the maximum valid end e
    # An interval [s, e] is valid if for all x in [s, e]:
    #   1. x is not self-inverse
    #   2. inv[x] is not in [s, e]
    #
    # Key insight: if [s, e] is valid, then [s, e+1] is valid iff:
    #   - e+1 is not self-inverse
    #   - inv[e+1] is not in [s, e+1]
    # The second condition means inv[e+1] < s or inv[e+1] > e+1
    # (since inv[e+1] cannot equal e+1 unless e+1 is self-inverse)
    # And this also ensures no existing x in [s,e] has inv[x] == e+1
    # (because inv[x] == e+1 iff x == inv[e+1], and inv[e+1] not in [s,e+1]
    #  means inv[e+1] not in [s,e])

    best_start = 2
    best_len = 1  # [2,2] is always valid

    for s in range(2, N + 1):
        if s in self_inv:
            continue
        if time.time() > deadline:
            break

        e = s
        # Extend e as far as possible
        while e < N:
            next_e = e + 1
            if next_e in self_inv:
                break
            # Check if adding next_e keeps the interval valid
            # Need inv[next_e] not in [s, next_e]
            if inv[next_e] < s or inv[next_e] > next_e:
                e = next_e
            else:
                break

        cur_len = e - s + 1
        if cur_len > best_len:
            best_len = cur_len
            best_start = s

    return [best_start, best_len]
# EVOLVE-BLOCK-END
