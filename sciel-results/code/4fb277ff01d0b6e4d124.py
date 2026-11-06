"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 55.0

    # Precompute inverses of 1..10 mod 101 using Fermat's little theorem
    P = 101
    inv = [0] * 11
    for i in range(1, 11):
        inv[i] = pow(i, P - 2, P)

    # denominators 1..10
    DENS = list(range(1, 11))

    # best[r] stores the list of denominators with minimal size summing to r mod 101
    best = [None] * P
    min_size = [11] * P  # Max possible size is 10, so 11 is infinity

    # Iterate through all non-empty subsets grouped by size for early stopping
    # 10 denominators, so sizes 1..10
    for size in range(1, 11):
        if time.time() > deadline:
            break
        # If all residues already have size <= current size, we can stop
        # But we need to check if there's any residue not yet covered at this size
        # Actually, we can stop if min_size[r] <= size for all r
        # But we're iterating by size ascending, so if all are <= size-1, we're done
        if all(min_size[r] <= size for r in range(P)):
            break

        for combo in combinations(DENS, size):
            if time.time() > deadline:
                break
            residue = sum(inv[d] for d in combo) % P
            if size < min_size[residue]:
                min_size[residue] = size
                best[residue] = list(combo)

    # Construct result
    result = []
    for r in range(P):
        if best[r] is not None:
            result.append(best[r])
        else:
            # This should not happen for p=101, D=10 with 1023 subsets
            # Fallback: find any subset (shouldn't be needed)
            result.append([1])  # Placeholder, but mathematically this case won't occur

    return result

# EVOLVE-BLOCK-END
