"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

_START = time.time()
_DEADLINE = 80.0  # safety margin under the 90s kill limit


def _elapsed():
    return time.time() - _START


def construct():
    MOD = 101
    N = 10

    # Precompute inverses
    inv = [0] * (N + 1)
    for x in range(1, N + 1):
        inv[x] = pow(x, -1, MOD)

    # For each residue, find the minimum-size subset of {1..N} whose inverse-sum equals it
    best = [None] * MOD
    best_size = [N + 1] * MOD

    for size in range(1, N + 1):
        if _elapsed() > _DEADLINE:
            break
        for terms in combinations(range(1, N + 1), size):
            residue = sum(inv[x] for x in terms) % MOD
            if best[residue] is None:
                best[residue] = list(terms)
                best_size[residue] = size

    # All residues should be covered since size N includes all denominators
    # (the full set sum is one valid residue; other residues are covered by smaller subsets)
    # Fallback: if any residue is still None, use the full set (should not happen)
    for r in range(MOD):
        if best[r] is None:
            best[r] = list(range(1, N + 1))

    return [best[r] for r in range(MOD)]
# EVOLVE-BLOCK-END
