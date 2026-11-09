"""Rewritten program: exact min-popcount covering certificate for residues mod 101."""
# EVOLVE-BLOCK-START
import time

_P = 101
_DENOMS = list(range(1, 11))
_INV = [pow(x, -1, _P) for x in _DENOMS]

def construct():
    """Return 101 nonempty lists; row r has distinct denominators in 1..10
    whose modular inverses sum to r mod 101. Minimize the largest row length."""
    t0 = time.time()
    deadline = t0 + 80  # safety margin before 90s kill

    # Precompute for every nonempty subset (mask 1..1023) the residue and popcount.
    # There are only 1023 subsets, so this is trivially fast.
    # best[residue] = (popcount, terms_list)  -- keep the smallest popcount seen.
    best = [None] * _P  # index 0..100; None means not yet assigned

    for mask in range(1, 1 << 10):
        if time.time() > deadline:
            break
        # Compute residue and popcount
        residue = 0
        pop = 0
        terms = []
        for i in range(10):
            if mask >> i & 1:
                residue = (residue + _INV[i]) % _P
                pop += 1
                terms.append(_DENOMS[i])
        cur = best[residue]
        if cur is None or pop < cur[0]:
            best[residue] = (pop, terms)

    # Fallback: if any residue is still None, assign an arbitrary nonempty subset
    if any(b is None for b in best):
        full_map = {}
        for mask in range(1, 1 << 10):
            residue = 0
            pop = 0
            terms = []
            for i in range(10):
                if mask >> i & 1:
                    residue = (residue + _INV[i]) % _P
                    pop += 1
                    terms.append(_DENOMS[i])
            full_map.setdefault(residue, (pop, terms))
        for r in range(_P):
            if best[r] is None:
                best[r] = full_map[r]

    return [best[r][1] for r in range(_P)]
# EVOLVE-BLOCK-END
