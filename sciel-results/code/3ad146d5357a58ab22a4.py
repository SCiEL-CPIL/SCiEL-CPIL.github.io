"""Exact minimum-popcount covering certificate for residues mod 101."""
# EVOLVE-BLOCK-START
import time

_P = 101
_DENOMS = list(range(1, 11))
_INV = [pow(x, -1, _P) for x in _DENOMS]

def construct():
    """Return 101 nonempty lists; row r has distinct denominators in 1..10
    whose modular inverses sum to r mod 101. Minimize the largest row length."""
    t0 = time.time()
    deadline = t0 + 85  # safety margin before 90s kill

    # Precompute for every nonempty subset (mask 1..1023) the residue and popcount.
    # best[residue] = (popcount, mask) -- keep the smallest popcount seen.
    best = [None] * _P

    for mask in range(1, 1 << 10):
        if time.time() > deadline:
            break
        residue = 0
        pop = 0
        for i in range(10):
            if mask >> i & 1:
                residue = (residue + _INV[i]) % _P
                pop += 1
        cur = best[residue]
        if cur is None or pop < cur[0]:
            best[residue] = (pop, mask)

    # Fallback: if any residue is still None, assign an arbitrary nonempty subset
    if any(b is None for b in best):
        for r in range(_P):
            if best[r] is None:
                for mask in range(1, 1 << 10):
                    residue = 0
                    for i in range(10):
                        if mask >> i & 1:
                            residue = (residue + _INV[i]) % _P
                    if residue == r:
                        pop = bin(mask).count('1')
                        best[r] = (pop, mask)
                        break

    # Build result lists from masks
    result = []
    for r in range(_P):
        pop, mask = best[r]
        terms = [_DENOMS[i] for i in range(10) if mask >> i & 1]
        result.append(terms)

    return result
# EVOLVE-BLOCK-END
