"""Rewritten program: exact min-popcount covering certificate for residues mod 101."""
# EVOLVE-BLOCK-START
import time

_P = 101
_DENOMS = list(range(1, 11))
_INV = [pow(x, -1, _P) for x in _DENOMS]
_NDEN = 10
_NMASKS = 1 << _NDEN

# Precompute at module level for speed: residue and popcount for each mask
_MASK_RESIDUE = [0] * _NMASKS
_MASK_POP = [0] * _NMASKS
_MASK_TERMS = [None] * _NMASKS

for _mask in range(1, _NMASKS):
    _res = 0
    _pop = 0
    _terms = []
    _m = _mask
    _i = 0
    while _m:
        if _m & 1:
            _res = (_res + _INV[_i]) % _P
            _pop += 1
            _terms.append(_DENOMS[_i])
        _i += 1
        _m >>= 1
    _MASK_RESIDUE[_mask] = _res
    _MASK_POP[_mask] = _pop
    _MASK_TERMS[_mask] = _terms

def construct():
    """Return 101 nonempty lists; row r has distinct denominators in 1..10
    whose modular inverses sum to r mod 101. Minimize the largest row length."""
    t0 = time.time()
    deadline = t0 + 80  # safety margin before 90s kill

    # best[residue] = (popcount, terms_list)
    best = [None] * _P

    for mask in range(1, _NMASKS):
        if time.time() > deadline:
            break
        residue = _MASK_RESIDUE[mask]
        pop = _MASK_POP[mask]
        terms = _MASK_TERMS[mask]
        cur = best[residue]
        if cur is None or pop < cur[0]:
            best[residue] = (pop, terms)

    # Verify all residues are covered
    for r in range(_P):
        if best[r] is None:
            # Fallback: should not happen, but handle gracefully
            best[r] = (1, [_DENOMS[0]])

    # Final validation: ensure each row is valid
    result = [best[r][1] for r in range(_P)]
    for r, terms in enumerate(result):
        s = sum(_INV[d - 1] for d in terms) % _P
        assert s == r, f"Row {r}: got sum {s}, expected {r}"
        assert len(terms) == len(set(terms)), f"Row {r}: duplicate denominators"
        assert all(1 <= d <= 10 for d in terms), f"Row {r}: out of range"
        assert len(terms) > 0, f"Row {r}: empty"

    return result
# EVOLVE-BLOCK-END
