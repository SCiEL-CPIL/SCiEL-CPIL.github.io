"""Minimal run-start s for 5 consecutive non-zero quadratic residues mod 1009.

r(k,m,p) with k=2 (quadratic), m=5 (run length), p=1009.
Return {'start':s,'roots':[r0..r4]} where r_i^2 mod 1009 == s+i, s in [1,1004].
Minimize s; zero residues and wraparound excluded.
"""
import time

_P = 1009
_START = time.time()
_DEADLINE = 90.0
_MARGIN = 5.0


def _elapsed():
    return time.time() - _START


def construct():
    # Map residue -> one root r in [1, P-1] with r^2 % P == residue (skip residue 0).
    root = [0] * _P
    for x in range(1, _P):
        r = (x * x) % _P
        if r != 0 and root[r] == 0:
            root[r] = x

    # Build a bitset: bit v is 1 iff v (1..1008) is a non-zero quadratic residue.
    bitmask = 0
    for v in range(1, _P):
        if root[v] != 0:
            bitmask |= (1 << v)

    # Compute consecutive mask: bit s set iff bits s..s+4 all set in bitmask.
    consecutive = bitmask
    for i in range(1, 5):
        consecutive &= (bitmask >> i)

    # Valid s range: 1 <= s <= 1004 (so s+4 <= 1008, no wraparound).
    # Build valid mask for positions 1..1004.
    valid_mask = 0
    for s in range(1, _P - 4):
        valid_mask |= (1 << s)

    candidates = consecutive & valid_mask

    if candidates == 0:
        # Fallback: linear scan
        max_s = _P - 5
        for s in range(1, max_s + 1):
            if (bitmask >> s) & 0b11111 == 0b11111:
                roots = [root[s + i] for i in range(5)]
                return {'start': s, 'roots': roots}
            if _elapsed() > _DEADLINE - _MARGIN:
                break
        raise RuntimeError('No 5-run of non-zero quadratic residues found modulo 1009')

    # Find the least set bit in candidates.
    least_bit = candidates & (-candidates)
    s = least_bit.bit_length() - 1

    # Verify s is in valid range; fallback to linear scan if something is off.
    if s < 1 or s > _P - 5:
        max_s = _P - 5
        for s in range(1, max_s + 1):
            if (bitmask >> s) & 0b11111 == 0b11111:
                roots = [root[s + i] for i in range(5)]
                return {'start': s, 'roots': roots}
        raise RuntimeError('No valid 5-run found')

    # Cross-check with a linear scan to ensure we found the true minimum.
    # This is cheap (at most ~1004 iterations) and guards against any bit-manipulation bug.
    for s_check in range(1, _P - 4):
        if (bitmask >> s_check) & 0b11111 == 0b11111:
            if s_check < s:
                s = s_check
            break

    roots = [root[s + i] for i in range(5)]
    return {'start': s, 'roots': roots}
