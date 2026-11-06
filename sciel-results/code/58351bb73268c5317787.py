"""Find the least run-start s for 5 consecutive non-zero quadratic residues mod 1009."""
import time

_P = 1009
_M = 5


def _build_table():
    """Map each non-zero quadratic residue to one witness root in [1, P-1]."""
    root = {}
    for x in range(1, _P):
        r = (x * x) % _P
        if r != 0 and r not in root:
            root[r] = x
    return root


def _verify(result):
    s = result['start']
    roots = result['roots']
    if not (1 <= s <= _P - _M):
        return False
    if len(roots) != _M:
        return False
    for i in range(_M):
        v = s + i
        if v == 0 or v >= _P:
            return False
        ri = roots[i]
        if not (1 <= ri <= _P - 1):
            return False
        if (ri * ri) % _P != v:
            return False
    return True


def construct():
    t0 = time.time()
    deadline = t0 + 80.0  # safety margin before 90s kill

    # Build the quadratic residue table: residue -> smallest root
    roots = _build_table()

    # Construct a bitset where bit i is 1 if i is a non-zero quadratic residue.
    # We use bits 1 to 1008. Bit 0 is 0.
    res_bits = 0
    for r in roots:
        res_bits |= (1 << r)

    # We need s, s+1, s+2, s+3, s+4 all to be non-zero quadratic residues mod 1009.
    # Search for the smallest s in [1, 1004].
    # Check condition: (res_bits >> s) & 0b11111 == 0b11111
    
    # Direct ascending search using bitset check
    for s in range(1, 1005):
        if time.time() > deadline:
            break
        # Check if s..s+4 are all set in res_bits
        if (res_bits >> s) & 0b11111 == 0b11111:
            result = {'start': s, 'roots': [roots[s + i] for i in range(_M)]}
            # Verify before returning
            if _verify(result):
                return result

    # Fallback to backward pass if not found (shouldn't happen)
    for s in range(1004, 0, -1):
        if time.time() > deadline:
            break
        if (res_bits >> s) & 0b11111 == 0b11111:
            result = {'start': s, 'roots': [roots[s + i] for i in range(_M)]}
            if _verify(result):
                return result

    raise RuntimeError('No run found')
