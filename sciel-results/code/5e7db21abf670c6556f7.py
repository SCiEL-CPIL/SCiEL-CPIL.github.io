"""Find the least run-start s for 5 consecutive non-zero quadratic residues mod 1009."""
import time

_P = 1009
_M = 5


def _build_table():
    """Map each non-zero quadratic residue to one witness root in [1, P-1]."""
    # is_res[i] is 1 if i is a non-zero QR, else 0
    is_res = [0] * _P
    roots = [0] * _P  # roots[i] is a root x such that x^2 % P == i, 0 if none
    
    for x in range(1, _P):
        r = (x * x) % _P
        if r != 0 and roots[r] == 0:
            roots[r] = x
            is_res[r] = 1
            
    # Build prefix sum for is_res
    # prefix[i] = sum(is_res[0], ..., is_res[i-1])
    prefix = [0] * (_P + 1)
    for i in range(1, _P + 1):
        prefix[i] = prefix[i-1] + is_res[i-1]
        
    return is_res, roots, prefix


def _verify(result):
    s = result['start']
    roots = result['roots']
    if not (1 <= s <= _P - _M):
        return False
    if len(roots) != _M:
        return False
    for i in range(_M):
        v = (s + i) % _P
        if v == 0:
            return False
        ri = roots[i]
        if not (1 <= ri <= _P - 1):
            return False
        if (ri * ri) % _P != v:
            return False
    return True


def construct():
    start_wall = time.time()
    deadline = start_wall + 80.0

    is_res, roots, prefix = _build_table()

    # Exact least-start search: ascending s, return the first (smallest) feasible run.
    best = None
    # s ranges from 1 to P-M (inclusive) to ensure s+4 < P
    # Wait, s+i must be <= P-1? 
    # The problem says 1 <= ri <= 1008.
    # And ri^2 mod 1009 == s+i.
    # If s+i == 1009, then v=0. But 0 is excluded ("zero residues ... are excluded").
    # So s+i must be in [1, 1008].
    # Max s+4 = 1008 => s <= 1004.
    # Min s = 1.
    
    for s in range(1, _P - _M + 1):
        # Check if s, s+1, s+2, s+3, s+4 are all residues
        # Sum of is_res[s] to is_res[s+4]
        # prefix[s+5] - prefix[s]
        if prefix[s + _M] - prefix[s] == _M:
            best = {'start': s, 'roots': [roots[s + i] for i in range(_M)]}
            break  # ascending => first hit is the true minimum
            
        if time.time() > deadline:
            break

    if best is not None and _verify(best):
        return best

    # Fallback (should not trigger)
    for s in range(1, _P - _M + 1):
        if all(is_res[s + i] == 1 for i in range(_M)):
            cand = {'start': s, 'roots': [roots[s + i] for i in range(_M)]}
            if _verify(cand):
                return cand
    raise RuntimeError('No feasible run found')
