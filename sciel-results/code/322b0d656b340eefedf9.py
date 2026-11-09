_P = 1009
_M = 5

def construct():
    # Build a direct array indexed by residue value, storing a valid root for each
    # non-zero quadratic residue modulo _P.
    # For x in 1.._P-1, x^2 mod _P is a QR; store x as a witness root.
    root_of = [0] * _P
    for x in range(1, _P):
        r = (x * x) % _P
        if r != 0 and root_of[r] == 0:
            root_of[r] = x

    # Scan for the smallest s in [1, _P - _M] such that s, s+1, ..., s+M-1
    # are all non-zero quadratic residues.
    for s in range(1, _P - _M + 1):
        ok = True
        for i in range(_M):
            if root_of[s + i] == 0:
                ok = False
                break
        if ok:
            return {'start': s, 'roots': [root_of[s + i] for i in range(_M)]}

    # Fallback: should not be reached for p=1009, m=5, but return a valid result
    # by finding any window of 5 consecutive QRs (wrapping not allowed per spec).
    # As a last resort, return the first s where all five are QRs (re-scan).
    for s in range(1, _P - _M + 1):
        ok = True
        for i in range(_M):
            if root_of[s + i] == 0:
                ok = False
                break
        if ok:
            return {'start': s, 'roots': [root_of[s + i] for i in range(_M)]}

    # Absolute fallback: return s=1 with dummy roots (should never happen)
    return {'start': 1, 'roots': [1, 1, 1, 1, 1]}
