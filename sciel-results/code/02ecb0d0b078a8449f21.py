import time

_P = 1009
_M = 5

def construct():
    start_time = time.time()
    deadline = start_time + 60.0  # safety margin before 90s kill

    # Precompute quadratic residues mod 1009
    # is_res[r] = True if r is a non-zero quadratic residue mod _P
    # root_of[r] = a witness x such that x*x % _P == r
    is_res = [False] * _P
    root_of = [0] * _P
    
    for x in range(1, _P):
        r = (x * x) % _P
        if r != 0 and not is_res[r]:
            is_res[r] = True
            root_of[r] = x

    # We need to find the minimum s in [1, 1004] such that
    # s, s+1, s+2, s+3, s+4 are all non-zero quadratic residues mod 1009
    # Note: since we're looking at s+i where i is 0..4 and s <= 1004,
    # s+i can be up to 1008, which is within [1, 1008].
    # No wraparound needed since s+i <= 1008 < 1009.
    
    # Ascending search to find the minimum s
    for s in range(1, _P - _M + 1):  # s from 1 to 1004
        if time.time() > deadline:
            # Should not happen for this small problem, but safety guard
            break
        # Check if s, s+1, ..., s+4 are all quadratic residues
        ok = True
        for i in range(_M):
            val = s + i
            if val >= _P or not is_res[val]:
                ok = False
                break
        if ok:
            return {'start': s, 'roots': [root_of[s + i] for i in range(_M)]}

    raise RuntimeError('No run found')
