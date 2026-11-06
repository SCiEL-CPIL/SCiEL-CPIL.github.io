"""Find the least run-start s for 5 consecutive non-zero quadratic residues mod 1009."""
import time

_P = 1009
_M = 5
_TIMEOUT = 60.0


def construct():
    """
    Return {'start': s, 'roots': [r0, ..., r4]} where:
      - 1 <= s <= 1004
      - 1 <= ri <= 1008
      - ri**2 mod 1009 == s + i for i in 0..4
      - s is minimized
      - zero residues and wraparound are excluded
    """
    start_time = time.time()
    
    # Build a boolean array marking which values in [1, P-1] are quadratic residues
    # and a mapping from each non-zero quadratic residue to one witness root in [1, P-1]
    is_residue = [False] * _P
    roots = [0] * _P
    
    for x in range(1, _P):
        r = (x * x) % _P
        if r != 0:
            is_residue[r] = True
            if roots[r] == 0:
                roots[r] = x
    
    # Ascending search to find the minimum s
    # Since s >= 1 and s <= P - M = 1004, we have s+i <= 1008 < 1009
    # and s+i >= 1, so no wraparound and no zero residue is possible
    for s in range(1, _P - _M + 1):
        if time.time() - start_time > _TIMEOUT - 1.0:
            # Safety guard; this loop is very fast so we should not hit this
            break
        
        # Check that s, s+1, ..., s+4 are all quadratic residues
        if all(is_residue[s + i] for i in range(_M)):
            return {'start': s, 'roots': [roots[s + i] for i in range(_M)]}
    
    raise RuntimeError('No run found')
