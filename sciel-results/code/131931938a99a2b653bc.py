import time

_P = 1009
_M = 5

def _precompute():
    """
    Precomputes a mapping from non-zero quadratic residues to one of their square roots.
    """
    roots = {}
    for x in range(1, _P):
        r = (x * x) % _P
        # We only care about non-zero residues
        if r != 0:
            # Store the first root found for each residue
            if r not in roots:
                roots[r] = x
    return roots

# Precompute once at module load to ensure it's ready for construct()
_ROOTS = _precompute()
_ROOT_SET = set(_ROOTS.keys())

def construct():
    """
    Finds the smallest start s such that s, s+1, ..., s+4 are all non-zero quadratic residues mod 1009.
    Returns a dictionary with 'start' and 'roots'.
    """
    start_time = time.monotonic()
    deadline = start_time + 55.0
    
    max_s = _P - _M + 1  # s can be at most 1004 so that s+4 <= 1008 < 1009
    
    for s in range(1, max_s):
        # Safety check for time, though this loop is very fast
        if time.monotonic() > deadline:
            raise RuntimeError("Deadline exceeded")

        # Check if s, s+1, s+2, s+3, s+4 are all in the set of quadratic residues
        # Using 'in' on a set is O(1)
        if (s in _ROOT_SET and
            (s + 1) in _ROOT_SET and
            (s + 2) in _ROOT_SET and
            (s + 3) in _ROOT_SET and
            (s + 4) in _ROOT_SET):
            
            # Construct the result
            return {
                'start': s,
                'roots': [
                    _ROOTS[s],
                    _ROOTS[s + 1],
                    _ROOTS[s + 2],
                    _ROOTS[s + 3],
                    _ROOTS[s + 4]
                ]
            }
            
    raise RuntimeError("No run of 5 consecutive quadratic residues found")
