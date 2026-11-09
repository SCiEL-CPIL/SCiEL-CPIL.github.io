import time

_P = 1009
_M = 5

def _precompute():
    """
    Precomputes a boolean array indicating which numbers are non-zero quadratic residues
    and a dictionary mapping each residue to one of its roots.
    """
    is_qr = [False] * _P
    roots = {}
    
    # 0 is a residue (0^2=0) but we exclude it.
    # We iterate x from 1 to P-1
    for x in range(1, _P):
        r = (x * x) % _P
        if r != 0:
            is_qr[r] = True
            # Store the first root found
            if r not in roots:
                roots[r] = x
    return is_qr, roots

# Precompute once at module load
_IS_QR, _ROOTS = _precompute()

def construct():
    """
    Finds the smallest start s such that s, s+1, ..., s+4 are all non-zero quadratic residues mod 1009.
    Returns a dictionary with 'start' and 'roots'.
    """
    start_time = time.monotonic()
    deadline = start_time + 55.0
    
    # s can be at most 1004 so that s+4 <= 1008 < 1009
    # If s=1005, s+4=1009=0, which is excluded.
    max_s = _P - _M + 1  # 1005? No.
    # s+i <= 1008 for i=4 => s <= 1004.
    # So range is 1 to 1004 inclusive.
    
    for s in range(1, 1005):
        if time.monotonic() > deadline:
            # Fallback: return the best found? No, we break on first found.
            # If we didn't find one, raise error.
            raise RuntimeError("Deadline exceeded")

        # Check if s, s+1, s+2, s+3, s+4 are all QRs
        if (_IS_QR[s] and
            _IS_QR[s + 1] and
            _IS_QR[s + 2] and
            _IS_QR[s + 3] and
            _IS_QR[s + 4]):
            
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
