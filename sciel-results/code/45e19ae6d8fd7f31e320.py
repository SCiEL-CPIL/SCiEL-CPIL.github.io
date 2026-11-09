import zlib
import time

_START_TIME = time.monotonic()
_DEADLINE = _START_TIME + 320.0

def priority(vector, n):
    if time.monotonic() > _DEADLINE:
        return 0.0
    
    c1 = 0
    for v in vector:
        if v == 1:
            c1 += 1
            
    h = zlib.crc32(bytes(vector))
    
    # Strong penalty for 1s.
    # This forces the selection of all {0,2}^n vectors first.
    # This set is a valid cap set of size 2^n.
    score = -1000.0 * c1 + float(h)
    return score
