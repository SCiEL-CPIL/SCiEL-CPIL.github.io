import time

_P = 1009
_START = time.time()
_DEADLINE = 90.0
_MARGIN = 5.0

def construct():
    # Build root map: residue -> smallest root r in [1, P-1] with r^2 % P == residue.
    # We only care about residues 1..1008.
    root = [0] * _P
    for x in range(1, _P):
        r = (x * x) % _P
        if r != 0 and root[r] == 0:
            root[r] = x

    # Use bitset representation for fast 5-consecutive run detection
    # bitset has bit i set iff i is a non-zero quadratic residue
    bitset = 0
    for i in range(1, _P):
        if root[i] != 0:
            bitset |= (1 << i)

    # For each possible start s (1 to 1004), check if bits s..s+4 are all set
    # We can do this by checking: (bitset >> s) & 0b11111 == 0b11111
    # This is a single bit operation per candidate s
    
    # Scan s from 1 upward
    max_s = _P - 5
    for s in range(1, max_s + 1):
        # Check if s, s+1, s+2, s+3, s+4 are all residues using bitset
        if ((bitset >> s) & 0b11111) == 0b11111:
            roots = [root[s + i] for i in range(5)]
            return {'start': s, 'roots': roots}

    raise RuntimeError('No 5-run of non-zero quadratic residues found modulo 1009')
