_CONSEC_BITSET = None
_ROOTS_LOOKUP = None

def construct():
    global _CONSEC_BITSET, _ROOTS_LOOKUP
    
    if _CONSEC_BITSET is None:
        p = 1009
        bitset = 0
        for x in range(1, p):
            r = (x * x) % p
            if r != 0:
                bitset |= (1 << r)
        
        _CONSEC_BITSET = bitset & (bitset >> 1) & (bitset >> 2) & (bitset >> 3) & (bitset >> 4)
        
        _ROOTS_LOOKUP = {}
        for x in range(1, p):
            r = (x * x) % p
            if r != 0 and r not in _ROOTS_LOOKUP:
                _ROOTS_LOOKUP[r] = x

    # Find min s
    consec = _CONSEC_BITSET
    for s in range(1, 1005):
        if (consec >> s) & 1:
            roots = [_ROOTS_LOOKUP[s + i] for i in range(5)]
            return {'start': s, 'roots': roots}
            
    raise RuntimeError('No run found')
