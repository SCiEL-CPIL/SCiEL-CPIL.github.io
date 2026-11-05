import time

def construct():
    p = 1009
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Precompute: for each quadratic residue r, store one root
    # Also build a bitmask where bit v is set if v is a QR
    roots = {}
    mask = 0
    for x in range(1, p):
        r = (x * x) % p
        if r not in roots:
            roots[r] = x
            mask |= (1 << r)
    
    # For each s from 1 to 1004, check if s..s+4 are all QRs
    # Using bitmask: (mask >> s) & 0x1F == 0x1F means bits s..s+4 are all set
    for s in range(1, 1005):
        if time.time() > deadline:
            raise RuntimeError("Deadline exceeded")
        
        if (mask >> s) & 0x1F == 0x1F:
            rs = [roots[s + i] for i in range(5)]
            return {'start': s, 'roots': rs}
    
    raise RuntimeError("No valid start found")
