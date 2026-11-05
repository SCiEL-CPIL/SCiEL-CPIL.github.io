import time

def construct():
    p = 1009
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin

    # Build bitmask of non-zero quadratic residues mod p
    mask = 0
    roots = {}
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            mask |= (1 << r)
            if r not in roots:
                roots[r] = x
        if time.time() > deadline:
            break

    # Scan s from 1 to 1004 (ascending) to find minimal s
    best_s = None
    best_roots = None
    for s in range(1, 1005):
        if time.time() > deadline:
            break
        if ((mask >> s) & 0x1F) == 0x1F:
            r0 = s
            r1 = s + 1
            r2 = s + 2
            r3 = s + 3
            r4 = s + 4
            return {'start': s, 'roots': [roots[r0], roots[r1], roots[r2], roots[r3], roots[r4]]}
    
    if best_s is not None:
        return {'start': best_s, 'roots': best_roots}
        
    raise RuntimeError('No run of 5 consecutive non-zero quadratic residues found')
