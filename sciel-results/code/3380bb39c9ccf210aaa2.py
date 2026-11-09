import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # safety margin before 60s limit
    
    p = 1009
    
    # Precompute one valid root for each non-zero quadratic residue
    # Using a direct array for O(1) lookup
    roots = [0] * p  # roots[r] = x such that x*x % p == r, 0 if not a residue
    
    for x in range(1, p):
        r = (x * x) % p
        if r != 0 and roots[r] == 0:
            roots[r] = x
    
    # Search for the minimal s such that s, s+1, s+2, s+3, s+4 are all non-zero quadratic residues
    # with s >= 1 and s+4 <= 1008 (i.e., s <= 1004)
    # and none of s+i equals 0 mod 1009 (which is automatically satisfied since 1 <= s+i <= 1008)
    
    for s in range(1, 1005):
        if time.time() > deadline:
            break
        # Check if all five consecutive values are non-zero quadratic residues
        ok = True
        for i in range(5):
            val = s + i
            if roots[val] == 0:
                ok = False
                break
        if ok:
            return {'start': s, 'roots': [roots[s + i] for i in range(5)]}
    
    # Fallback: should not happen, but return a known valid result
    # Find any valid run
    for s in range(1004, 0, -1):
        ok = True
        for i in range(5):
            val = s + i
            if roots[val] == 0:
                ok = False
                break
        if ok:
            return {'start': s, 'roots': [roots[s + i] for i in range(5)]}
    
    raise RuntimeError('No valid run found')
