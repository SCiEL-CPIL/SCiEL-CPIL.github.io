import time

def construct():
    p = 1009
    m = 5
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 55.0

    # is_res[r] = True if r is a non-zero quadratic residue mod p
    # root[r] = smallest x in [1, p-1] such that x^2 % p == r
    is_res = [False] * p
    root = [0] * p

    for x in range(1, p):
        r = (x * x) % p
        # We want the smallest root, so we only update if root[r] is 0
        # Or since x increases, the first x found is the smallest.
        if root[r] == 0:
            root[r] = x
            is_res[r] = True

    # Search for the smallest s such that s, s+1, ..., s+m-1 are all residues
    # s must be in [1, p-m] because s+m-1 <= p-1 (1008)
    # If s+m-1 >= p, it wraps around. The problem says "wraparound are excluded".
    # So s+i must be in [1, 1008].
    # Max s is 1008 - 4 = 1004.
    
    for s in range(1, p - m + 1):
        valid = True
        for i in range(m):
            if not is_res[s + i]:
                valid = False
                break
        if valid:
            # Found the minimal s
            res_roots = [root[s + i] for i in range(m)]
            return {'start': s, 'roots': res_roots}
            
    # Fallback if no run found (should not happen for p=1009)
    # Return the last possible candidate to satisfy interface, though it might be invalid
    # But we know a run exists.
    s = p - m
    res_roots = [root[s + i] for i in range(m)]
    return {'start': s, 'roots': res_roots}
