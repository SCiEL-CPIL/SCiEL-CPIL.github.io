import time

P = 1009
M = 5

def construct():
    start_time = time.time()
    # 90s limit, but this problem is solved in <1ms. 
    # We set a generous but safe deadline.
    deadline = start_time + 5.0 

    # 1. Pre-compute quadratic residues and their roots
    # is_qr[i] will be True if i is a non-zero quadratic residue mod P
    # root_map[i] will store one valid root r such that r^2 % P == i
    is_qr = [False] * P
    root_map = [0] * P  # 0 indicates no root found yet (or residue 0)

    for x in range(1, P):
        r = (x * x) % P
        if not is_qr[r]:
            is_qr[r] = True
            root_map[r] = x
            
    # 2. Search for the minimum start s
    # s must be in [1, P-M] to ensure s..s+M-1 are in [1, P-1]
    # P-M = 1009-5 = 1004.
    # So s in [1, 1004].
    
    best_s = -1
    best_roots = []
    
    for s in range(1, P - M + 1):
        # Check time just in case, though it's overkill
        if time.time() > deadline:
            break
        
        # Check if s, s+1, ..., s+4 are all QRs
        if all(is_qr[s + i] for i in range(M)):
            best_s = s
            best_roots = [root_map[s + i] for i in range(M)]
            break  # Found the minimum s, so we can stop
    
    if best_s != -1:
        return {'start': best_s, 'roots': best_roots}
    
    # Fallback (should not reach here for this specific prime and M)
    # If no run found, return an invalid structure to signal error, 
    # but the problem guarantees a solution exists for this instance.
    return {'start': 1, 'roots': [1, 1, 1, 1, 1]}
