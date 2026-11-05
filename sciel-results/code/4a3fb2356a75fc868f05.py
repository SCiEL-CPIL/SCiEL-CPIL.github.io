import time

P = 1009
M = 5

def construct():
    start_time = time.time()
    deadline = start_time + 5.0

    # 1. Pre-compute quadratic residues and their roots
    # root_map[r] = x such that x*x % P == r. We store the smallest such x > 0.
    root_map = [0] * P
    is_qr = [False] * P
    mask = 0

    for x in range(1, P):
        r = (x * x) % P
        if not is_qr[r]:
            is_qr[r] = True
            root_map[r] = x
            mask |= (1 << r)

    # 2. Search for the minimum start s in [1, P-M]
    # We check if s, s+1, ..., s+4 are all QRs using bitmask
    # (mask >> s) & 0x1F == 0x1F means bits s, s+1, s+2, s+3, s+4 are set in mask.
    
    # We iterate s from 1 upwards.
    # Max s is P-M because we need s+4 <= P-1 (since residues are 1..P-1, 0 is excluded).
    # Wait, the condition is 1 <= ri <= 1008. The residues are s..s+4.
    # If s+4 > P-1, we might wrap around or hit 0.
    # The problem states "zero residues and wraparound are excluded".
    # So s+4 must be <= P-1. Thus s <= P-5.
    # P=1009, so s <= 1004.
    
    for s in range(1, P - M + 1):
        if time.time() > deadline:
            break
            
        # Check if 5 consecutive bits starting at s are set
        if (mask >> s) & 0x1F == 0x1F:
            # Found the minimal s
            roots = [root_map[s + i] for i in range(M)]
            return {'start': s, 'roots': roots}

    # Fallback (should not happen for P=1009, M=5)
    return {'start': 1, 'roots': [1, 1, 1, 1, 1]}
