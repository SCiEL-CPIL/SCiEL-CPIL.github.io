import time

P = 1009
M = 5

def construct():
    start_time = time.time()
    deadline = start_time + 5.0

    # 1. Identify quadratic residues and non-residues
    # is_qr[r] is True if r is a non-zero quadratic residue mod P
    is_qr = [False] * P
    root_map = [0] * P
    
    for x in range(1, P):
        r = (x * x) % P
        is_qr[r] = True
        if root_map[r] == 0:
            root_map[r] = x

    # 2. Find the minimum start s
    # We scan s from 1 to P-M
    # Using the non-residue gap logic or direct check
    # Direct check is O(1) with is_qr array
    # is_qr[s] and is_qr[s+1] ... and is_qr[s+4]
    
    best_s = -1
    best_roots = []

    for s in range(1, P - M + 1):
        if time.time() > deadline:
            break
        # Check if s..s+4 are all QR
        # Since M is small (5), a simple loop or unrolled check is fast
        if (is_qr[s] and is_qr[s+1] and is_qr[s+2] and is_qr[s+3] and is_qr[s+4]):
            best_s = s
            best_roots = [root_map[s], root_map[s+1], root_map[s+2], root_map[s+3], root_map[s+4]]
            break

    if best_s != -1:
        return {'start': best_s, 'roots': best_roots}

    # Fallback
    return {'start': 1, 'roots': [1, 1, 1, 1, 1]}
