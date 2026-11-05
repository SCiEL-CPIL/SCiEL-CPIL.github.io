import time

P = 1009
M = 5

def construct():
    start_time = time.time()
    deadline = start_time + 5.0

    # 1. Pre-compute quadratic residues and their roots
    root_map = [0] * P
    qr_mask = 0

    for x in range(1, P):
        r = (x * x) % P
        if root_map[r] == 0:
            root_map[r] = x
        qr_mask |= (1 << r)

    # 2. Search for the minimum start s in [1, P-M]
    best_s = -1
    best_roots = []
    
    # Mask for 5 consecutive bits
    five_consecutive = 0x1F  # 11111 in binary

    for s in range(1, P - M + 1):
        if time.time() > deadline:
            break
        # Check if s, s+1, ..., s+4 are all QRs using bitmask
        if ((qr_mask >> s) & five_consecutive) == five_consecutive:
            best_s = s
            best_roots = [root_map[s], root_map[s + 1], root_map[s + 2], root_map[s + 3], root_map[s + 4]]
            break

    if best_s != -1:
        return {'start': best_s, 'roots': best_roots}

    return {'start': 1, 'roots': [1, 1, 1, 1, 1]}
