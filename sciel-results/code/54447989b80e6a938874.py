import time

P = 1009
M = 5

def construct():
    start_time = time.time()
    deadline = start_time + 5.0

    # 1. Pre-compute quadratic residues and their roots
    is_qr = [False] * P
    root_map = [0] * P

    for x in range(1, P):
        r = (x * x) % P
        if not is_qr[r]:
            is_qr[r] = True
            root_map[r] = x

    # 2. Search for the minimum start s in [1, P-M]
    best_s = -1
    best_roots = []

    for s in range(1, P - M + 1):
        if time.time() > deadline:
            break
        # Check if s, s+1, ..., s+4 are all QRs
        if is_qr[s] and is_qr[s + 1] and is_qr[s + 2] and is_qr[s + 3] and is_qr[s + 4]:
            best_s = s
            best_roots = [root_map[s], root_map[s + 1], root_map[s + 2], root_map[s + 3], root_map[s + 4]]
            break

    if best_s != -1:
        return {'start': best_s, 'roots': best_roots}

    return {'start': 1, 'roots': [1, 1, 1, 1, 1]}
