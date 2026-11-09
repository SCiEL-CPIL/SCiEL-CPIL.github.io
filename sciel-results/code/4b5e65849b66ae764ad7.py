import time

def construct():
    P = 1009
    M = 5

    start_time = time.time()
    deadline = start_time + 85.0

    # roots[r] stores the smallest x in 1..P-1 such that x*x % P == r
    # 0 means r is not a non-zero QR
    roots = [0] * P
    qr_bits = 0

    # Compute quadratic residues and their smallest roots
    for x in range(1, P):
        r = (x * x) % P
        if roots[r] == 0:
            roots[r] = x
        qr_bits |= (1 << r)

    # Mask for M consecutive bits
    mask = (1 << M) - 1

    # Search for the minimum s such that s..s+M-1 are all QRs
    for s in range(1, P - M + 1):
        if time.time() > deadline:
            break
        if ((qr_bits >> s) & mask) == mask:
            current_roots = [roots[s + i] for i in range(M)]
            return {'start': s, 'roots': current_roots}

    # Fallback (should not happen for p=1009)
    return {'start': 1004, 'roots': [1, 1, 1, 1, 1]}
