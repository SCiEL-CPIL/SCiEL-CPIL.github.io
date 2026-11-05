def construct():
    import time
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill

    P = 1009
    M = 5

    # Precompute quadratic residues modulo P
    # qr_root[x] = smallest positive r such that r^2 % P == x, for x in 1..P-1
    qr_root = [0] * P
    for r in range(1, P):
        x = (r * r) % P
        if x != 0 and qr_root[x] == 0:
            qr_root[x] = r

    # Build a boolean array for which values are valid quadratic residues with known roots
    is_qr = [False] * P
    for x in range(1, P):
        if qr_root[x] != 0:
            is_qr[x] = True

    # Search for the smallest s in [1, P-M] such that s+i is a QR for i=0..M-1
    # Use a sliding window / prefix sum approach for efficiency
    # Count consecutive QRs starting at each position
    
    # Build prefix sum of is_qr to quickly check if a window of length M is all True
    # prefix[i] = sum of is_qr[0..i-1]
    prefix = [0] * (P + 1)
    for i in range(1, P + 1):
        prefix[i] = prefix[i - 1] + (1 if is_qr[i - 1] else 0)

    # For a window [s, s+M-1], we need all to be QR
    # That means sum of is_qr[s..s+M-1] == M
    # Using prefix: prefix[s+M] - prefix[s] == M
    
    for s in range(1, P - M + 1):
        if time.time() > deadline:
            break
        # Check if s, s+1, ..., s+M-1 are all quadratic residues
        # s ranges from 1 to P-M, so s+M-1 ranges from M to P-1, all in [1, P-1]
        window_sum = prefix[s + M] - prefix[s]
        if window_sum == M:
            roots = [qr_root[s + i] for i in range(M)]
            return {'start': s, 'roots': roots}

    # Fallback (should not happen for P=1009, M=5)
    return {'start': 1, 'roots': [1, 1, 1, 1, 1]}
