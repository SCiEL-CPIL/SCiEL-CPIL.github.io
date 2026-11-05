def construct():
    import time
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s kill
    
    P = 1009
    M = 5
    
    # Precompute quadratic residues modulo P
    # qr_root[x] = r such that r^2 % P == x, for x in 1..P-1
    # We want the smallest positive root for each residue
    qr_root = [0] * P
    for r in range(1, P):
        x = (r * r) % P
        if x != 0 and qr_root[x] == 0:
            qr_root[x] = r
    
    # Create a bitmask where bit i is set if i is a quadratic residue
    # We need to find s such that s, s+1, s+2, s+3, s+4 are all quadratic residues
    # and none of them are 0 (mod P), and roots exist in [1, P-1]
    
    # Build a boolean array for which values are valid quadratic residues with known roots
    is_qr = [False] * P
    for x in range(1, P):
        if qr_root[x] != 0:
            is_qr[x] = True
    
    # Search for the smallest s in [1, P-M] such that s+i is a QR for i=0..M-1
    for s in range(1, P - M + 1):
        if time.time() > deadline:
            break
        valid = True
        for i in range(M):
            val = s + i
            if val >= P:  # wraparound excluded, but s <= P-M ensures val <= P-1
                valid = False
                break
            if not is_qr[val]:
                valid = False
                break
        if valid:
            roots = [qr_root[s + i] for i in range(M)]
            return {'start': s, 'roots': roots}
    
    # If no solution found (should not happen for P=1009, M=5), return best effort
    # This is a fallback; in practice, a solution should be found quickly
    return {'start': 1, 'roots': [1, 1, 1, 1, 1]}
