def construct():
    P = 1009
    M = 5
    
    # Precompute quadratic residues and a corresponding root for each
    is_qr = [False] * P
    root_map = [0] * P
    
    for x in range(1, P):
        y = (x * x) % P
        if not is_qr[y]:
            is_qr[y] = True
            root_map[y] = x

    # Search for the minimum s
    for s in range(1, P - M + 1):
        if is_qr[s] and is_qr[s+1] and is_qr[s+2] and is_qr[s+3] and is_qr[s+4]:
            return {
                'start': s,
                'roots': [root_map[s], root_map[s+1], root_map[s+2], root_map[s+3], root_map[s+4]]
            }
            
    # Fallback (should not be reached for P=1009)
    return {'start': 1, 'roots': [1, 1, 1, 1, 1]}
