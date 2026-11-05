import time

P = 1009
M = 5
DEADLINE = 60.0

def construct():
    start_time = time.time()
    
    # Precompute quadratic residues and their roots
    # is_qr[v] = True if v is a non-zero QR mod P
    # root[v] = x such that x^2 % P == v
    is_qr = [False] * P
    root = [0] * P
    
    for x in range(1, P):
        r = (x * x) % P
        if r != 0:
            is_qr[r] = True
            if root[r] == 0:
                root[r] = x
        
        if time.time() - start_time > DEADLINE - 1.0:
            break
    
    # Build a bitmask where bit v is set if v is a non-zero QR
    mask = 0
    for v in range(1, P):
        if is_qr[v]:
            mask |= (1 << v)
    
    # Search for the smallest s
    # s must be in [1, P - M]
    # s, s+1, ..., s+4 must be QRs
    # Check via bitmask: (mask >> s) & 0x1F == 0x1F
    for s in range(1, P - M + 1):
        if time.time() - start_time > DEADLINE - 1.0:
            break
            
        if ((mask >> s) & 0x1F) == 0x1F:
            # Found the minimal s
            res_roots = [root[s], root[s+1], root[s+2], root[s+3], root[s+4]]
            
            # Validate roots
            valid = True
            for i in range(M):
                r = res_roots[i]
                if (r * r) % P != s + i:
                    # Fallback: find root manually
                    found = False
                    for x in range(1, P):
                        if (x * x) % P == s + i:
                            res_roots[i] = x
                            found = True
                            break
                    if not found:
                        valid = False
                        break
            
            if valid:
                return {'start': s, 'roots': res_roots}
    
    # If no run found (should not happen for P=1009, M=5)
    raise RuntimeError("No run found")
