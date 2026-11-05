import time

P = 1009
M = 5
DEADLINE = 85.0

def construct():
    t0 = time.time()
    
    # Determine which values are non-zero quadratic residues
    is_qr = [False] * P
    roots = [0] * P
    
    for x in range(1, P):
        r = (x * x) % P
        if r != 0:
            is_qr[r] = True
            if roots[r] == 0:
                roots[r] = x
            
    # Build bitmask
    mask = 0
    for v in range(1, P):
        if is_qr[v]:
            mask |= (1 << v)
            
    target_mask = (1 << M) - 1
    
    for s in range(1, P - M + 1):
        # Check if bits s, s+1, ..., s+M-1 are all set
        # Shift mask right by s, then check lowest M bits
        if (mask >> s) & target_mask == target_mask:
            result_roots = [roots[s + i] for i in range(M)]
            # Validate
            for i in range(M):
                if (result_roots[i] * result_roots[i]) % P != s + i:
                    # Fallback search if root was not recorded (should not happen)
                    for xv in range(1, P):
                        if (xv * xv) % P == s + i:
                            result_roots[i] = xv
                            break
            return {'start': s, 'roots': result_roots}
            
    raise RuntimeError("No run found")
