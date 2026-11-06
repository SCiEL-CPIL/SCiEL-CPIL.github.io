import time

def construct():
    p = 1009
    deadline = time.time() + 85
    
    # 1. Identify all non-zero quadratic residues
    is_qr = [False] * p
    # Precompute roots for each residue
    # root[r] will store one x such that x^2 % p == r
    roots = [0] * p
    
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            is_qr[r] = True
            # Store the first root found
            if roots[r] == 0:
                roots[r] = x
                
    # 2. Find minimum s in [1, 1004] such that s..s+4 are all QRs
    # s+4 <= 1008 => s <= 1004
    # s >= 1
    min_s = -1
    for s in range(1, 1005):
        if is_qr[s] and is_qr[s+1] and is_qr[s+2] and is_qr[s+3] and is_qr[s+4]:
            min_s = s
            break
            
    if min_s == -1:
        # Fallback, though unlikely
        raise RuntimeError("No run of 5 consecutive QRs found")
        
    # 3. Construct the result
    result_roots = []
    for i in range(5):
        r = min_s + i
        # roots[r] should be non-zero
        if roots[r] == 0:
             raise RuntimeError(f"No root found for residue {r}")
        result_roots.append(roots[r])
        
    return {'start': min_s, 'roots': result_roots}
