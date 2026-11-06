import time

_P = 1009
_M = 5

def construct():
    # 1. Identify all non-zero quadratic residues
    # Use a set for O(1) lookup, or a boolean array for O(1) access in the scan.
    # Boolean array is faster for the subsequent scan.
    is_res = [False] * _P
    root_map = {}
    
    for x in range(1, _P):
        r = (x * x) % _P
        if r != 0:
            is_res[r] = True
            # Store one root for each residue for certification
            if r not in root_map:
                root_map[r] = x

    # 2. Scan for the smallest s such that s, s+1, ..., s+4 are all residues
    # s must be in [1, P-M] to avoid wraparound and zero.
    # P-M = 1009 - 5 = 1004.
    
    min_s = -1
    for s in range(1, _P - _M + 1):
        # Check if s, s+1, s+2, s+3, s+4 are all True in is_res
        if is_res[s] and is_res[s+1] and is_res[s+2] and is_res[s+3] and is_res[s+4]:
            min_s = s
            break

    if min_s == -1:
        # Fallback: should not happen for p=1009
        # If it does, we can't satisfy the constraint. 
        # The problem implies a solution exists.
        raise RuntimeError("No run of 5 consecutive non-zero quadratic residues found.")

    # 3. Construct the result
    roots = []
    for i in range(_M):
        val = min_s + i
        roots.append(root_map[val])
        
    # 4. Verify
    for i, r in enumerate(roots):
        if (r * r) % _P != min_s + i:
            raise ValueError("Verification failed")
            
    return {'start': min_s, 'roots': roots}
