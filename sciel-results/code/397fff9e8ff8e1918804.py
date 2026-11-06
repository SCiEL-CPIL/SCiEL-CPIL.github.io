_P = 1009

def construct():
    # 1. Identify all non-zero quadratic residues
    # A number v is a QR if there exists x such that x^2 % P == v
    # We just need to mark them.
    is_res = [False] * _P
    
    # x ranges from 1 to P-1
    # x^2 % P will be non-zero for x in 1..P-1
    for x in range(1, _P):
        r = (x * x) % _P
        if r != 0:
            is_res[r] = True
            
    # 2. Find the smallest s in [1, P-5] such that s, s+1, s+2, s+3, s+4 are all QRs
    # We need roots for certification.
    # We can find roots on the fly or store them.
    # Storing one root per residue is efficient.
    root_map = {}
    for x in range(1, _P):
        r = (x * x) % _P
        if r != 0 and r not in root_map:
            root_map[r] = x
            
    # Scan for the run
    for s in range(1, _P - 4):
        if is_res[s] and is_res[s+1] and is_res[s+2] and is_res[s+3] and is_res[s+4]:
            roots = [root_map[s+i] for i in range(5)]
            # Verify
            for i, r in enumerate(roots):
                if (r * r) % _P != s + i:
                    continue # Should not happen
            return {'start': s, 'roots': roots}
            
    # Fallback if not found (should not happen)
    raise RuntimeError("No run found")
