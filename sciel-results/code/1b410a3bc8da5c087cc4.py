def construct():
    p = 1009
    # Use lists for O(1) access and lower overhead than dict/set
    is_qr = [False] * p
    root_of = [0] * p
    
    # Precompute quadratic residues and their roots
    for x in range(1, p):
        r = (x * x) % p
        is_qr[r] = True
        root_of[r] = x
        
    # Find the minimal start s
    # s ranges from 1 to p-5 (1004)
    for s in range(1, p - 4):
        if is_qr[s] and is_qr[s+1] and is_qr[s+2] and is_qr[s+3] and is_qr[s+4]:
            res_roots = [root_of[s + i] for i in range(5)]
            return {'start': s, 'roots': res_roots}
            
    raise RuntimeError("No run of 5 consecutive quadratic residues found")
