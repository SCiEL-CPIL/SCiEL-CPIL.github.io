def construct():
    p = 1009
    run_len = 5
    
    # Compute QR set
    qr_set = set()
    for x in range(1, p):
        qr_set.add((x * x) % p)
        
    # Find minimal s
    for s in range(1, p - run_len + 1):
        # Check if s, s+1, ..., s+4 are in qr_set
        # Note: s+i < p is guaranteed by range limit
        if all((s + i) in qr_set for i in range(run_len)):
            # Find roots
            roots = []
            for i in range(run_len):
                val = s + i
                for r in range(1, p):
                    if (r * r) % p == val:
                        roots.append(r)
                        break
            return {'start': s, 'roots': roots}
            
    raise RuntimeError("No run found")
