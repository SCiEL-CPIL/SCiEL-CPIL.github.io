    for tup2 in itertools.product([-1, 0, 1], repeat=15):
        n2 = 0
        for i, c in zip(range(15), tup2):
            if c != 0:
                n2 += c * w2[i]
                
        A = -n2 * D1
        q = A // D2
        
        # Check around q and q+1
        idx1 = bisect.bisect_left(N1_vals, q)
        idx2 = bisect.bisect_left(N1_vals, q + 1)
        
        candidates = set()
        for idx in [idx1, idx2]:
            for delta in range(-5, 6):
                ci = idx + delta
                if 0 <= ci < len(N1_vals):
                    candidates.add(ci)
                    
        for ci in candidates:
            n1 = N1_vals[ci]
            err = abs(n1 * D2 + n2 * D1)
            if err == 0:
                continue
            
            if best_err is None or err < best_err:
                best_err = err
                best_code1 = sums1[ci][1]
                best_code2 = encode(tup2)
