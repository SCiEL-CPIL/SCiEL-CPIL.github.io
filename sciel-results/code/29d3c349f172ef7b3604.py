<<<<<<< SEARCH
    # Choose d: need girth >= g
    # LUW girth: for odd d, >= d+5; for even d, >= d+4
    # So we need: if d odd, d >= g-5; if d even, d >= g-4
    # Pick smallest valid d
    if d_override is not None:
        d = d_override
    else:
        # Find smallest d such that girth bound >= g
        d = None
        for cand in range(3, 20):
            bound = cand + 5 if cand % 2 == 1 else cand + 4
            if bound >= g:
                d = cand
                break
        if d is None:
            d = max(3, g - 5)
=======
    # Choose d: need girth >= g
    # LUW girth: for odd d, >= d+5; for even d, >= d+4
    # So we need: if d odd, d >= g-5; if d even, d >= g-4
    if d_override is not None:
        d = d_override
    else:
        # Find smallest d such that girth bound >= g
        d = None
        for cand in range(3, 20):
            bound = cand + 5 if cand % 2 == 1 else cand + 4
            if bound >= g:
                d = cand
                break
        if d is None:
            d = max(3, g - 5)
    
    # Quick Moore bound check: if the LUW graph would definitely be too large, skip
    # Moore bound for k-regular girth g: n >= 1 + k * sum_{i=0}^{floor((g-1)/2)-1} (k-1)^i
    # For even g: n >= 2 * k * sum_{i=0}^{g/2-1} (k-1)^i
    # For odd g: n >= 1 + k * sum_{i=0}^{(g-1)/2-1} (k-1)^i
    # LUW has approximately q*(d+1) vertices per side, total ~ 2*q*(d+1)
    # But the actual connected component may be much smaller.
    # We'll just build it and verify.
>>>>>>> REPLACE

