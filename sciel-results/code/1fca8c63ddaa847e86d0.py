def construct():
    n = 128
    S = set()
    
    # Iterate from largest to smallest
    for x in range(n, 0, -1):
        # Check if adding x violates the condition
        # Violation: exists distinct a,b,c in S_new such that a+b, a+c, b+c in S_new
        # Since we add x, x must be one of a,b,c or one of the sums.
        # As established, x cannot be a sum of two existing elements in S 
        # because existing elements are > x, so their sum > 2x > x.
        # Thus x must be one of a, b, or c.
        # WLOG x = a.
        # We need b, c in S (so b, c > x) such that:
        # x + b in S
        # x + c in S
        # b + c in S
        
        valid = True
        # To optimize, we can pre-filter b
        # b must be in S
        # x+b must be in S
        # So b must be in S and (x+b) in S
        candidates_b = []
        for b in S:
            if x + b in S:
                candidates_b.append(b)
                
        for b in candidates_b:
            # Now check for c
            # c must be in S, c != b
            # x+c in S
            # b+c in S
            for c in S:
                if c == b:
                    continue
                if x + c in S and b + c in S:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            S.add(x)
            
    return list(S)
