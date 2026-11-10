def construct():
    import time
    start_time = time.time()
    deadline = 60.0 # Safety margin
    
    S = set()
    # Iterate from largest to smallest
    for x in range(180, 0, -1):
        if time.time() - start_time > deadline:
            break
            
        can_add = True
        # Check for conflicts with existing elements in S
        # We need b, c in S such that 1/x = 1/b + 1/c
        # bc = x(b+c) => (b-x)(c-x) = x^2
        # Let u = b-x. Then c-x = x^2/u => c = x + x^2/u
        # b = x + u
        # We need u to be a divisor of x^2.
        # Also b, c must be in S.
        # Since we are adding x, S contains only numbers > x.
        # So b > x and c > x are guaranteed if u > 0 and v > 0.
        
        # Iterate through possible divisors u of x^2
        # Or simply iterate through b in S
        # Iterating through b in S is O(|S|)
        for b in S:
            if b <= x:
                continue
            u = b - x
            if x * x % u == 0:
                v = (x * x) // u
                c = x + v
                if c in S and c != b:
                    can_add = False
                    break
        
        if can_add:
            S.add(x)
            
    return list(S)
