def construct():
        n = 256
        # Start with the "safe" upper set
        S = set(range(171, n + 1))
        
        # Greedy add smaller numbers
        for x in range(170, 0, -1):
            valid = True
            # Check if x divides any sum of two elements in S
            # Since x < all elements in S currently, x is the 'a' in the triple
            # We need to check pairs (b, c) in S
            lst = list(S)
            m = len(lst)
            for i in range(m):
                b = lst[i]
                for j in range(i, m):
                    c = lst[j]
                    if (b + c) % x == 0:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                S.add(x)
                
        # Local search to improve
        # Try adding any number
        improved = True
        while improved:
            improved = False
            for x in range(1, n + 1):
                if x in S:
                    continue
                # Check validity
                valid = True
                lst = list(S)
                m = len(lst)
                # x can be a, b, or c
                # We need to check all triples in S U {x}
                # This is O(N^3) per check, which is slow for N=256 (16M ops)
                # But we can optimize.
                # Actually, checking validity of S U {x} only requires checking triples involving x.
                
                # 1. x is a: x < b <= c in S
                for b in S:
                    if b <= x: continue
                    for c in S:
                        if c < b: continue
                        if (b + c) % x == 0:
                            valid = False
                            break
                    if not valid: break
                
                if valid:
                    # 2. x is b: a < x <= c in S
                    for a in S:
                        if a >= x: continue
                        for c in S:
                            if c < x: continue
                            # a < x <= c
                            if (x + c) % a == 0:
                                valid = False
                                break
                        if not valid: break
                
                if valid:
                    # 3. x is c: a < b < x in S
                    for a in S:
                        if a >= x: continue
                        for b in S:
                            if b <= a or b >= x: continue
                            # a < b < x
                            if (b + x) % a == 0:
                                valid = False
                                break
                        if not valid: break
                
                if valid:
                    S.add(x)
                    improved = True
                    
        return list(S)
