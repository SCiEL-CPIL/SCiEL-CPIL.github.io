def construct():
    n = 256
    S = set()
    # Try to add numbers from n down to 1
    for a in range(n, 0, -1):
        # Check if adding a to S is valid
        # Condition: for all b, c in S with a < b <= c, a does not divide b+c
        # Note: if a is added, it becomes the smallest 'a' in future checks?
        # No, the condition is for ALL triples in the final set.
        # If we build S incrementally, we must ensure:
        # 1. New a with existing b,c: a < b <= c => a not divide b+c
        # 2. Existing b,c with new a as b or c?
        #    The condition is a < b <= c.
        #    If we add a, it can be the 'a' in a<b<=c.
        #    It can also be 'b' or 'c' in x < a <= c or x < b <= a.
        #    Since we are adding in decreasing order, existing elements are larger.
        #    So if we add a, existing elements are > a.
        #    So a can only be the 'a' in the triple a < b <= c.
        #    It cannot be b or c because b,c must be > a, and existing are > a, but new a is smallest.
        #    Wait, if we add 10, and S has 20, 30.
        #    Triple 10 < 20 <= 30. Check 10 | 50.
        #    Triple 10 < 30 <= 30? No, b<=c.
        #    Are there triples where a is not the new element?
        #    If we add a, and S has b,c > a.
        #    The only new triples involving a are those where a is the smallest.
        #    Because any triple with b or c being a would require an element < a, which is not in S yet.
        #    So we only need to check:
        #    For all b, c in S with b <= c:
        #       a < b and a < c (implied by b>a)
        #       Check a | (b+c)
        
        valid = True
        # Optimization: only check if a has multiples in sum range
        # But S is sparse?
        # S size grows.
        for b in S:
            if b <= a: continue # b must be > a
            for c in S:
                if c < b: continue # c >= b
                if c <= a: continue # c > a
                if (b + c) % a == 0:
                    valid = False
                    break
            if not valid: break
        
        if valid:
            S.add(a)
            
    return list(S)
