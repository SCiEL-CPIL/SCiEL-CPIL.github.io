def construct():
    n = 256
    
    def is_valid(S):
        """Check if set S satisfies the constraint."""
        S_list = sorted(S)
        for i, a in enumerate(S_list):
            for j in range(i + 1, len(S_list)):
                b = S_list[j]
                for k in range(j, len(S_list)):
                    c = S_list[k]
                    if (b + c) % a == 0:
                        return False
        return True
    
    def can_add(S, x):
        """Check if x can be added to set S without violating constraints."""
        if x in S:
            return True
        S_list = sorted(S)
        
        # Check if x as 'a' causes issues: for all b,c in S with b,c > x, (b+c) % x != 0
        for i, b in enumerate(S_list):
            if b <= x:
                continue
            for j in range(i, len(S_list)):
                c = S_list[j]
                if c < b:
                    continue
                if (b + c) % x == 0:
                    return False
        
        # Check if x as 'b' or 'c' causes issues
        for a in S_list:
            if a >= x:
                continue
            # x as b: need c >= x in S, check (x + c) % a != 0
            for c in S_list:
                if c < x:
                    continue
                if (x + c) % a == 0:
                    return False
            # x as c: need b with a < b <= x, check (b + x) % a != 0
            # This is covered by the above if we consider b as any element with a < b <= x
            # Actually, when x is c, b can be any element with a < b <= x
            # The check above only checks x as b (with c >= x)
            # We need to also check x as c (with b <= x)
            for b in S_list:
                if b <= a or b > x:
                    continue
                if (b + x) % a == 0:
                    return False
        
        return True
    
    # Start with the interval [171, 256]
    S = set(range(171, 257))
    
    # Try to add smaller elements
    for x in range(170, 0, -1):
        if can_add(S, x):
            S.add(x)
    
    # Local search: try swapping elements
    improved = True
    while improved:
        improved = False
        S_list = sorted(S)
        for i in range(len(S_list)):
            for j in range(i + 1, len(S_list)):
                a = S_list[i]
                b = S_list[j]
                # Try removing b and adding a smaller element
                for x in range(1, a):
                    if x in S:
                        continue
                    S.remove(b)
                    if can_add(S, x):
                        S.add(x)
                        if len(S) > len(S_list) - 1:
                            improved = True
                        else:
                            S.remove(x)
                            S.add(b)
                        break
                    else:
                        S.add(b)
    
    return list(S)
