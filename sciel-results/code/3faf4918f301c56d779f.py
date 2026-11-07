def construct():
    S = set()
    S_list = []
    
    for x in range(1, 129):
        # Check if x can be added
        can_add = True
        
        # We need to check if there exist distinct a, b, c in S
        # such that a + b = x, a + c in S, b + c in S
        # Note: a, b, c must be distinct.
        
        # Iterate over pairs (a, b) in S such that a + b = x
        # Since a, b < x, we can iterate a from 1 to x//2
        for a in range(1, x // 2 + 1):
            if a not in S:
                continue
            b = x - a
            if b not in S:
                continue
            if a == b:
                continue # a, b must be distinct for the triple a,b,c to have distinct a,b? 
                         # The problem says "THREE DISTINCT members a,b,c". 
                         # So a, b, c are distinct. 
                         # If a==b, they are not distinct members. 
                         # So we skip a==b.
            
            # Now we have distinct a, b in S with a+b=x.
            # We need to check if there exists c in S, c != a, c != b
            # such that (a+c) in S and (b+c) in S.
            
            # Optimization: 
            # If we find such a c, then adding x creates a bad triple (a,b,c).
            # So we set can_add = False and break.
            
            # Iterate c in S
            for c in S_list:
                if c == a or c == b:
                    continue
                if (a + c) in S and (b + c) in S:
                    can_add = False
                    break
            if not can_add:
                break
        
        if can_add:
            S.add(x)
            S_list.append(x)
            
    return [int(x) for x in S_list]
