import time

def construct():
    N = 128
    # Start with the safe upper half
    S = set(range(N//2 + 1, N + 1))
    
    # Candidates to add: 1 to N//2
    candidates = list(range(1, N//2 + 1))
    
    # We will iterate through candidates and add them if valid.
    # Order might matter. Smaller numbers might be "safer" or "riskier"?
    # Let's try adding in increasing order.
    
    for x in candidates:
        if x in S:
            continue
            
        # Check if S U {x} is valid
        valid = True
        
        # 1. Check if x completes a sum for a triple in S
        # We need a,b,c in S such that one sum is x and others are in S.
        # a+b=x, a+c in S, b+c in S
        # Since a,b,c in S, and x not in S, x is the new element.
        # We can iterate pairs (a,b) in S.
        # If a+b == x, then we need to find c in S, c!=a,b such that a+c in S and b+c in S.
        # To make this fast, we can precompute for current S?
        # Or just check.
        # Since we are adding one by one, S is fixed during check.
        
        # Optimization:
        # Precompute sums_in_S: dict from sum -> list of pairs (a,b) such that a+b=sum and a,b in S.
        # This might be heavy to update.
        
        # Let's just do the O(N^2) check for sums if N is small enough.
        # N=128. Pairs ~ 8000.
        # For each pair (a,b) with a+b=x:
        #   Check if there exists c in S, c!=a,b, with a+c in S and b+c in S.
        
        # Actually, let's combine with the "x as member" check.
        
        # Check 1: x is a member.
        # Iterate b, c in S.
        # If b+c in S or b+c == x:
        #   Check if x+b in S or x+b == x (impossible) -> x+b in S
        #   Check if x+c in S
        #   Also need a,b,c distinct. Here the triple is (x, b, c).
        #   So b != c, b != x, c != x. (b,c in S, x not in S, so distinct ok).
        
        # Check 2: x is a sum.
        # Iterate a, b in S.
        # If a+b == x:
        #   Need c in S, c!=a,b such that a+c in S and b+c in S.
        
        # Let's implement these checks.
        
        S_list = list(S)
        n = len(S_list)
        
        # Check 1: x in triple
        for i in range(n):
            b = S_list[i]
            for j in range(i+1, n):
                c = S_list[j]
                # Triple is (x, b, c)
                # Sums: x+b, x+c, b+c
                # Need all in S U {x}
                
                # b+c
                if b + c > N:
                    continue # b+c not in S U {x}
                if b + c not in S and b + c != x:
                    continue
                    
                # x+b
                if x + b > N:
                    continue
                if x + b not in S and x + b != x: # x+b != x always
                    continue
                # Note: x+b could be x? No. x+b > x.
                # So x+b must be in S.
                if x + b not in S:
                    continue
                
                # x+c
                if x + c > N:
                    continue
                if x + c not in S:
                    continue
                
                # If we reach here, (x,b,c) forms a violation.
                valid = False
                break
            if not valid:
                break
        
        if not valid:
            continue
            
        # Check 2: x is a sum
        # a+b = x
        for i in range(n):
            a = S_list[i]
            # b = x - a
            b = x - a
            if b <= a: # ensure a < b to avoid duplicates and a!=b
                continue
            if b not in S:
                continue
                
            # We have a, b in S such that a+b=x.
            # Need c in S, c != a, b such that a+c in S and b+c in S.
            found_c = False
            for c in S_list:
                if c == a or c == b:
                    continue
                if a + c in S and b + c in S:
                    found_c = True
                    break
            if found_c:
                valid = False
                break
                
        if valid:
            S.add(x)
            
    return sorted(S)
