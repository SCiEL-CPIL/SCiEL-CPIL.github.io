def construct():
    import itertools
    
    def is_valid(A, B):
        A_set = set(A)
        B_set = set(B)
        S_set = A_set | B_set
        
        # Check triples
        # 1. 3 in A
        # 2. 2 in A, 1 in B
        # 3. 1 in A, 2 in B (impossible as b+c > 128)
        # 4. 3 in B (impossible as b+c > 128)
        
        # Optimization: 
        # If a,b in A, c in B:
        # Need a+b in S, a+c in S, b+c in S.
        # Since a,b in A (<=64), a+b <= 128.
        # Since a,b >= 1, a+b >= 2.
        # a+c >= 66, b+c >= 66.
        
        # Precompute A pairs sums
        A_list = list(A)
        B_list = list(B)
        
        # Check 3 in A
        # If A is sum-free, this is empty. We assume A is sum-free in our candidates.
        
        # Check 2 in A, 1 in B
        # For each c in B:
        #   For each pair a,b in A:
        #     if a+b in S and a+c in S and b+c in S: return False
        
        # To speed up, for each c in B:
        #   Let valid_a_for_c = {a in A | a+c in S}
        #   We need a,b in valid_a_for_c such that a+b in S.
        
        for c in B_list:
            # Find a in A such that a+c in S
            # a+c is in [66, 192]. S is [1, 128].
            # So a+c must be in B (since > 64).
            # So a+c in B.
            valid_a = []
            for a in A_list:
                s = a + c
                if s in B_set:
                    valid_a.append(a)
            
            # Now check if any pair a,b in valid_a has a+b in S
            # a+b is in A or B.
            # If A is sum-free, a+b not in A. So a+b must be in B.
            for i in range(len(valid_a)):
                a = valid_a[i]
                for j in range(i+1, len(valid_a)):
                    b = valid_a[j]
                    s_ab = a + b
                    if s_ab in S_set:
                        return False
        return True

    def greedy_B(A):
        A_set = set(A)
        S_A = A_set
        B = []
        B_set = set()
        
        # Try to add elements from 128 down to 65
        for c in range(128, 64, -1):
            # Tentatively add c
            B_set.add(c)
            
            # Check validity
            # Only need to check conflicts involving c
            # Conflicts: a,b in A, c' in B such that sums in S.
            # If c is the new element, it can be:
            # 1. The 'c' in (a,b,c)
            # 2. One of the sums a+b, a+c, b+c
            
            # Case 1: c is the element c in (a,b,c)
            # We need a,b in A such that a+b in S, a+c in S, b+c in S.
            # a+c in S => a+c in B (since >64).
            # b+c in S => b+c in B.
            # a+b in S => a+b in A or B.
            
            # Case 2: c is a sum.
            # Subcase 2a: c = a+b. Then a,b in A.
            # We need a+c in S and b+c in S.
            # a+c = a+a+b. b+c = b+a+b.
            # These must be in S.
            # If a+c in S and b+c in S, and c=a+b in S, then violation.
            
            # Subcase 2b: c = a+c'. Then c' in B, a in A.
            # We need a+b in S and b+c' in S.
            # Here the triple is (a, b, c'). Sums: a+b, a+c'=c, b+c'.
            # So we need a+b in S and b+c' in S.
            
            # This is getting complicated to check incrementally.
            # Given small size, just run full is_valid on A, B_set.
            
            if is_valid(A, B_set):
                B.append(c)
            else:
                B_set.remove(c)
                
        return B

    best_size = 0
    best_set = []
    
    # Candidate A sets
    # 1. Odds in [1, 64]
    A1 = list(range(1, 65, 2))
    # 2. Upper half [33, 64]
    A2 = list(range(33, 65))
    # 3. Current A
    A3 = list(range(1, 34, 2))
    # 4. Random sum-free? No, stick to structured.
    
    candidates = [A1, A2, A3]
    
    for A in candidates:
        B = greedy_B(A)
        S = A + B
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
            
    return best_set
