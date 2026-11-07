def construct():
    import time
    import numpy as np
    from scipy.optimize import milp, LinearConstraint, Bounds
    from scipy.sparse import csr_matrix

    start_time = time.time()
    deadline = start_time + 45.0

    # 1. Greedy Fallback
    candidates = list(range(256, 1, -1))
    S_greedy = []
    for x in candidates:
        can_add = True
        # Check 1: x does not divide b*c for any distinct b,c in S
        for i in range(len(S_greedy)):
            for j in range(i+1, len(S_greedy)):
                b = S_greedy[i]
                c = S_greedy[j]
                if (b * c) % x == 0:
                    can_add = False
                    break
            if not can_add:
                break
        
        if can_add:
            # Check 2: No existing b divides x*c for any c in S \ {b}
            # This check is redundant if Check 1 is symmetric?
            # The condition is: for ALL distinct a,b,c in S, a does not divide b*c.
            # When adding x, we must check:
            # 1. x does not divide b*c for any b,c in S (distinct).
            # 2. b does not divide x*c for any b,c in S (distinct, b!=c, b!=x, c!=x).
            # 3. c does not divide x*b ... same as 2.
            # So yes, both checks are needed.
            
            for b in S_greedy:
                for c in S_greedy:
                    if b == c:
                        continue
                    # b and c are distinct members of S.
                    # We add x. Triple is {b, c, x}.
                    # Condition: b | c*x ?
                    if (x * c) % b == 0:
                        can_add = False
                        break
                if not can_add:
                    break
            
        if can_add:
            S_greedy.append(x)
            
    greedy_size = len(S_greedy)
    
    # 2. ILP Attempt
    try:
        n_vars = 255
        # Map val -> idx: idx = val - 2
        
        # Generate unique forbidden triples
        # A triple (a,b,c) is forbidden if a|bc (and permutations? No, the condition is "a does not divide b*c".
        # The problem statement: "For every THREE DISTINCT members a,b,c, require that a does not divide b*c."
        # This implies the condition must hold for ALL orderings?
        # "a, b, c" are distinct members. Usually this means for the set {a,b,c}, for any element a in the set, a does not divide the product of the other two.
        # Let's re-read carefully: "For every THREE DISTINCT members a,b,c, require that a does not divide b*c."
        # If this means "for any choice of distinct a,b,c from the set, the specific a does not divide b*c",
        # then it must hold for all 6 permutations?
        # If I pick a=2, b=3, c=4. 2 | 3*4=12. Invalid.
        # If I pick a=3, b=2, c=4. 3 | 2*4=8? No.
        # If the condition is "for ALL distinct a,b,c", then it fails if ANY a divides b*c.
        # So the set {2,3,4} is invalid because 2|12.
        # So the constraint is: For any distinct a,b,c in S, it is NOT true that a|bc AND NOT true that b|ac AND NOT true that c|ab.
        # Actually, if ANY of these is true, the set is invalid.
        # So the forbidden condition for a triple {u,v,w} is:
        # (u | vw) OR (v | uw) OR (w | uv).
        # So we must exclude any triple where at least one divides the product of the other two.
        
        # My previous constraint generation only checked a|bc for a specific a.
        # I need to check if ANY of the three divides the product of the other two.
        
        forbidden_triples = set()
        
        # To speed up, iterate over all triples? 255^3/6 ~ 2.7M.
        # Check divisibility for each.
        # This is fast enough.
        
        for i in range(2, 257):
            for j in range(i+1, 257):
                for k in range(j+1, 257):
                    # Check if i | j*k
                    if (j * k) % i == 0:
                        forbidden_triples.add((i, j, k))
                    elif (i * k) % j == 0:
                        forbidden_triples.add((i, j, k))
                    elif (i * j) % k == 0:
                        forbidden_triples.add((i, j, k))
                        
        if time.time() > deadline - 15:
            return S_greedy

        m = len(forbidden_triples)
        if m == 0:
            return list(range(2, 257))
            
        row_indices = []
        col_indices = []
        data = []
        
        for idx, (a, b, c) in enumerate(forbidden_triples):
            row_indices.extend([idx, idx, idx])
            col_indices.extend([a-2, b-2, c-2])
            data.extend([1, 1, 1])
            
        A_ub = csr_matrix((data, (row_indices, col_indices)), shape=(m, n_vars))
        b_ub = np.ones(m)
        
        c_obj = np.ones(n_vars)
        bounds = Bounds(lb=0, ub=1)
        lin_cons = LinearConstraint(A_ub, ub=b_ub)
        
        # Solve
        res = milp(c=c_obj, constraints=lin_cons, integrality=np.ones(n_vars), bounds=bounds, options={"time_limit": 25})
        
        if res.status in [0, 1]:
            x_sol = res.x
            S_ilp = [i + 2 for i in range(n_vars) if x_sol[i] > 0.5]
            
            if len(S_ilp) > greedy_size:
                # Verify
                valid = True
                for a in S_ilp:
                    for b in S_ilp:
                        if a == b: continue
                        for c in S_ilp:
                            if a == c or b == c: continue
                            if (b * c) % a == 0:
                                valid = False
                                break
                        if not valid: break
                    if not valid: break
                if valid:
                    return S_ilp
                else:
                    return S_greedy
            else:
                return S_greedy
        else:
            return S_greedy
            
    except Exception:
        return S_greedy
