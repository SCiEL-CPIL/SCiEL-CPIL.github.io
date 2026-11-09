def construct():
    import time
    start_time = time.time()
    deadline = start_time + 50
    
    N = 256
    
    # Precompute gcd conflicts for local search
    # conflict[i][j] = True if gcd(i, j) > 1
    # Using 1-based indexing for easier logic, 0 is dummy
    # Actually, let's just use a function for local search to save memory
    from math import gcd

    def is_valid_set(S):
        # Check if all pairs in S are coprime
        for i in range(len(S)):
            for j in range(i+1, len(S)):
                if gcd(S[i], S[j]) > 1:
                    return False
        return True

    def local_search(S):
        # S is a list of ints
        # Returns improved list
        improved = True
        S_set = set(S)
        current_sum = sum(S)
        
        while improved and time.time() < deadline - 2:
            improved = False
            
            # 1. Try to add elements
            # Sort candidates descending
            candidates = [x for x in range(N, 0, -1) if x not in S_set]
            for v in candidates:
                if time.time() > deadline - 2:
                    break
                # Check if v is coprime to all in S
                ok = True
                for u in S:
                    if gcd(v, u) > 1:
                        ok = False
                        break
                if ok:
                    S.append(v)
                    S_set.add(v)
                    current_sum += v
                    improved = True
                    # Restart candidate loop or just continue?
                    # If we add v, it might block others, but we are adding max possible.
                    # We can break and restart to be safe, or just continue.
                    # Continuing is fine because we are iterating descending.
                    # Actually, adding v might make a previously incompatible v' compatible? No, adding constraints only removes options.
                    # So we can just continue.
            
            # 2. Try to swap elements
            # For each u in S, try to replace with v > u not in S
            # Sort S by value ascending to try replacing small ones first?
            # Or just iterate all pairs.
            S_sorted = sorted(S)
            for i, u in enumerate(S_sorted):
                if time.time() > deadline - 2:
                    break
                # Try to find v > u not in S that is coprime to S \ {u}
                # Iterate v descending from N to u+1
                for v in range(N, u, -1):
                    if v in S_set:
                        continue
                    # Check if v is coprime to all w in S except u
                    ok = True
                    for j, w in enumerate(S):
                        if i == j: # This logic is tricky with indices if S changes
                            continue
                        # Actually, S is a list. Let's use set logic for check
                        pass
                    
                    # Better check:
                    ok = True
                    for w in S:
                        if w == u:
                            continue
                        if gcd(v, w) > 1:
                            ok = False
                            break
                    
                    if ok:
                        # Swap u with v
                        S.remove(u)
                        S_set.remove(u)
                        S.append(v)
                        S_set.add(v)
                        current_sum += (v - u)
                        improved = True
                        break # Break inner v loop, restart outer or continue?
                        # Restarting is safer to catch new opportunities
                        # But breaking inner and continuing outer might miss things if S changed significantly.
                        # Let's break out of both and restart the while loop
                        break
                
                if improved:
                    break
        
        return S

    # 1. Try ILP
    result = None
    try:
        import scipy.optimize as opt
        import numpy as np
        from scipy.sparse import csc_matrix
        
        A_rows = []
        A_cols = []
        A_vals = []
        
        for i in range(1, N+1):
            for j in range(i+1, N+1):
                if gcd(i, j) > 1:
                    idx = len(A_rows) // 2
                    A_rows.append(idx)
                    A_cols.append(i-1)
                    A_vals.append(1)
                    
                    A_rows.append(idx)
                    A_cols.append(j-1)
                    A_vals.append(1)
        
        A = csc_matrix((A_vals, (A_rows, A_cols)), shape=(len(A_rows)//2, N))
        b = np.ones(len(A_rows)//2)
        
        c = -np.arange(1, N+1, dtype=float)
        bounds = [(0, 1)] * N
        integrality = np.ones(N)
        
        # Use remaining time for ILP
        remaining_time = max(1, int(deadline - time.time()) - 5)
        res = opt.milp(c, constraints=(A, b), bounds=bounds, integrality=integrality, options={'time_limit': remaining_time})
        
        # Accept if optimal (0) or feasible solution found (1, 2, etc. if x is not None)
        if res.x is not None:
            x = res.x
            # Round to nearest integer, check validity
            x_int = [1 if val > 0.5 else 0 for val in x]
            result = [int(i+1) for i, val in enumerate(x_int) if val == 1]
            
            # Verify validity
            if is_valid_set(result):
                # Ensure 1 is included if possible? 1 is coprime to all.
                # If 1 is not in result, add it?
                # 1 adds 1 to sum and conflicts with nothing.
                if 1 not in result:
                    result.append(1)
            else:
                result = None
    except Exception:
        result = None
    
    # 2. Fallback: Prime Powers
    if result is None:
        def is_prime(n):
            if n < 2: return False
            for i in range(2, int(n**0.5)+1):
                if n % i == 0:
                    return False
            return True
            
        primes = [a for a in range(2, N+1) if is_prime(a)]
        result = [1]
        for p in primes:
            max_p = p
            while max_p * p <= N:
                max_p *= p
            result.append(max_p)
            
    # 3. Local Search Refinement
    if time.time() < deadline - 2:
        result = local_search(result)
        
    return result
