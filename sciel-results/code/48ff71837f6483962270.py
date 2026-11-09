def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    N = 180
    elements = list(range(1, N+1))
    
    # Precompute all conflict triples: (a, b, c) where 1/a = 1/b + 1/c
    # a = k*m*n, b = k*m*(m+n), c = k*n*(m+n), gcd(m,n)=1
    # Build conflict graph: edge between x,y if they appear together in some triple
    # Actually, we need hyperedges of size 3. For max independent set, 
    # we need: no three elements form a valid triple.
    
    # Let's build the set of all valid triples
    triples = set()
    for k in range(1, 181):
        for m in range(1, 181):
            for n in range(1, 181):
                if m >= n:  # avoid duplicates, assume m < n
                    continue
                # Check gcd(m,n) = 1
                if _gcd(m, n) != 1:
                    continue
                a = k * m * n
                b = k * m * (m + n)
                c = k * n * (m + n)
                if a <= N and b <= N and c <= N:
                    # Check all distinct
                    if len({a, b, c}) == 3:
                        triples.add((min(a,b,c), min(max(a,b), max(b,c)), max(a,b,c)))
                        
    # --- NEW: Try ILP Solver ---
    try:
        import numpy as np
        from scipy.optimize import milp, LinearConstraint, Bounds
        
        triples_list = list(triples)
        num_vars = N
        num_constraints = len(triples_list)
        
        # Objective: maximize sum(x) -> minimize -sum(x)
        c_obj = np.full(num_vars, -1.0)
        
        # Constraints: for each triple (a,b,c), x_a + x_b + x_c <= 2
        A = np.zeros((num_constraints, num_vars))
        for i, (a, b, c_val) in enumerate(triples_list):
            A[i, a-1] = 1
            A[i, b-1] = 1
            A[i, c_val-1] = 1
            
        bounds = Bounds(lb=0, ub=1)
        integrality = np.ones(num_vars)
        constraint = LinearConstraint(A, lb=-np.inf, ub=2)
        
        res = milp(c=c_obj, constraints=constraint, integrality=integrality, bounds=bounds)
        if res.status == 0:
            x = np.round(res.x).astype(int)
            result_set = [i+1 for i in range(num_vars) if x[i] == 1]
            if len(result_set) > 0:
                # Verify
                res_set = set(result_set)
                valid = True
                for (a, b, c_val) in triples_list:
                    if a in res_set and b in res_set and c_val in res_set:
                        valid = False
                        break
                if valid:
                    return result_set
    except Exception:
        pass
    # --- END NEW ---

    # Build conflict adjacency... (rest of original code)
