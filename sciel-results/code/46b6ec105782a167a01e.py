def construct():
    import numpy as np
    from scipy.sparse import csr_matrix
    from scipy.optimize import milp, LinearConstraint, Bounds
    from math import gcd
    import time

    N = 512
    start_time = time.time()
    time_limit = 55.0

    # 1. Identify conflicts
    # We need x_i + x_j <= 1 for all pairs (i, j) where lcm(i, j) > N
    # Variables are indexed 0 to N-1 representing numbers 1 to N
    
    row_idx = []
    col_idx = []
    data = []
    
    constraint_id = 0
    for i in range(1, N + 1):
        idx_i = i - 1
        for j in range(i + 1, N + 1):
            idx_j = j - 1
            g = gcd(i, j)
            if i * j > N * g:
                # Add constraint row: x_i + x_j <= 1
                row_idx.append(constraint_id)
                col_idx.append(idx_i)
                data.append(1)
                
                row_idx.append(constraint_id)
                col_idx.append(idx_j)
                data.append(1)
                
                constraint_id += 1

    # Fallback heuristic if MIP fails or is not available
    def greedy_fallback():
        S = []
        # Sort by degree of conflict (ascending) to pick "safe" nodes first
        # Calculate degrees
        deg = [0] * (N + 1)
        for i in range(1, N + 1):
            for j in range(i + 1, N + 1):
                g = gcd(i, j)
                if i * j > N * g:
                    deg[i] += 1
                    deg[j] += 1
        
        vertices = list(range(1, N + 1))
        vertices.sort(key=lambda x: deg[x])
        
        for v in vertices:
            compatible = True
            for u in S:
                g = gcd(v, u)
                if v * u > N * g:
                    compatible = False
                    break
            if compatible:
                S.append(v)
        return S if S else [1]

    if constraint_id == 0:
        return list(range(1, N + 1))

    A = csr_matrix((data, (row_idx, col_idx)), shape=(constraint_id, N))
    
    # Objective: Maximize sum(x_i) -> Minimize -sum(x_i)
    c = -np.ones(N)
    
    # Constraints: A @ x <= 1
    constraints = LinearConstraint(A, -np.inf, np.ones(constraint_id))
    
    # Bounds: 0 <= x_i <= 1
    bounds = Bounds(lb=0, ub=1)
    
    # Integrality: x_i must be integer
    integrality = np.ones(N)
    
    try:
        res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds, options={"time_limit": 45})
        
        if res.status == 0:
            sol = res.x
            selected = [i + 1 for i in range(N) if sol[i] > 0.5]
            if selected:
                return selected
        else:
            return greedy_fallback()
            
    except Exception:
        return greedy_fallback()
