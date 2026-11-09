def construct():
    import time
    start_time = time.time()
    deadline = start_time + 50
    
    # 1. Try ILP
    try:
        import scipy.optimize as opt
        import numpy as np
        from math import gcd
        
        N = 256
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
        
        from scipy.sparse import csc_matrix
        A = csc_matrix((A_vals, (A_rows, A_cols)), shape=(len(A_rows)//2, N))
        b = np.ones(len(A_rows)//2)
        
        c = -np.arange(1, N+1, dtype=float)
        bounds = [(0, 1)] * N
        integrality = np.ones(N)
        
        res = opt.milp(c, constraints=(A, b), bounds=bounds, integrality=integrality, options={'time_limit': 40})
        
        if res.status == 0:
            x = res.x
            result = [int(i+1) for i, val in enumerate(x) if val > 0.5]
            if 1 not in result:
                result.append(1)
            return result
            
    except Exception:
        pass
    
    # 2. Fallback: Prime Powers
    def is_prime(n):
        if n < 2: return False
        for i in range(2, int(n**0.5)+1):
            if n % i == 0:
                return False
        return True
        
    primes = [a for a in range(2, 257) if is_prime(a)]
    result = [1]
    for p in primes:
        max_p = p
        while max_p * p <= 256:
            max_p *= p
        result.append(max_p)
    return result
