import math
import numpy as np

def construct():
    n = 256
    result = [1]
    
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        from scipy.sparse import coo_matrix
        
        num_vars = n - 1
        c = np.arange(2, n + 1, dtype=float)
        
        rows = []
        cols = []
        data = []
        
        for i in range(2, n + 1):
            for j in range(i + 1, n + 1):
                if math.gcd(i, j) > 1:
                    r = len(rows) // 2
                    rows.append(r)
                    cols.append(i - 2)
                    data.append(1.0)
                    
                    rows.append(r)
                    cols.append(j - 2)
                    data.append(1.0)
        
        if rows:
            A_ub = coo_matrix((data, (rows, cols)), shape=(len(rows) // 2, num_vars)).tocsr()
            b_ub = np.ones(A_ub.shape[0])
        else:
            A_ub = coo_matrix((0, num_vars)).tocsr()
            b_ub = np.empty(0)
            
        bounds = Bounds(lb=0, ub=1)
        integrality = np.ones(num_vars)
        
        res = milp(
            c=c,
            constraints=LinearConstraint(A_ub, ub=b_ub),
            integrality=integrality,
            bounds=bounds
        )
        
        if res.success:
            selected = np.round(res.x).astype(int)
            for idx, val in enumerate(selected):
                if val == 1:
                    result.append(idx + 2)
        else:
            raise RuntimeError("Solver failed")
            
    except Exception:
        # Fallback to greedy or primes if SciPy is not available or fails
        # Let's use the previous greedy logic as fallback? No, simpler: just primes.
        # Primes are always a valid set.
        is_prime = [True] * (n + 1)
        is_prime[0] = False
        is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        for i in range(2, n + 1):
            if is_prime[i]:
                result.append(i)
        
    return result
