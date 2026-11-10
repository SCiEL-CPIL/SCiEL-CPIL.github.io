import numpy as np
from scipy.optimize import linprog

def prime_number_theorem_partial_function():
    # Use a moderate K to balance performance and objective value
    # K=1000 is likely fast enough for linprog
    K = 1000
    
    # Variables: f_1, ..., f_K
    # Objective: maximize -sum(f_k * log(k) / k)
    # Equivalent to minimize sum(f_k * log(k) / k)
    c = np.zeros(K)
    for k in range(1, K+1):
        c[k-1] = np.log(k) / k
        
    # Constraints:
    # 1. M(x) <= 1.0001 for x in 1..10K
    # M(x) = sum_{k=1}^K f_k * floor(x/k)
    # A_ub matrix: rows are x, cols are k
    # A_ub[i, j] = floor((i+1) / (j+1))
    
    # This matrix is 10K x K. For K=1000, 10000x1000.
    # This might be large for linprog.
    # Let's try K=100 first to be safe, or optimize the constraint generation.
    
    # Actually, we can reduce constraints.
    # M(x) is monotonic? No.
    # But M(x) for x > K is sum of many terms.
    
    # Let's stick to K=100 for speed, as seed was K=100.
    # If K=100 works, we can try to increase it.
    
    K = 100
    
    c = np.zeros(K)
    for k in range(1, K+1):
        c[k-1] = np.log(k) / k
        
    A_ub = []
    b_ub = []
    
    max_x = 10 * K
    for x in range(1, max_x + 1):
        row = np.zeros(K)
        for k in range(1, K+1):
            row[k-1] = x // k
        A_ub.append(row)
        b_ub.append(1.0001)
        
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    # Equality constraint: sum(f_k / k) = 0
    A_eq = np.zeros((1, K))
    for k in range(1, K+1):
        A_eq[0, k-1] = 1.0 / k
    b_eq = np.array([0.0])
    
    # Bounds: -10 <= f_k <= 10
    bounds = [(-10, 10)] * K
    
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    if res.status != 0:
        # Fallback to seed
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values
        
    f = res.x
    domain = np.arange(1, K+1, dtype=np.int64)
    values = f.astype(np.float64)
    
    # Clip (should be within bounds, but just in case)
    values = np.clip(values, -10, 10)
    
    # Re-verify feasibility
    M = np.zeros(max_x + 1, dtype=np.float64)
    for i, k in enumerate(domain):
        f_k = values[i]
        if f_k == 0:
            continue
        q_max = max_x // k
        for q in range(1, q_max + 1):
            start = q * k
            end = min((q+1) * k - 1, max_x)
            if start <= max_x:
                M[start:end+1] += f_k * q
                
    if np.max(M[1:]) > 1.0001:
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values
        
    return domain, values
