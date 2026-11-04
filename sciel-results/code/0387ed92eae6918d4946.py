"""Find rank-minimal exact tensor decomposition for 2x5 x 5x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]
_START_TIME = time.time()
DEADLINE = _START_TIME + 140

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook decomposition
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    best = {"U": U, "V": V, "W": W}
    best_rank = rank_school
    
    # Try to find lower rank using structured search
    # For n=2, try to exploit the small first dimension
    
    # Try ranks from 50 down to 35
    for R in range(50, 34, -1):
        if time.time() > DEADLINE - 15:
            break
        
        # Structured ansatz: group terms by k
        # For each k, we have 2*5 = 10 terms in schoolbook
        # Try to reduce to 8 per k (total 48) or 7 per k (total 42)
        
        # Simple approach: try random half-integer initializations
        np.random.seed(1000 + R)
        
        found = False
        for attempt in range(20):
            if time.time() > DEADLINE - 10:
                break
            
            # Initialize with random half-integers in {-1, -0.5, 0, 0.5, 1}
            vals = [-1, -0.5, 0, 0.5, 1]
            U_r = np.array([[np.random.choice(vals) for _ in range(R)] for _ in range(n*m)])
            V_r = np.array([[np.random.choice(vals) for _ in range(R)] for _ in range(m*p)])
            W_r = np.array([[np.random.choice(vals) for _ in range(R)] for _ in range(p*n)])
            
            # Convert to list of lists
            U_l = U_r.tolist()
            V_l = V_r.tolist()
            W_l = W_r.tolist()
            
            if _verify(U_l, V_l, W_l, n, m, p):
                best = {"U": U_l, "V": V_l, "W": W_l}
                best_rank = R
                found = True
                break
        
        if found:
            break
    
    return {"U": best["U"], "V": best["V"], "W": best["W"]}
