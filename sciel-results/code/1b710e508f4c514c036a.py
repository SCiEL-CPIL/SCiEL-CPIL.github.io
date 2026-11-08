"""
Program for Matrix multiplication 4,4,5.
Uses the schoolbook decomposition which is guaranteed to be valid with rank n*m*p.
"""
import time

DIMENSIONS = [4, 4, 5]

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    The decomposition is exact and uses the schoolbook method with rank n*m*p.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Each schoolbook term corresponds to one (i,j,k) triple
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                
                U[idx_u][column] = 1
                V[idx_v][column] = 1
                W[idx_w][column] = 1
                column += 1
                
                # Time check (should never trigger for rank 80)
                if time.time() > deadline:
                    # Return partial result only if we've completed all terms
                    # For safety, break and return what we have
                    return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
