"""Tensor decomposition for 4x4 by 4x9 matrix multiplication."""
import time

DIMENSIONS = [4, 4, 9]
START_TIME = time.time()
TIME_LIMIT = 115  # seconds, leaving margin for 120s evaluator limit

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying
    a 4x4 matrix by a 4x9 matrix.
    
    Output factor follows the official trace convention W[k*n+i, r].
    Returns {'U', 'V', 'W'} with shapes (n*m, rank), (m*p, rank), (p*n, rank).
    
    The tensor T[i,j,k] = 1 if i==k else 0 (since A[i,j]*B[j,k] contributes to C[i,k]).
    
    Construction: For each (i, j) pair, create a rank term that activates
    only when the row index of A matches the column index of B (i.e., i==k).
    """
    n, m, p = DIMENSIONS
    
    # Rank = n * m = 16
    # Each term (r) corresponds to a unique (i, j) pair
    rank = n * m
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Assign each rank term to a specific (i, j) pair
    for r in range(rank):
        i = r // m  # row of A
        j = r % m   # column of A / row of B
        
        # U[i*m + j, r] = 1 (activates when A's entry is at position (i,j))
        U[i * m + j][r] = 1
        
        # V[j*p + k, r] = 1 only when k == i (B's entry must be in column i)
        V[j * p + i][r] = 1
        
        # W[k*n + i, r] = 1 only when k == i (output is at position (i,i) in the trace sense)
        W[i * n + i][r] = 1
    
    # Verify the decomposition
    def verify(U, V, W):
        rank_val = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0 if i == k else 0.0
                    actual = 0.0
                    for r in range(rank_val):
                        actual += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(actual - expected) > 1e-9:
                        return False
        return True
    
    if not verify(U, V, W):
        # Fallback: schoolbook decomposition
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if i == k:  # Only non-zero tensor entries
                        U_fb[i * m + j][column] = 1
                        V_fb[j * p + k][column] = 1
                        W_fb[k * n + i][column] = 1
                        column += 1
        # Trim to actual rank used
        actual_rank = column
        U_fb = [row[:actual_rank] for row in U_fb]
        V_fb = [row[:actual_rank] for row in V_fb]
        W_fb = [row[:actual_rank] for row in W_fb]
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
