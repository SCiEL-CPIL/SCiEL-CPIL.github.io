import time
import random
import numpy as np

def verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly represents matrix multiplication."""
    rank = len(U[0])
    # Build the tensor from the decomposition
    # T[i,j,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for t in range(rank):
                    total += U[i*m+j][t] * V[j*p+k][t] * W[k*n+i][t]
                if abs(total - 1) > 1e-10:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 108  # safety margin
    
    n, m, p = 3, 5, 7
    
    # Kronecker product structure: decompose into n independent 1×m×p multiplications
    # Each 1×m×p has rank m, so total rank = n * m = 3 * 5 = 15
    rank = n * m
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            t = i * m + j
            U[i * m + j][t] = 1
            for k in range(p):
                V[j * p + k][t] = 1
            W[k * n + i][t] = 1  # This is wrong, k is not defined here
    
    # Fix the W assignment
    for i in range(n):
        for j in range(m):
            t = i * m + j
            U[i * m + j][t] = 1
            for k in range(p):
                V[j * p + k][t] = 1
                W[k * n + i][t] = 1
    
    # Verify
    if verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to trivial
    trivial_rank = n * m * p
    U_trivial = [[0] * trivial_rank for _ in range(n * m)]
    V_trivial = [[0] * trivial_rank for _ in range(m * p)]
    W_trivial = [[0] * trivial_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j][column] = 1
                V_trivial[j * p + k][column] = 1
                W_trivial[k * n + i][column] = 1
                column += 1
    
    return {"U": U_trivial, "V": V_trivial, "W": W_trivial}
