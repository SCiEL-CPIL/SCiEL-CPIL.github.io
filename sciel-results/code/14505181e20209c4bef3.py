import numpy as np
import time

def construct():
    n, m, p = 3, 3, 5
    
    # The matrix multiplication tensor T is defined by T[i, j, k] = 1
    # for all i in {0..n-1}, j in {0..m-1}, k in {0..p-1}.
    # This corresponds to the bilinear map (A, B) -> C where C_ik = sum_j A_ij B_jk.
    # The all-ones tensor has rank 1.
    # Decomposition:
    # U is a vector of 1s of length n*m
    # V is a vector of 1s of length m*p
    # W is a vector of 1s of length p*n
    # Then sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1*1*1 = 1.
    
    rank = 1
    U = [[1.0] for _ in range(n * m)]
    V = [[1.0] for _ in range(m * p)]
    W = [[1.0] for _ in range(p * n)]
    
    # Verify with exact integer arithmetic
    U_int = [[int(round(x * 2)) for x in row] for row in U]
    V_int = [[int(round(x * 2)) for x in row] for row in V]
    W_int = [[int(round(x * 2)) for x in row] for row in W]
    
    is_valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(rank):
                    total += U_int[i*m+j][c] * V_int[j*p+k][c] * W_int[k*n+i][c]
                # Note: The values are half-integers in the general case, 
                # but here they are integers. 
                # The equation requires the sum to be 1.
                # If we used half-integers, we'd scale. But 1 is integer.
                if total != 1:
                    is_valid = False
                    break
        if not is_valid:
            break
            
    if is_valid:
        return {"U": U, "V": V, "W": W}
    
    # Fallback: Schoolbook
    rank_schoolbook = n * m * p
    U_s = [[0] * rank_schoolbook for _ in range(n * m)]
    V_s = [[0] * rank_schoolbook for _ in range(m * p)]
    W_s = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i * m + j][column] = 1
                V_s[j * p + k][column] = 1
                W_s[k * n + i][column] = 1
                column += 1
    return {"U": U_s, "V": V_s, "W": W_s}
