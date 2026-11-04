import time
import numpy as np

DIMENSIONS = [3, 3, 3]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n, m, p = DIMENSIONS
    # We will use Rank 26
    rank = 26
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # 1. Add schoolbook terms for all (i, j, k) where max(i, j, k) == 2
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if max(i, j, k) == 2:
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
                    
    # 2. Add Strassen terms for the 2x2x2 block (i, j, k in {0, 1})
    # Strassen's 7 products:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A10(B00 - B11)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # We need to map these to U, V, W vectors.
    # U is A part, V is B part, W is C part (coefficients for C_ki? No, C_ik).
    # Wait, W index is k*n + i. So W[k, i] is the coefficient for C[i, k]?
    # Let's check the convention: W[k*n+i, r].
    # In matrix mult, C[i, k] = sum_j A[i, j] B[j, k].
    # The tensor T[i, j, k] = 1.
    # The decomposition is sum_r U[i, j, r] V[j, k, r] W[k, i, r] ?
    # If W is indexed by [k, i], then the term is U_ij * V_jk * W_ki.
    # This sums to C_ik?
    # Let's verify: sum_r (linear comb of A) * (linear comb of B) * (linear comb of C positions).
    # Yes, if W_ki is 1, it adds to C[i, k].
    
    # Strassen formulas for C:
    # C00 = M1 - M4 - M6 + M7
    # C10 = M2 + M4
    # C01 = M3 + M6
    # C11 = M1 - M3 + M5 - M7
    
    # So for each M_r, we define U_r, V_r, and W_r contributions.
    
    # M1
    U[0 * m + 0][col] = 1; U[1 * m + 1][col] = 1
    V[0 * p + 0][col] = 1; V[1 * p + 1][col] = 1
    W[0 * n + 0][col] = 1; W[1 * n + 1][col] = 1
    col += 1
    
    # M2
    U[1 * m + 0][col] = 1; U[1 * m + 1][col] = 1
    V[0 * p + 0][col] = 1
    W[0 * n + 1][col] = 1  # C10
    col += 1
    
    # M3
    U[0 * m + 0][col] = 1
    V[1 * p + 0][col] = 1; V[1 * p + 1][col] = -1
    W[1 * n + 0][col] = 1  # C01
    col += 1
    
    # M4
    U[1 * m + 0][col] = 1
    V[0 * p + 0][col] = 1; V[1 * p + 1][col] = -1
    W[0 * n + 1][col] = 1  # C10
    W[1 * n + 1][col] = -1 # C11
    col += 1
    
    # M5
    U[0 * m + 0][col] = 1; U[0 * m + 1][col] = 1
    V[1 * p + 1][col] = 1
    W[1 * n + 1][col] = 1  # C11
    col += 1
    
    # M6
    U[1 * m + 0][col] = 1; U[0 * m + 0][col] = -1
    V[0 * p + 0][col] = 1; V[0 * p + 1][col] = 1
    W[0 * n + 0][col] = -1 # C00
    W[1 * n + 0][col] = 1  # C01
    col += 1
    
    # M7
    U[0 * m + 1][col] = 1; U[1 * m + 1][col] = -1
    V[1 * p + 0][col] = 1; V[1 * p + 1][col] = 1
    W[0 * n + 0][col] = 1  # C00
    W[1 * n + 1][col] = -1 # C11
    col += 1
    
    if col != rank:
        # Should not happen, but resize if logic is off
        pass
        
    return {"U": U, "V": V, "W": W}
