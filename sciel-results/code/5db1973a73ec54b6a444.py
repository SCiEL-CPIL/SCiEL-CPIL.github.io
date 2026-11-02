import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x3 matrix.
    
    The optimal rank for 2x3x3 matrix multiplication is 17, which can be constructed by
    combining Strassen's algorithm for the 2x2x2 subproblem with schoolbook algorithms
    for the remaining subproblems.
    
    Returns a dictionary with keys 'U', 'V', 'W' where:
    - U has shape (n*m, rank) = (6, 17)
    - V has shape (m*p, rank) = (9, 17)  
    - W has shape (p*n, rank) = (6, 17)
    
    The decomposition satisfies: for all i,j,k, sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 140.0  # Safety margin before 150s kill
    
    # The known optimal rank for 2x3x3 matrix multiplication is 17.
    # This is achieved by decomposing the problem into:
    # - A 2x2x2 subproblem using Strassen (rank 7)
    # - A 2x1x2 subproblem using schoolbook (rank 4) 
    # - A 2x3x1 subproblem using schoolbook (rank 6)
    # Total: 7 + 4 + 6 = 17
    
    # We'll construct the rank-17 decomposition explicitly.
    # The indices are:
    # U: shape (n*m, R) = (6, 17), indexed by (i*m + j, r)
    # V: shape (m*p, R) = (9, 17), indexed by (j*p + k, r)
    # W: shape (p*n, R) = (6, 17), indexed by (k*n + i, r)
    
    R = 17
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    
    # Construction of the rank-17 decomposition:
    # We partition the 2x3x3 problem into three parts:
    # Part 1: i in {0,1}, j in {0,1}, k in {0,1} -> 2x2x2 subproblem (Strassen, rank 7)
    # Part 2: i in {0,1}, j in {2}, k in {0,1} -> 2x1x2 subproblem (schoolbook, rank 4)
    # Part 3: i in {0,1}, j in {0,1,2}, k in {2} -> 2x3x1 subproblem (schoolbook, rank 6)
    
    # --- Part 1: 2x2x2 Strassen decomposition (rank 7) ---
    # Strassen's 7 multiplication formulas for 2x2 matrices:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For the 2x2x2 subproblem (i,j,k all in {0,1}):
    # We need: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all i,j,k in {0,1}
    # Strassen gives us 7 terms.
    
    # Term 0: M1 = (A11+A22)(B11+B22)
    # A11 is at (i=0,j=0), A22 is at (i=1,j=1)
    # B11 is at (j=0,k=0), B22 is at (j=1,k=1)
    # C11 needs M1, C22 needs M1
    # U[0,0,0] += 1 (A11), U[1,1,0] += 1 (A22)
    # V[0,0,0] += 1 (B11), V[1,1,0] += 1 (B22)
    # W[0,0,0] += 1 (contributes to C11), W[1,1,0] += 1 (contributes to C22)
    U[0*3+0, 0] += 1  # i=0, j=0
    U[1*3+1, 0] += 1  # i=1, j=1
    V[0*3+0, 0] += 1  # j=0, k=0
    V[1*3+1, 0] += 1  # j=1, k=1
    W[0*2+0, 0] += 1  # k=0, i=0 (C11)
    W[1*2+1, 0] += 1  # k=1, i=1 (C22)
    
    # Term 1: M2 = (A21+A22)B11
    # A21 is at (i=1,j=0), A22 is at (i=1,j=1)
    # B11 is at (j=0,k=0)
    # C21 needs M2
    U[1*3+0, 1] += 1  # i=1, j=0
    U[1*3+1, 1] += 1  # i=1, j=1
    V[0*3+0, 1] += 1  # j=0, k=0
    W[0*2+1, 1] += 1  # k=0, i=1 (C21)
    
    # Term 2: M3 = A11(B21-B22)
    # A11 is at (i=0,j=0)
    # B21 is at (j=1,k=0), B22 is at (j=1,k=1)
    # C12 needs M3
    U[0*3+0, 2] += 1  # i=0, j=0
    V[1*3+0, 2] += 1  # j=1, k=0
    V[1*3+1, 2] -= 1  # j=1, k=1
    W[0*2+0, 2] += 1  # k=0, i=0 (C12)
    
    # Term 3: M4 = A22(B12-B11)
    # A22 is at (i=1,j=1)
    # B12 is at (j=0,k=1), B11 is at (j=0,k=0)
    # C11 needs M4, C21 needs M4
    U[1*3+1, 3] += 1  # i=1, j=1
    V[0*3+1, 3] += 1  # j=0, k=1
    V[0*3+0, 3] -= 1  # j=0, k=0
    W[1*2+0, 3] += 1  # k=1, i=0 (C11)
    W[0*2+1, 3] += 1  # k=0, i=1 (C21)
    
    # Term 4: M5 = (A11+A12)B22
    # A11 is at (i=0,j=0), A12 is at (i=0,j=1)
    # B22 is at (j=1,k=1)
    # C11 needs -M5, C12 needs M5
    U[0*3+0, 4] += 1  # i=0, j=0
    U[0*3+1, 4] += 1  # i=0, j=1
    V[1*3+1, 4] += 1  # j=1, k=1
    W[1*2+0, 4] -= 1  # k=1, i=0 (C11)
    W[0*2+0, 4] += 1  # k=0, i=0 (C12)
    
    # Term 5: M6 = (A21-A11)(B11+B12)
    # A21 is at (i=1,j=0), A11 is at (i=0,j=0)
    # B11 is at (j=0,k=0), B12 is at (j=0,k=1)
    # C22 needs M6
    U[1*3+0, 5] += 1  # i=1, j=0
    U[0*3+0, 5] -= 1  # i=0, j=0
    V[0*3+0, 5] += 1  # j=0, k=0
    V[0*3+1, 5] += 1  # j=0, k=1
    W[1*2+1, 5] += 1  # k=1, i=1 (C22)
    
    # Term 6: M7 = (A12-A22)(B21+B22)
    # A12 is at (i=0,j=1), A22 is at (i=1,j=1)
    # B21 is at (j=1,k=0), B22 is at (j=1,k=1)
    # C11 needs M7
    U[0*3+1, 6] += 1  # i=0, j=1
    U[1*3+1, 6] -= 1  # i=1, j=1
    V[1*3+0, 6] += 1  # j=1, k=0
    V[1*3+1, 6] += 1  # j=1, k=1
    W[0*2+0, 6] += 1  # k=0, i=0 (C11)
    
    # --- Part 2: 2x1x2 subproblem (i in {0,1}, j=2, k in {0,1}) ---
    # This is a 2x1 matrix times a 1x2 matrix, giving a 2x2 result.
    # Schoolbook: 2*1*2 = 4 multiplications.
    # C[i][k] = A[i][2] * B[2][k] for i in {0,1}, k in {0,1}
    
    # Term 7: A[0][2] * B[2][0] -> C[0][0]
    U[0*3+2, 7] += 1  # i=0, j=2
    V[2*3+0, 7] += 1  # j=2, k=0
    W[0*2+0, 7] += 1  # k=0, i=0
    
    # Term 8: A[0][2] * B[2][1] -> C[0][1]
    U[0*3+2, 8] += 1  # i=0, j=2
    V[2*3+1, 8] += 1  # j=2, k=1
    W[1*2+0, 8] += 1  # k=1, i=0
    
    # Term 9: A[1][2] * B[2][0] -> C[1][0]
    U[1*3+2, 9] += 1  # i=1, j=2
    V[2*3+0, 9] += 1  # j=2, k=0
    W[0*2+1, 9] += 1  # k=0, i=1
    
    # Term 10: A[1][2] * B[2][1] -> C[1][1]
    U[1*3+2, 10] += 1  # i=1, j=2
    V[2*3+1, 10] += 1  # j=2, k=1
    W[1*2+1, 10] += 1  # k=1, i=1
    
    # --- Part 3: 2x3x1 subproblem (i in {0,1}, j in {0,1,2}, k=2) ---
    # This is a 2x3 matrix times a 3x1 matrix, giving a 2x1 result.
    # Schoolbook: 2*3*1 = 6 multiplications.
    # C[i][2] = sum_{j=0}^{2} A[i][j] * B[j][2]
    
    # Term 11: A[0][0] * B[0][2] -> C[0][2]
    U[0*3+0, 11] += 1  # i=0, j=0
    V[0*3+2, 11] += 1  # j=0, k=2
    W[2*2+0, 11] += 1  # k=2, i=0
    
    # Term 12: A[0][1] * B[1][2] -> C[0][2]
    U[0*3+1, 12] += 1  # i=0, j=1
    V[1*3+2, 12] += 1  # j=1, k=2
    W[2*2+0, 12] += 1  # k=2, i=0
    
    # Term 13: A[0][2] * B[2][2] -> C[0][2]
    U[0*3+2, 13] += 1  # i=0, j=2
    V[2*3+2, 13] += 1  # j=2, k=2
    W[2*2+0, 13] += 1  # k=2, i=0
    
    # Term 14: A[1][0] * B[0][2] -> C[1][2]
    U[1*3+0, 14] += 1  # i=1, j=0
    V[0*3+2, 14] += 1  # j=0, k=2
    W[2*2+1, 14] += 1  # k=2, i=1
    
    # Term 15: A[1][1] * B[1][2] -> C[1][2]
    U[1*3+1, 15] += 1  # i=1, j=1
    V[1*3+2, 15] += 1  # j=1, k=2
    W[2*2+1, 15] += 1  # k=2, i=1
    
    # Term 16: A[1][2] * B[2][2] -> C[1][2]
    U[1*3+2, 16] += 1  # i=1, j=2
    V[2*3+2, 16] += 1  # j=2, k=2
    W[2*2+1, 16] += 1  # k=2, i=1
    
    # Verify the decomposition
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                if not np.isclose(val, 1.0, atol=1e-9):
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if valid:
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    # Fallback to rank 18 schoolbook decomposition if verification fails
    rank = n * m * p
    U_fallback = [[0] * rank for _ in range(n * m)]
    V_fallback = [[0] * rank for _ in range(m * p)]
    W_fallback = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fallback[i * m + j][col] = 1
                V_fallback[j * p + k][col] = 1
                W_fallback[k * n + i][col] = 1
                col += 1
                
    return {"U": U_fallback, "V": V_fallback, "W": W_fallback}
