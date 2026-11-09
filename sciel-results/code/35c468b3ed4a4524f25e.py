import time
import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank 28 decomposition
    # U: (n*m, R), V: (m*p, R), W: (p*n, R)
    # Indices:
    # A: i in [0, n), j in [0, m). Flat: i*m + j
    # B: j in [0, m), k in [0, p). Flat: j*p + k
    # C: i in [0, n), k in [0, p). Flat: k*n + i (Trace convention)
    
    R = 28
    U = [[0.0] * R for _ in range(n * m)]
    V = [[0.0] * R for _ in range(m * p)]
    W = [[0.0] * R for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(u_indices, v_indices, w_indices, coeffs):
        nonlocal col
        if col >= R:
            return
        for idx, coeff in zip(u_indices, coeffs):
            U[idx][col] += coeff
        for idx, coeff in zip(v_indices, coeffs):
            V[idx][col] += coeff
        for idx, coeff in zip(w_indices, coeffs):
            W[idx][col] += coeff
        col += 1

    # 1. A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # Strassen
    # A indices: 0, 1, 3, 4 (i*m+j)
    # B indices: 0, 5, 10, 11 (j*p+k)
    # C indices: 0, 1, 2, 3 (k*n+i)
    
    # M1 = (A00+A11)(B00+B11)
    add_term([0, 4], [0, 11], [0, 3], [1, 1, 1, 1]) # Wait, coeffs apply to u, v, w separately?
    # The term is (sum u_i A_i) * (sum v_j B_j) * (sum w_k C_k)
    # So u = [1, 1], v = [1, 1], w = [1, 1]
    # This adds 1 to U[0], U[4], V[0], V[11], W[0], W[3]
    
    # M2 = (A10+A11)B00
    add_term([3, 4], [0], [1, 2], [1, 1, 1, 1]) # C10 (idx 1), C11 (idx 2)? No.
    # C10 is i=1, k=0 -> 0*2+1 = 1.
    # C11 is i=1, k=1 -> 1*2+1 = 3? No.
    # C indices:
    # k=0, i=0 -> 0
    # k=0, i=1 -> 1
    # k=1, i=0 -> 2
    # k=1, i=1 -> 3
    
    # M2 contributes to C10 and C11?
    # M2 = (A10+A11)B00
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # So M2 has +1 for C10 (idx 1) and -1 for C11 (idx 3).
    
    # Let's restart the Strassen mapping carefully.
    
    # Reset col
    col = 0
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2]
    # A: 0,1,3,4
    # B: 0,5,10,11
    # C: 0,1,2,3
    
    # M1
    add_term([0, 4], [0, 11], [0, 3], [1, 1, 1, 1])
    # M2
    add_term([3, 4], [0], [1, 3], [1, 1, 1, -1])
    # M3
    add_term([0], [5, 11], [2, 3], [1, 1, -1, 1]) # B10-B11? No, B[0,1]-B[1,1]
    # B[0,1] is idx 5. B[1,1] is idx 11.
    # M3 = A00 (B01 - B11)
    # C01 (idx 2) += M3
    # C11 (idx 3) += M3
    # So w = [0, 1, 1, 0] at indices 2, 3?
    # w[2] += 1, w[3] += 1.
    # My add_term takes u_indices, v_indices, w_indices, coeffs.
    # coeffs are for u, then v, then w?
    # The function I wrote above is ambiguous.
    
    # Let's write a clearer helper.
    
    # Redefine add_term to take lists of (index, coeff) for each matrix.
    
    # I will just hardcode the 28 terms.
    
    # Term 1: M1
    U[0][0] = 1; U[4][0] = 1
    V[0][0] = 1; V[11][0] = 1
    W[0][0] = 1; W[3][0] = 1
    col = 1
    
    # Term 2: M2
    U[3][1] = 1; U[4][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    col = 2
    
    # Term 3: M3
    U[0][2] = 1
    V[5][2] = 1; V[11][2] = -1
    W[2][2] = 1; W[3][2] = 1
    col = 3
    
    # Term 4: M4
    U[4][3] = 1
    V[10][3] = 1; V[0][3] = -1
    W[1][3] = 1; W[0][3] = -1
    col = 4
    
    # Term 5: M5
    U[0][4] = 1; U[1][4] = 1
    V[11][4] = 1
    W[0][4] = -1; W[2][4] = 1
    col = 5
    
    # Term 6: M6
    U[3][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[5][5] = 1
    W[3][5] = 1
    col = 6
    
    # Term 7: M7
    U[1][6] = 1; U[4][6] = -1
    V[10][6] = 1; V[11][6] = 1
    W[0][6] = 1; W[2][6] = -1
    col = 7
    
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4]
    # B cols 2,3. Indices:
    # j=0, k=2 -> 2
    # j=0, k=3 -> 3
    # j=1, k=2 -> 7
    # j=1, k=3 -> 8
    # C cols 2,3. Indices:
    # k=2, i=0 -> 4
    # k=2, i=1 -> 5
    # k=3, i=0 -> 6
    # k=3, i=1 -> 7
    
    # M1'
    U[0][7] = 1; U[4][7] = 1
    V[2][7] = 1; V[8][7] = 1
    W[4][7] = 1; W[7][7] = 1
    col = 8
    
    # M2'
    U[3][8] = 1; U[4][8] = 1
    V[2][8] = 1
    W[5][8] = 1; W[7][8] = -1
    col = 9
    
    # M3'
    U[0][9] = 1
    V[3][9] = 1; V[8][9] = -1
    W[6][9] = 1; W[7][9] = 1
    col = 10
    
    # M4'
    U[4][10] = 1
    V[7][10] = 1; V[2][10] = -1
    W[5][10] = 1; W[4][10] = -1
    col = 11
    
    # M5'
    U[0][11] = 1; U[1][11] = 1
    V[8][11] = 1
    W[4][11] = -1; W[6][11] = 1
    col = 12
    
    # M6'
    U[3][12] = 1; U[0][12] = -1
    V[2][12] = 1; V[3][12] = 1
    W[7][12] = 1
    col = 13
    
    # M7'
    U[1][13] = 1; U[4][13] = -1
    V[7][13] = 1; V[8][13] = 1
    W[4][13] = 1; W[6][13] = -1
    col = 14
    
    # Block 3: A[0:2, 0:2] * B[0:2, 4:5]
    # B col 4. Indices:
    # j=0, k=4 -> 4
    # j=1, k=4 -> 9
    # C col 4. Indices:
    # k=4, i=0 -> 8
    # k=4, i=1 -> 9
    
    # Schoolbook for 2x2 * 2x1
    # C0 = A00 B0 + A01 B1
    # C1 = A10 B0 + A11 B1
    
    U[0][14] = 1
    V[4][14] = 1
    W[8][14] = 1
    col = 15
    
    U[1][15] = 1
    V[4][15] = 1
    W[8][15] = 1
    col = 16
    
    U[3][16] = 1
    V[4][16] = 1
    W[9][16] = 1
    col = 17
    
    U[4][17] = 1
    V[9][17] = 1
    W[9][17] = 1
    col = 18
    
    # Block 4: A[0:2, 2:3] * B[2:3, 0:5]
    # A col 2. Indices:
    # i=0, j=2 -> 2
    # i=1, j=2 -> 5
    # B row 2. Indices:
    # j=2, k=0 -> 10
    # j=2, k=1 -> 11
    # j=2, k=2 -> 12
    # j=2, k=3 -> 13
    # j=2, k=4 -> 14
    
    # C all cols.
    # k=0: 0, 1
    # k=1: 2, 3
    # k=2: 4, 5
    # k=3: 6, 7
    # k=4: 8, 9
    
    # Outer product
    for i in range(2):
        for k in range(5):
            u_idx = i * 3 + 2
            v_idx = 2 * 5 + k
            w_idx = k * 2 + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
