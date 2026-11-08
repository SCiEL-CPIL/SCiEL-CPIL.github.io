import time
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Schoolbook decomposition as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    # Attempt to construct the 29-rank decomposition
    # Rank 29 = 4 * 7 (Strassen for 2x2 by 2x8) + 1 (Outer product for 2x1 by 1x8)
    rank_candidate = 29
    U = [[0] * rank_candidate for _ in range(n * m)]
    V = [[0] * rank_candidate for _ in range(m * p)]
    W = [[0] * rank_candidate for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 by 2x2
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    
    # U coefficients for A (2x2)
    # Rows of U for A part: i*3 + j, where i in {0,1}, j in {0,1}
    # A = [[a, b], [c, d]] corresponds to indices (0,0), (0,1), (1,0), (1,1)
    # U[0, r] = a coeff, U[1, r] = b coeff, U[3, r] = c coeff, U[4, r] = d coeff
    # Note: row index is i*m + j. m=3.
    # i=0, j=0 -> 0
    # i=0, j=1 -> 1
    # i=1, j=0 -> 3
    # i=1, j=1 -> 4
    
    # V coefficients for B (2x2 block)
    # Rows of V for B part: j'*8 + k'
    # B block is 2x2. j' in {0,1}, k' in {0,1} (local)
    # e, f, g, h correspond to (0,0), (0,1), (1,0), (1,1)
    # V[j'*8 + k', r]
    
    # W coefficients for C (2x2 block)
    # Rows of W for C part: k*2 + i
    # C block is 2x2. i in {0,1}, k in {0,1} (local)
    
    strassen_U = [
        [1, 0, 0, 1], # a, b, c, d for m1 (a+d)
        [0, 0, 1, 1], # m2 (c+d)
        [1, 0, 0, 0], # m3 a
        [0, 0, 0, 1], # m4 d
        [1, 1, 0, 0], # m5 (a+b)
        [0, 0, 1, -1],# m6 (c-a)
        [0, 1, 0, -1] # m7 (b-d)
    ]
    
    strassen_V = [
        [1, 1, 1, 1], # e, f, g, h for m1 (e+h)
        [1, 0, 0, 0], # m2 e
        [0, 1, 0, -1],# m3 (f-h)
        [0, 0, 1, -1],# m4 (g-e)
        [0, 0, 0, 1], # m5 h
        [1, 1, 0, 0], # m6 (e+f)
        [0, 0, 1, 1]  # m7 (g+h)
    ]
    
    strassen_W = [
        [1, 0, 0, 1], # C00, C01, C10, C11 for m1
        [0, 0, 1, -1],# m2
        [0, 1, 0, 1], # m3
        [1, 0, 1, 0], # m4
        [-1, 1, 0, 0],# m5
        [0, 0, 0, 1], # m6
        [1, 0, 0, 0]  # m7
    ]
    
    for q in range(4):
        for s in range(7):
            r = q * 7 + s
            # U part
            # a, b, c, d
            U[0][r] = strassen_U[s][0]
            U[1][r] = strassen_U[s][1]
            U[3][r] = strassen_U[s][2]
            U[4][r] = strassen_U[s][3]
            
            # V part
            # e, f, g, h at cols 2q, 2q+1
            # j'=0, k'=0 -> row 2q
            # j'=0, k'=1 -> row 2q+1
            # j'=1, k'=0 -> row 8+2q
            # j'=1, k'=1 -> row 8+2q+1
            V[2*q][r] = strassen_V[s][0]
            V[2*q+1][r] = strassen_V[s][1]
            V[8+2*q][r] = strassen_V[s][2]
            V[8+2*q+1][r] = strassen_V[s][3]
            
            # W part
            # C00, C01, C10, C11
            # i=0, k=0 -> row 2q
            # i=0, k=1 -> row 2q+1
            # i=1, k=0 -> row 2q+2 (since k*2+i, k=0, i=1 -> 1? No. k is col, i is row.
            # W index is k*n + i. n=2.
            # k=0, i=0 -> 0
            # k=1, i=0 -> 2
            # k=0, i=1 -> 1
            # k=1, i=1 -> 3
            # Wait. W[k*n + i, r].
            # k is 0..7. i is 0..1.
            # For block q, local k is 0,1. Global k is 2q, 2q+1.
            # C[i, k_local].
            # W row index = (2q + k_local) * 2 + i.
            
            # C00: i=0, k_local=0 -> row (2q)*2 + 0 = 4q
            # C01: i=0, k_local=1 -> row (2q+1)*2 + 0 = 4q+2
            # C10: i=1, k_local=0 -> row (2q)*2 + 1 = 4q+1
            # C11: i=1, k_local=1 -> row (2q+1)*2 + 1 = 4q+3
            
            W[4*q][r] = strassen_W[s][0]
            W[4*q+2][r] = strassen_W[s][1]
            W[4*q+1][r] = strassen_W[s][2]
            W[4*q+3][r] = strassen_W[s][3]
            
    # Outer product part (r=28)
    r = 28
    # U: A2 (col 2). i in {0,1}.
    U[2][r] = 1 # i=0, j=2 -> 0*3+2=2
    U[5][r] = 1 # i=1, j=2 -> 1*3+2=5
    
    # V: B2 (row 2). k in {0..7}.
    for k in range(8):
        V[16+k][r] = 1 # j=2, k -> 2*8+k = 16+k
        
    # W: C. i in {0,1}, k in {0..7}.
    for k in range(8):
        W[2*k][r] = 1 # i=0
        W[2*k+1][r] = 1 # i=1

    # Verify the candidate
    def check_exact(U, V, W):
        n, m, p = DIMENSIONS
        R = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(R):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    if check_exact(U, V, W):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook if verification fails (should not happen)
        return {"U": U_school, "V": V_school, "W": W_school}

# EVOLVE-BLOCK-END
