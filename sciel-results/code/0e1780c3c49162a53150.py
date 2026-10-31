import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    
    # Initialize matrices
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen coefficients for 2x2 matrices
    # S_A[r, i, j], S_B[r, k, l], S_C[r, i, j]
    # i, j, k, l in {0, 1}
    
    S_A = np.zeros((7, 2, 2), dtype=int)
    S_B = np.zeros((7, 2, 2), dtype=int)
    S_C = np.zeros((7, 2, 2), dtype=int)
    
    # r=0
    S_A[0, 0, 0] = 1; S_A[0, 1, 1] = 1
    S_B[0, 0, 0] = 1; S_B[0, 1, 1] = 1
    S_C[0, 0, 0] = 1; S_C[0, 1, 1] = 1
    
    # r=1
    S_A[1, 1, 0] = 1; S_A[1, 1, 1] = 1
    S_B[1, 0, 0] = 1
    S_C[1, 1, 0] = 1; S_C[1, 1, 1] = -1
    
    # r=2
    S_A[2, 0, 0] = 1
    S_B[2, 0, 1] = 1; S_B[2, 1, 1] = -1
    S_C[2, 0, 1] = 1; S_C[2, 1, 1] = 1
    
    # r=3
    S_A[3, 1, 1] = 1
    S_B[3, 1, 0] = 1; S_B[3, 0, 0] = -1
    S_C[3, 0, 0] = 1; S_C[3, 1, 0] = 1
    
    # r=4
    S_A[4, 0, 0] = 1; S_A[4, 0, 1] = 1
    S_B[4, 1, 1] = 1
    S_C[4, 0, 0] = -1; S_C[4, 0, 1] = 1
    
    # r=5
    S_A[5, 1, 0] = 1; S_A[5, 0, 0] = -1
    S_B[5, 0, 0] = 1; S_B[5, 0, 1] = 1
    S_C[5, 1, 1] = 1
    
    # r=6
    S_A[6, 0, 1] = 1; S_A[6, 1, 1] = -1
    S_B[6, 1, 0] = 1; S_B[6, 1, 1] = 1
    S_C[6, 0, 0] = 1
    
    # Construct U, V, W
    col = 0
    for r in range(7):
        for s in range(7):
            # U contributions
            # U[2*I+i, 2*J+j] += S_A2[s, i, j] * S_A[r, I, J]
            for I in range(2):
                for J in range(2):
                    if S_A[r, I, J] == 0: continue
                    for i in range(2):
                        for j in range(2):
                            if S_A[s, i, j] == 0: continue
                            row_u = 2*I + i
                            col_u = 2*J + j
                            # Wait, U index is [i*m+j, c]. Here m=4.
                            # A is 4x4. Index i (row), j (col).
                            # U row index: i * m + j = i * 4 + j.
                            # My I, J are block indices. i, j are local indices.
                            # Global row: 2*I + i. Global col: 2*J + j.
                            # So U row index is (2*I + i) * 4 + (2*J + j).
                            U[(2*I + i) * 4 + (2*J + j), col] += S_A[s, i, j] * S_A[r, I, J]
                            
            # V contributions
            # V[2*K+k, 2*L+l] += S_B2[s, k, l] * S_B[r, K, L]
            for K in range(2):
                for L in range(2):
                    if S_B[r, K, L] == 0: continue
                    for k in range(2):
                        for l in range(2):
                            if S_B[s, k, l] == 0: continue
                            # V row index: k * p + l = k * 4 + l.
                            V[(2*K + k) * 4 + (2*L + l), col] += S_B[s, k, l] * S_B[r, K, L]
                            
            # W contributions
            # W[2*I+i, 2*J+j] += S_C[r, I, J] * S_C2[s, i, j]
            # W index is [k*n+i, r]. Here output is C.
            # C row: 2*I + i. C col: 2*J + j.
            # W row index: (2*J + j) * 4 + (2*I + i). (k*n+i where k=col, i=row? No, W[k*n+i, r] is standard?
            # Task says: W[k*n+i,r]. i is row, k is col?
            # "output factor follows the official trace convention W[k*n+i,r]"
            # Usually tensor T[i,j,k] = sum U[i,r] V[j,r] W[k,r].
            # Here A is n x m, B is m x p. C is p x n? No, C is n x p.
            # A (n x m) * B (m x p) = C (n x p).
            # T[i, j, k] where i in A, j in B, k in C.
            # C[k, i] = sum_j A[i, j] B[j, k].
            # So W index corresponds to C.
            # If W is (p*n, rank), then row index is k*n + i.
            # k is col of C (0..3), i is row of C (0..3).
            # So W[(2*J + j) * 4 + (2*I + i), col].
            
            for I in range(2):
                for J in range(2):
                    if S_C[r, I, J] == 0: continue
                    for i in range(2):
                        for j in range(2):
                            if S_C[s, i, j] == 0: continue
                            # Global C row: 2*I + i. Global C col: 2*J + j.
                            # W row index: col * n + row = (2*J + j) * 4 + (2*I + i).
                            W[(2*J + j) * 4 + (2*I + i), col] += S_C[r, I, J] * S_C[s, i, j]
                            
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
