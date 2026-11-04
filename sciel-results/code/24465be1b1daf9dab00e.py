"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill

    n, m, p = DIMENSIONS  # 3, 5, 5

    # Use Strassen on a 2x2x2 sub-block (indices i=0,1; j=0,1; k=0,1)
    # and schoolbook for the remaining 67 terms.
    # Total rank: 7 + 67 = 74

    # Strassen factors for 2x2x2
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[ae+bg, af+bh],[ce+dg, cf+dh]]
    # M1 = (a+d)(e+h), M2 = (c+d)e, M3 = a(f-h), M4 = d(g-e)
    # M5 = (a+b)h, M6 = (c-a)(e+f), M7 = (b-d)(g+h)
    # C11 = M1+M4-M5+M7, C12 = M3+M5, C21 = M2+M4, C22 = M1-M2+M3+M6

    U_strassen = [
        [1, 0, 1, 0, 1, -1, 0],   # a
        [0, 0, 0, 0, 1, 0, 1],    # b
        [0, 1, 0, 0, 0, 1, 0],    # c
        [1, 1, 0, 1, 0, 0, -1]    # d
    ]
    V_strassen = [
        [1, 1, 0, -1, 0, 1, 0],   # e
        [0, 0, 1, 0, 0, 1, 0],    # f
        [0, 0, 0, 1, 0, 0, 1],    # g
        [1, 0, -1, 0, 1, 0, 1]    # h
    ]
    W_strassen = [
        [1, 0, 0, 1, -1, 0, 1],   # C11
        [0, 0, 1, 0, 1, 0, 0],    # C12
        [0, 1, 0, 1, 0, 0, 0],    # C21
        [1, -1, 1, 0, 0, 1, 0]    # C22
    ]

    # Indices for the 2x2x2 sub-block
    i_idx = [0, 1]  # rows of A
    j_idx = [0, 1]  # inner dimension
    k_idx = [0, 1]  # columns of B

    # Build the full decomposition
    rank_total = 74
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]

    # Place Strassen factors in columns 0-6
    for s in range(4):
        U[i_idx[s % 2] * m + j_idx[s // 2]][s] = U_strassen[s][0]
        # Wait, I need to map correctly.
        # U_strassen row 0 = a = A[0,0] -> U[0*m+0]
        # U_strassen row 1 = b = A[0,1] -> U[0*m+1]
        # U_strassen row 2 = c = A[1,0] -> U[1*m+0]
        # U_strassen row 3 = d = A[1,1] -> U[1*m+1]
        pass

    # Let me redo this more carefully
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]

    # Strassen: map local indices to global
    # a = A[i0,j0] where i0=0, j0=0 -> U row = i0*m+j0 = 0
    # b = A[i0,j1] where i0=0, j1=1 -> U row = 0*m+1 = 1
    # c = A[i1,j0] where i1=1, j0=0 -> U row = 1*m+0 = 5
    # d = A[i1,j1] where i1=1, j1=1 -> U row = 1*m+1 = 6
    # e = B[j0,k0] where j0=0, k0=0 -> V row = j0*p+k0 = 0
    # f = B[j0,k1] where j0=0, k1=1 -> V row = 0*p+1 = 1
    # g = B[j1,k0] where j1=1, k0=0 -> V row = 1*p+0 = 5
    # h = B[j1,k1] where j1=1, k1=1 -> V row = 1*p+1 = 6
    # C11 = C[i0,k0] where i0=0,k0=0 -> W row = k0*n+i0 = 0
    # C12 = C[i0,k1] where i0=0,k1=1 -> W row = 1*n+0 = 3
    # C21 = C[i1,k0] where i1=1,k0=0 -> W row = 0*n+1 = 1
    # C22 = C[i1,k1] where i1=1,k1=1 -> W row = 1*n+1 = 4

    u_rows = [0, 1, 5, 6]  # a, b, c, d
    v_rows = [0, 1, 5, 6]  # e, f, g, h
    w_rows = [0, 3, 1, 4]  # C11, C12, C21, C22

    for col in range(7):
        for s in range(4):
            U[u_rows[s]][col] = U_strassen[s][col]
            V[v_rows[s]][col] = V_strassen[s][col]
            W[w_rows[s]][col] = W_strassen[s][col]

    # Schoolbook for remaining 67 terms (all (i,j,k) not in the 2x2x2 sub-block)
    col = 7
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i in i_idx and j in j_idx and k in k_idx:
                    continue  # already handled by Strassen
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    # col should be 74
    assert col == rank_total, f"Expected rank {rank_total}, got {col}"

    # Verify correctness
    def verify_decomposition(U_mat, V_mat, W_mat, n, m, p):
        U = np.array(U_mat)
        V = np.array(V_mat)
        W = np.array(W_mat)
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(rank):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if val != 1:
                        return False
        return True

    if not verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook
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
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
