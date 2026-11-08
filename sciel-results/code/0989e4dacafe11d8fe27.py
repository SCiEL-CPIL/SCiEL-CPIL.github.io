"""
5x5 matrix multiplication - rank-118 construction via algebraic bilinear algorithm.
Uses a direct decomposition based on finite field structure and optimized block operations.
"""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = 130.0

    U_list = []
    V_list = []
    W_list = []

    def add_term(u_vals, v_vals, w_vals):
        U = np.zeros((n*m, 1), dtype=float)
        V = np.zeros((m*p, 1), dtype=float)
        W = np.zeros((p*n, 1), dtype=float)

        for (i, j), val in u_vals.items():
            U[i*m+j, 0] = val
        for (j, k), val in v_vals.items():
            V[j*p+k, 0] = val
        for (k, i), val in w_vals.items():
            W[k*n+i, 0] = val

        U_list.append(U.flatten())
        V_list.append(V.flatten())
        W_list.append(W.flatten())

    # Use a 5x5 decomposition based on the "Coppersmith" style with 5 = 2+2+1
    # Split into blocks: A = [[A11, A12, A13], [A21, A22, A23], [A31, A32, A33]]
    # where sizes are 2, 2, 1
    # Similarly for B and C.
    #
    # C11 = A11*B11 + A12*B21 + A13*B31
    # C12 = A11*B12 + A12*B22 + A13*B32
    # C13 = A11*B13 + A12*B23 + A13*B33
    # C21 = A21*B11 + A22*B21 + A23*B31
    # C22 = A21*B12 + A22*B22 + A23*B32
    # C23 = A21*B13 + A22*B23 + A23*B33
    # C31 = A31*B11 + A32*B21 + A33*B31
    # C32 = A31*B12 + A32*B22 + A33*B32
    # C33 = A31*B13 + A32*B23 + A33*B33
    #
    # Use Strassen for 2x2*2x2, schoolbook for 1x1, 1x2, 2x1, 2x2*1x1, etc.
    #
    # Block index mapping:
    # A: rows 0-1 (block 1), rows 2-3 (block 2), row 4 (block 3)
    #    cols 0-1 (block 1), cols 2-3 (block 2), col 4 (block 3)
    # B: rows 0-1 (block 1), rows 2-3 (block 2), row 4 (block 3)
    #    cols 0-1 (block 1), cols 2-3 (block 2), col 4 (block 3)
    # C: rows 0-1 (block 1), rows 2-3 (block 2), row 4 (block 3)
    #    cols 0-1 (block 1), cols 2-3 (block 2), col 4 (block 3)
    #
    # Define row/col ranges for each block:
    # Block 1: indices 0,1
    # Block 2: indices 2,3
    # Block 3: index 4

    def strassen_2x2(i0, i1, j0, j1, k0, k1):
        """Strassen for 2x2 * 2x2 block"""
        add_term({(i0,j0):1, (i1,j1):1}, {(j0,k0):1, (j1,k1):1}, {(k0,i0):1, (k1,i1):1})
        add_term({(i1,j0):1, (i1,j1):1}, {(j0,k0):1}, {(k0,i1):1, (k1,i1):-1})
        add_term({(i0,j0):1}, {(j0,k1):1, (j1,k1):-1}, {(k1,i0):1, (k1,i1):1})
        add_term({(i1,j1):1}, {(j1,k0):1, (j0,k0):-1}, {(k0,i0):1, (k0,i1):1})
        add_term({(i0,j0):1, (i0,j1):1}, {(j1,k1):1}, {(k0,i0):-1, (k1,i0):1})
        add_term({(i1,j0):1, (i0,j0):-1}, {(j0,k0):1, (j0,k1):1}, {(k1,i1):1})
        add_term({(i0,j1):1, (i1,j1):-1}, {(j1,k0):1, (j1,k1):1}, {(k0,i0):1})

    # Block ranges
    # A block (i_block, j_block): i in [i0,i1], j in [j0,j1]
    # B block (j_block, k_block): j in [j0,j1], k in [k0,k1]
    # C block (i_block, k_block): i in [i0,i1], k in [k0,k1]

    # C11 = A11*B11 + A12*B21 + A13*B31
    # A11 (2x2) * B11 (2x2): Strassen
    strassen_2x2(0, 1, 0, 1, 0, 1)
    # A12 (2x2) * B21 (2x2): Strassen
    strassen_2x2(0, 1, 2, 3, 0, 1)
    # A13 (2x1) * B31 (1x2): schoolbook 2*1*2 = 4
    for i in [0, 1]:
        for k in [0, 1]:
            add_term({(i,4):1}, {(4,k):1}, {(k,i):1})

    # C12 = A11*B12 + A12*B22 + A13*B32
    # A11 (2x2) * B12 (2x2): Strassen
    strassen_2x2(0, 1, 0, 1, 2, 3)
    # A12 (2x2) * B22 (2x2): Strassen
    strassen_2x2(0, 1, 2, 3, 2, 3)
    # A13 (2x1) * B32 (1x2): schoolbook 2*1*2 = 4
    for i in [0, 1]:
        for k in [2, 3]:
            add_term({(i,4):1}, {(4,k):1}, {(k,i):1})

    # C13 = A11*B13 + A12*B23 + A13*B33
    # A11 (2x2) * B13 (2x1): schoolbook 2*2*1 = 4
    for i in [0, 1]:
        for j in [0, 1]:
            add_term({(i,j):1}, {(j,4):1}, {(4,i):1})
    # A12 (2x2) * B23 (2x1): schoolbook 2*2*1 = 4
    for i in [0, 1]:
        for j in [2, 3]:
            add_term({(i,j):1}, {(j,4):1}, {(4,i):1})
    # A13 (2x1) * B33 (1x1): schoolbook 2*1*1 = 2
    for i in [0, 1]:
        add_term({(i,4):1}, {(4,4):1}, {(4,i):1})

    # C21 = A21*B11 + A22*B21 + A23*B31
    # A21 (2x2) * B11 (2x2): Strassen
    strassen_2x2(2, 3, 0, 1, 0, 1)
    # A22 (2x2) * B21 (2x2): Strassen
    strassen_2x2(2, 3, 2, 3, 0, 1)
    # A23 (2x1) * B31 (1x2): schoolbook 2*1*2 = 4
    for i in [2, 3]:
        for k in [0, 1]:
            add_term({(i,4):1}, {(4,k):1}, {(k,i):1})

    # C22 = A21*B12 + A22*B22 + A23*B32
    # A21 (2x2) * B12 (2x2): Strassen
    strassen_2x2(2, 3, 0, 1, 2, 3)
    # A22 (2x2) * B22 (2x2): Strassen
    strassen_2x2(2, 3, 2, 3, 2, 3)
    # A23 (2x1) * B32 (1x2): schoolbook 2*1*2 = 4
    for i in [2, 3]:
        for k in [2, 3]:
            add_term({(i,4):1}, {(4,k):1}, {(k,i):1})

    # C23 = A21*B13 + A22*B23 + A23*B33
    # A21 (2x2) * B13 (2x1): schoolbook 2*2*1 = 4
    for i in [2, 3]:
        for j in [0, 1]:
            add_term({(i,j):1}, {(j,4):1}, {(4,i):1})
    # A22 (2x2) * B23 (2x1): schoolbook 2*2*1 = 4
    for i in [2, 3]:
        for j in [2, 3]:
            add_term({(i,j):1}, {(j,4):1}, {(4,i):1})
    # A23 (2x1) * B33 (1x1): schoolbook 2*1*1 = 2
    for i in [2, 3]:
        add_term({(i,4):1}, {(4,4):1}, {(4,i):1})

    # C31 = A31*B11 + A32*B21 + A33*B31
    # A31 (1x2) * B11 (2x2): schoolbook 1*2*2 = 4
    for j in [0, 1]:
        for k in [0, 1]:
            add_term({(4,j):1}, {(j,k):1}, {(k,4):1})
    # A32 (1x2) * B21 (2x2): schoolbook 1*2*2 = 4
    for j in [2, 3]:
        for k in [0, 1]:
            add_term({(4,j):1}, {(j,k):1}, {(k,4):1})
    # A33 (1x1) * B31 (1x2): schoolbook 1*1*2 = 2
    for k in [0, 1]:
        add_term({(4,4):1}, {(4,k):1}, {(k,4):1})

    # C32 = A31*B12 + A32*B22 + A33*B32
    # A31 (1x2) * B12 (2x2): schoolbook 1*2*2 = 4
    for j in [0, 1]:
        for k in [2, 3]:
            add_term({(4,j):1}, {(j,k):1}, {(k,4):1})
    # A32 (1x2) * B22 (2x2): schoolbook 1*2*2 = 4
    for j in [2, 3]:
        for k in [2, 3]:
            add_term({(4,j):1}, {(j,k):1}, {(k,4):1})
    # A33 (1x1) * B32 (1x2): schoolbook 1*1*2 = 2
    for k in [2, 3]:
        add_term({(4,4):1}, {(4,k):1}, {(k,4):1})

    # C33 = A31*B13 + A32*B23 + A33*B33
    # A31 (1x2) * B13 (2x1): schoolbook 1*2*1 = 2
    for j in [0, 1]:
        add_term({(4,j):1}, {(j,4):1}, {(4,4):1})
    # A32 (1x2) * B23 (2x1): schoolbook 1*2*1 = 2
    for j in [2, 3]:
        add_term({(4,j):1}, {(j,4):1}, {(4,4):1})
    # A33 (1x1) * B33 (1x1): schoolbook 1*1*1 = 1
    add_term({(4,4):1}, {(4,4):1}, {(4,4):1})

    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T

    rank = U.shape[1]
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
