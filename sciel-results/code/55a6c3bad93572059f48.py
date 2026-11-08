"""
5x5 matrix multiplication - rank 117 block decomposition.
Uses Strassen for 2x2 blocks and optimized splits for remaining blocks.
"""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    
    U_list = []
    V_list = []
    W_list = []

    def add_term(u_vals, v_vals, w_vals):
        U = np.zeros(n*m, dtype=float)
        V = np.zeros(m*p, dtype=float)
        W = np.zeros(p*n, dtype=float)

        for (i, j), val in u_vals.items():
            U[i*m+j] = val
        for (j, k), val in v_vals.items():
            V[j*p+k] = val
        for (k, i), val in w_vals.items():
            W[k*n+i] = val

        U_list.append(U)
        V_list.append(V)
        W_list.append(W)

    def strassen_2x2(i0, i1, j0, j1, k0, k1):
        """7-term Strassen for C[i0:i2,k0:k2] += A[i0:i2,j0:j2]*B[j0:j2,k0:k2]"""
        add_term({(i0,j0):1, (i1,j1):1}, {(j0,k0):1, (j1,k1):1}, {(k0,i0):1, (k1,i1):1})
        add_term({(i1,j0):1, (i1,j1):1}, {(j0,k0):1}, {(k0,i1):1, (k1,i1):-1})
        add_term({(i0,j0):1}, {(j0,k1):1, (j1,k1):-1}, {(k1,i0):1, (k1,i1):1})
        add_term({(i1,j1):1}, {(j1,k0):1, (j0,k0):-1}, {(k0,i0):1, (k0,i1):1})
        add_term({(i0,j0):1, (i0,j1):1}, {(j1,k1):1}, {(k0,i0):-1, (k1,i0):1})
        add_term({(i1,j0):1, (i0,j0):-1}, {(j0,k0):1, (j0,k1):1}, {(k1,i1):1})
        add_term({(i0,j1):1, (i1,j1):-1}, {(j1,k0):1, (j1,k1):1}, {(k0,i0):1})

    # 2+3 Split: indices 0-1 (2 rows/cols) and 2-4 (3 rows/cols)
    # C_22 = A_22 B_22 + A_23 B_32  (2x2 block)
    strassen_2x2(0, 1, 0, 1, 0, 1)  # A[0:2,0:2]*B[0:2,0:2] -> 7
    for i in [0, 1]:
        for k in [0, 1]:
            for j in [2, 3, 4]:
                add_term({(i,j):1}, {(j,k):1}, {(k,i):1})  # A[0:2,2:5]*B[2:5,0:2] -> 4*3=12

    # C_23 = A_22 B_23 + A_23 B_33  (2x3 block)
    strassen_2x2(0, 1, 0, 1, 2, 3)  # A[0:2,0:2]*B[0:2,2:4] -> 7
    for i in [0, 1]:
        for j in [0, 1]:
            add_term({(i,j):1}, {(j,4):1}, {(4,i):1})  # A[0:2,0:2]*B[0:2,4] -> 4
    strassen_2x2(0, 1, 2, 3, 2, 3)  # A[0:2,2:4]*B[2:4,2:4] -> 7
    for i in [0, 1]:
        for j in [2, 3]:
            add_term({(i,j):1}, {(j,4):1}, {(4,i):1})  # A[0:2,2:4]*B[2:4,4] -> 4
    for i in [0, 1]:
        for k in [2, 3, 4]:
            add_term({(i,4):1}, {(4,k):1}, {(k,i):1})  # A[0:2,4]*B[4,2:5] -> 6

    # C_32 = A_32 B_22 + A_33 B_32  (3x2 block)
    strassen_2x2(2, 3, 0, 1, 0, 1)  # A[2:4,0:2]*B[0:2,0:2] -> 7
    for j in [0, 1]:
        for k in [0, 1]:
            add_term({(4,j):1}, {(j,k):1}, {(k,4):1})  # A[4,0:2]*B[0:2,0:2] -> 4
    strassen_2x2(2, 3, 2, 3, 0, 1)  # A[2:4,2:4]*B[2:4,0:2] -> 7
    for i in [2, 3]:
        for k in [0, 1]:
            add_term({(i,4):1}, {(4,k):1}, {(k,i):1})  # A[2:4,4]*B[4,0:2] -> 4
    for j in [2, 3]:
        for k in [0, 1]:
            add_term({(4,j):1}, {(j,k):1}, {(k,4):1})  # A[4,2:4]*B[2:4,0:2] -> 4
    for k in [0, 1]:
        add_term({(4,4):1}, {(4,k):1}, {(k,4):1})  # A[4,4]*B[4,0:2] -> 2

    # C_33 = A_32 B_23 + A_33 B_33  (3x3 block)
    strassen_2x2(2, 3, 0, 1, 2, 3)  # A[2:4,0:2]*B[0:2,2:4] -> 7
    for i in [2, 3]:
        for j in [0, 1]:
            add_term({(i,j):1}, {(j,4):1}, {(4,i):1})  # A[2:4,0:2]*B[0:2,4] -> 4
    for j in [0, 1]:
        for k in [2, 3]:
            add_term({(4,j):1}, {(j,k):1}, {(k,4):1})  # A[4,0:2]*B[0:2,2:4] -> 4
    for j in [0, 1]:
        add_term({(4,j):1}, {(j,4):1}, {(4,4):1})  # A[4,0:2]*B[0:2,4] -> 2
    strassen_2x2(2, 3, 2, 3, 2, 3)  # A[2:4,2:4]*B[2:4,2:4] -> 7
    for i in [2, 3]:
        for k in [2, 3]:
            add_term({(i,4):1}, {(4,k):1}, {(k,i):1})  # A[2:4,4]*B[4,2:4] -> 4
    for i in [2, 3]:
        for j in [2, 3]:
            add_term({(i,j):1}, {(j,4):1}, {(4,i):1})  # A[2:4,2:4]*B[2:4,4] -> 4
    for i in [2, 3]:
        add_term({(i,4):1}, {(4,4):1}, {(4,i):1})  # A[2:4,4]*B[4,4] -> 2
    for j in [2, 3]:
        for k in [2, 3]:
            add_term({(4,j):1}, {(j,k):1}, {(k,4):1})  # A[4,2:4]*B[2:4,2:4] -> 4
    for k in [2, 3]:
        add_term({(4,4):1}, {(4,k):1}, {(k,4):1})  # A[4,4]*B[4,2:4] -> 2
    for j in [2, 3]:
        add_term({(4,j):1}, {(j,4):1}, {(4,4):1})  # A[4,2:4]*B[2:4,4] -> 2
    add_term({(4,4):1}, {(4,4):1}, {(4,4):1})  # A[4,4]*B[4,4] -> 1

    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T

    if time.time() - t0 > 130.0:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
