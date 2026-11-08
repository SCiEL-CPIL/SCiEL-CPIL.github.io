"""
5x5 matrix multiplication - optimized rank decomposition (2+3 split).
Uses Strassen for 2x2 blocks and carefully optimized schoolbook/combined
decompositions for rectangular cross-blocks to minimize total rank.
Refined: ensures exact decomposition by correctly handling all sub-block
interactions in the 3x3 blocks via 2+1 splits.
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
        add_term({(i0,j0):1, (i1,j1):1}, {(j0,k0):1, (j1,k1):1}, {(k0,i0):1, (k1,i1):1})
        add_term({(i1,j0):1, (i1,j1):1}, {(j0,k0):1}, {(k0,i1):1, (k1,i1):-1})
        add_term({(i0,j0):1}, {(j0,k1):1, (j1,k1):-1}, {(k1,i0):1, (k1,i1):1})
        add_term({(i1,j1):1}, {(j1,k0):1, (j0,k0):-1}, {(k0,i0):1, (k0,i1):1})
        add_term({(i0,j0):1, (i0,j1):1}, {(j1,k1):1}, {(k0,i0):-1, (k1,i0):1})
        add_term({(i1,j0):1, (i0,j0):-1}, {(j0,k0):1, (j0,k1):1}, {(k1,i1):1})
        add_term({(i0,j1):1, (i1,j1):-1}, {(j1,k0):1, (j1,k1):1}, {(k0,i0):1})

    def schoolbook_2x3_3x2(i0, i1, j0, j1, j2, k0, k1):
        """
        Computes C[i0:i1+1, k0:k1+1] += A[i0:i1+1, j0:j2+1] * B[j0:j2+1, k0:k1+1]
        where A is 2x3 and B is 3x2.
        Uses 2+1 split on the middle dimension j.
        A_2x2 (j0,j1) * B_2x2 (j0,j1) -> Strassen
        A_2x1 (j2) * B_1x2 (j2) -> Schoolbook
        A_2x2 (j0,j1) * B_1x2 (j2) -> NO overlap in j, so zero.
        A_2x1 (j2) * B_2x2 (j0,j1) -> NO overlap in j, so zero.
        """
        strassen_2x2(i0, i1, j0, j1, k0, k1)
        for i in [i0, i1]:
            for k in [k0, k1]:
                add_term({(i, j2): 1}, {(j2, k): 1}, {(k, i): 1})

    def schoolbook_2x2_2x3(i0, i1, j0, j1, k0, k1, k2):
        """
        Computes C[i0:i1+1, k0:k2+1] += A[i0:i1+1, j0:j1+1] * B[j0:j1+1, k0:k2+1]
        where A is 2x2 and B is 2x3.
        Uses 2+1 split on the output dimension k.
        A_2x2 * B_2x2 (k0,k1) -> Strassen
        A_2x2 * B_2x1 (k2) -> Schoolbook
        """
        strassen_2x2(i0, i1, j0, j1, k0, k1)
        for i in [i0, i1]:
            for j in [j0, j1]:
                add_term({(i, j): 1}, {(j, k2): 1}, {(k2, i): 1})

    def schoolbook_3x2_2x2(i0, i1, i2, j0, j1, k0, k1):
        """
        Computes C[i0:i2+1, k0:k1+1] += A[i0:i2+1, j0:j1+1] * B[j0:j1+1, k0:k1+1]
        where A is 3x2 and B is 2x2.
        Uses 2+1 split on the input dimension i.
        A_2x2 (i0,i1) * B_2x2 -> Strassen
        A_1x2 (i2) * B_2x2 -> Schoolbook
        """
        strassen_2x2(i0, i1, j0, j1, k0, k1)
        for j in [j0, j1]:
            for k in [k0, k1]:
                add_term({(i2, j): 1}, {(j, k): 1}, {(k, i2): 1})

    def schoolbook_3x2_2x3(i0, i1, i2, j0, j1, k0, k1, k2):
        """
        Computes C[i0:i2+1, k0:k2+1] += A[i0:i2+1, j0:j1+1] * B[j0:j1+1, k0:k2+1]
        where A is 3x2 and B is 2x3.
        Uses 2+1 split on both i and k.
        A_2x2 (i0,i1) * B_2x2 (k0,k1) -> Strassen
        A_2x2 (i0,i1) * B_2x1 (k2) -> Schoolbook
        A_1x2 (i2) * B_2x2 (k0,k1) -> Schoolbook
        A_1x2 (i2) * B_2x1 (k2) -> Schoolbook
        """
        strassen_2x2(i0, i1, j0, j1, k0, k1)
        for i in [i0, i1]:
            for j in [j0, j1]:
                add_term({(i, j): 1}, {(j, k2): 1}, {(k2, i): 1})
        for j in [j0, j1]:
            for k in [k0, k1]:
                add_term({(i2, j): 1}, {(j, k): 1}, {(k, i2): 1})
        for j in [j0, j1]:
            add_term({(i2, j): 1}, {(j, k2): 1}, {(k2, i2): 1})

    def schoolbook_3x3_3x3(i0, i1, i2, j0, j1, j2, k0, k1, k2):
        """
        Computes C[i0:i2+1, k0:k2+1] += A[i0:i2+1, j0:j2+1] * B[j0:j2+1, k0:k2+1]
        where A and B are 3x3.
        Uses 2+1 split on i, j, k.
        A_2x2 (i0,i1; j0,j1) * B_2x2 (j0,j1; k0,k1) -> Strassen
        A_2x1 (i0,i1; j2) * B_1x2 (j2; k0,k1) -> Schoolbook
        A_2x1 (i0,i1; j2) * B_1x1 (j2; k2) -> Schoolbook
        A_1x2 (i2; j0,j1) * B_2x2 (j0,j1; k0,k1) -> Schoolbook
        A_1x2 (i2; j0,j1) * B_2x1 (j0,j1; k2) -> Schoolbook
        A_1x1 (i2; j2) * B_1x2 (j2; k0,k1) -> Schoolbook
        A_1x1 (i2; j2) * B_1x1 (j2; k2) -> Schoolbook
        """
        strassen_2x2(i0, i1, j0, j1, k0, k1)
        for i in [i0, i1]:
            for k in [k0, k1]:
                add_term({(i, j2): 1}, {(j2, k): 1}, {(k, i): 1})
        for i in [i0, i1]:
            add_term({(i, j2): 1}, {(j2, k2): 1}, {(k2, i): 1})
        for j in [j0, j1]:
            for k in [k0, k1]:
                add_term({(i2, j): 1}, {(j, k): 1}, {(k, i2): 1})
        for j in [j0, j1]:
            add_term({(i2, j): 1}, {(j, k2): 1}, {(k2, i2): 1})
        for k in [k0, k1]:
            add_term({(i2, j2): 1}, {(j2, k): 1}, {(k, i2): 1})
        add_term({(i2, j2): 1}, {(j2, k2): 1}, {(k2, i2): 1})

    # Block indices: rows/cols 0,1 are "2" block; rows/cols 2,3,4 are "3" block
    # Split "3" into {2,3} (2x2) and {4} (1x1) for Strassen applicability

    # C_22 (2x2) = A_22(2x2) * B_22(2x2) + A_23(2x3) * B_32(3x2)
    strassen_2x2(0, 1, 0, 1, 0, 1)
    schoolbook_2x3_3x2(0, 1, 2, 3, 4, 0, 1)
    # C_22 subtotal: 7 + 11 = 18

    # C_23 (2x3) = A_22(2x2) * B_23(2x3) + A_23(2x3) * B_33(3x3)
    schoolbook_2x2_2x3(0, 1, 0, 1, 2, 3, 4)
    # A_23 (2x3) * B_33 (3x3):
    # A_2x2 (j=2,3) * B_2x2 (j=2,3; k=2,3) -> Strassen
    strassen_2x2(0, 1, 2, 3, 2, 3)
    # A_2x2 (j=2,3) * B_2x1 (j=2,3; k=4) -> Schoolbook
    for i in [0, 1]:
        for j in [2, 3]:
            add_term({(i, j): 1}, {(j, 4): 1}, {(4, i): 1})
    # A_2x1 (j=4) * B_1x2 (j=4; k=2,3) -> Schoolbook
    for i in [0, 1]:
        for k in [2, 3]:
            add_term({(i, 4): 1}, {(4, k): 1}, {(k, i): 1})
    # A_2x1 (j=4) * B_1x1 (j=4; k=4) -> Schoolbook
    for i in [0, 1]:
        add_term({(i, 4): 1}, {(4, 4): 1}, {(4, i): 1})
    # C_23 subtotal: 11 + 7 + 4 + 4 + 2 = 28

    # C_32 (3x2) = A_32(3x2) * B_22(2x2) + A_33(3x3) * B_32(3x2)
    schoolbook_3x2_2x2(2, 3, 4, 0, 1, 0, 1)
    # A_33 (3x3) * B_32 (3x2):
    # A_2x2 (i=2,3; j=2,3) * B_2x2 (j=2,3; k=0,1) -> Strassen
    strassen_2x2(2, 3, 2, 3, 0, 1)
    # A_2x1 (i=2,3; j=4) * B_1x2 (j=4; k=0,1) -> Schoolbook
    for i in [2, 3]:
        for k in [0, 1]:
            add_term({(i, 4): 1}, {(4, k): 1}, {(k, i): 1})
    # A_1x2 (i=4; j=2,3) * B_2x2 (j=2,3; k=0,1) -> Schoolbook
    for j in [2, 3]:
        for k in [0, 1]:
            add_term({(4, j): 1}, {(j, k): 1}, {(k, 4): 1})
    # A_1x1 (i=4; j=4) * B_1x2 (j=4; k=0,1) -> Schoolbook
    for k in [0, 1]:
        add_term({(4, 4): 1}, {(4, k): 1}, {(k, 4): 1})
    # C_32 subtotal: 11 + 7 + 4 + 4 + 2 = 28

    # C_33 (3x3) = A_32(3x2) * B_23(2x3) + A_33(3x3) * B_33(3x3)
    schoolbook_3x2_2x3(2, 3, 4, 0, 1, 2, 3, 4)
    schoolbook_3x3_3x3(2, 3, 4, 2, 3, 4, 2, 3, 4)
    # C_33 subtotal: 15 + 28 = 43

    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T

    rank = U.shape[1]

    if time.time() - t0 > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
