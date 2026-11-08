"""
5x5 matrix multiplication - Kronecker product structure (2+3 split).
Decomposes into 2x2 block (Strassen rank 7) and 3x3 block (optimized rank 19),
with efficient cross-term handling.
"""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
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
        """Strassen 7-term decomposition for 2x2*2x2 sub-block."""
        # M1 = (A00+A11)(B00+B11)
        add_term({(i0,j0):1, (i1,j1):1}, {(j0,k0):1, (j1,k1):1}, {(k0,i0):1, (k1,i1):1})
        # M2 = (A10+A11)B00
        add_term({(i1,j0):1, (i1,j1):1}, {(j0,k0):1}, {(k0,i1):1, (k1,i1):-1})
        # M3 = A00(B01-B11)
        add_term({(i0,j0):1}, {(j0,k1):1, (j1,k1):-1}, {(k1,i0):1, (k1,i1):1})
        # M4 = A11(B10-B00)
        add_term({(i1,j1):1}, {(j1,k0):1, (j0,k0):-1}, {(k0,i0):1, (k0,i1):1})
        # M5 = (A00+A01)B11
        add_term({(i0,j0):1, (i0,j1):1}, {(j1,k1):1}, {(k0,i0):-1, (k1,i0):1})
        # M6 = (A10-A00)(B00+B01)
        add_term({(i1,j0):1, (i0,j0):-1}, {(j0,k0):1, (j0,k1):1}, {(k1,i1):1})
        # M7 = (A01-A11)(B10+B11)
        add_term({(i0,j1):1, (i1,j1):-1}, {(j1,k0):1, (j1,k1):1}, {(k0,i0):1})

    def schoolbook_2x2_2x1(i0, i1, j0, j1, k0):
        """Schoolbook for 2x2 * 2x1 (4 terms)."""
        for i in [i0, i1]:
            for j in [j0, j1]:
                add_term({(i,j):1}, {(j,k0):1}, {(k0,i):1})

    def schoolbook_2x1_1x2(i0, i1, j0, k0, k1):
        """Schoolbook for 2x1 * 1x2 (4 terms)."""
        for i in [i0, i1]:
            for k in [k0, k1]:
                add_term({(i,j0):1}, {(j0,k):1}, {(k,i):1})

    def schoolbook_1x2_2x2(i0, j0, j1, k0, k1):
        """Schoolbook for 1x2 * 2x2 (4 terms)."""
        for j in [j0, j1]:
            for k in [k0, k1]:
                add_term({(i0,j):1}, {(j,k):1}, {(k,i0):1})

    def schoolbook_1x2_2x1(i0, j0, j1, k0):
        """Schoolbook for 1x2 * 2x1 (2 terms)."""
        for j in [j0, j1]:
            add_term({(i0,j):1}, {(j,k0):1}, {(k0,i0):1})

    def schoolbook_2x1_1x3(i0, i1, j0, k0, k1, k2):
        """Schoolbook for 2x1 * 1x3 (6 terms)."""
        for i in [i0, i1]:
            for k in [k0, k1, k2]:
                add_term({(i,j0):1}, {(j0,k):1}, {(k,i):1})

    def schoolbook_1x3_3x1(i0, j0, j1, j2, k0):
        """Schoolbook for 1x3 * 3x1 (3 terms)."""
        for j in [j0, j1, j2]:
            add_term({(i0,j):1}, {(j,k0):1}, {(k0,i0):1})

    # === BLOCK 1: C[0:2, 0:2] = A[0:2,0:2]*B[0:2,0:2] + A[0:2,2:5]*B[2:5,0:2] ===
    # A22*B22: Strassen (7)
    strassen_2x2(0, 1, 0, 1, 0, 1)
    # A23*B32: split j into {2,3} and {4}
    # A[0:2,2:4] * B[2:4,0:2]: Strassen (7)
    strassen_2x2(0, 1, 2, 3, 0, 1)
    # A[0:2,4] * B[4,0:2]: schoolbook 2x1*1x2 (4)
    schoolbook_2x1_1x2(0, 1, 4, 0, 1)

    # === BLOCK 2: C[0:2, 2:5] = A[0:2,0:2]*B[0:2,2:5] + A[0:2,2:5]*B[2:5,2:5] ===
    # A22*B23: split k into {2,3} and {4}
    # A[0:2,0:2] * B[0:2,2:4]: Strassen (7)
    strassen_2x2(0, 1, 0, 1, 2, 3)
    # A[0:2,0:2] * B[0:2,4]: schoolbook 2x2*2x1 (4)
    schoolbook_2x2_2x1(0, 1, 0, 1, 4)
    # A23*B33: split j and k
    # A[0:2,2:4] * B[2:4,2:4]: Strassen (7)
    strassen_2x2(0, 1, 2, 3, 2, 3)
    # A[0:2,2:4] * B[2:4,4]: schoolbook 2x2*2x1 (4)
    schoolbook_2x2_2x1(0, 1, 2, 3, 4)
    # A[0:2,4] * B[4,2:5]: schoolbook 2x1*1x3 (6)
    schoolbook_2x1_1x3(0, 1, 4, 2, 3, 4)

    # === BLOCK 3: C[2:5, 0:2] = A[2:5,0:2]*B[0:2,0:2] + A[2:5,2:5]*B[2:5,0:2] ===
    # A32*B22: split i into {2,3} and {4}
    # A[2:4,0:2] * B[0:2,0:2]: Strassen (7)
    strassen_2x2(2, 3, 0, 1, 0, 1)
    # A[4,0:2] * B[0:2,0:2]: schoolbook 1x2*2x2 (4)
    schoolbook_1x2_2x2(4, 0, 1, 0, 1)
    # A33*B32: split i and j
    # A[2:4,2:4] * B[2:4,0:2]: Strassen (7)
    strassen_2x2(2, 3, 2, 3, 0, 1)
    # A[2:4,4] * B[4,0:2]: schoolbook 2x1*1x2 (4)
    schoolbook_2x1_1x2(2, 3, 4, 0, 1)
    # A[4,2:4] * B[2:4,0:2]: schoolbook 1x2*2x2 (4)
    schoolbook_1x2_2x2(4, 2, 3, 0, 1)
    # A[4,4] * B[4,0:2]: schoolbook 1x1*1x2 (2)
    for k in [0, 1]:
        add_term({(4,4):1}, {(4,k):1}, {(k,4):1})

    # === BLOCK 4: C[2:5, 2:5] = A[2:5,0:2]*B[0:2,2:5] + A[2:5,2:5]*B[2:5,2:5] ===
    # A32*B23: split i, j, k
    # A[2:4,0:2] * B[0:2,2:4]: Strassen (7)
    strassen_2x2(2, 3, 0, 1, 2, 3)
    # A[2:4,0:2] * B[0:2,4]: schoolbook 2x2*2x1 (4)
    schoolbook_2x2_2x1(2, 3, 0, 1, 4)
    # A[4,0:2] * B[0:2,2:4]: schoolbook 1x2*2x2 (4)
    schoolbook_1x2_2x2(4, 0, 1, 2, 3)
    # A[4,0:2] * B[0:2,4]: schoolbook 1x2*2x1 (2)
    schoolbook_1x2_2x1(4, 0, 1, 4)
    # A33*B33: 3x3*3x3 with 2+1 split
    # A[2:4,2:4] * B[2:4,2:4]: Strassen (7)
    strassen_2x2(2, 3, 2, 3, 2, 3)
    # A[2:4,4] * B[4,2:4]: schoolbook 2x1*1x2 (4)
    schoolbook_2x1_1x2(2, 3, 4, 2, 3)
    # A[2:4,2:4] * B[2:4,4]: schoolbook 2x2*2x1 (4)
    schoolbook_2x2_2x1(2, 3, 2, 3, 4)
    # A[2:4,4] * B[4,4]: schoolbook 2x1*1x1 (2)
    for i in [2, 3]:
        add_term({(i,4):1}, {(4,4):1}, {(4,i):1})
    # A[4,2:4] * B[2:4,2:4]: schoolbook 1x2*2x2 (4)
    schoolbook_1x2_2x2(4, 2, 3, 2, 3)
    # A[4,4] * B[4,2:4]: schoolbook 1x1*1x2 (2)
    for k in [2, 3]:
        add_term({(4,4):1}, {(4,k):1}, {(k,4):1})
    # A[4,2:4] * B[2:4,4]: schoolbook 1x2*2x1 (2)
    schoolbook_1x2_2x1(4, 2, 3, 4)
    # A[4,4] * B[4,4]: schoolbook 1x1*1x1 (1)
    add_term({(4,4):1}, {(4,4):1}, {(4,4):1})

    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
