"""
5x5 matrix multiplication - rank 112 block decomposition (2+3 split).
Uses Strassen for 2x2 block and schoolbook for 3x3 and cross terms.
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

    # Strassen for 2x2 matrix multiplication
    def strassen_2x2(i0, i1, j0, j1, k0, k1):
        """Strassen for block with row idx i0,i1; col j0,j1; B col k0,k1"""
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

    # Schoolbook multiplication for rectangular blocks
    def schoolbook_rect(i_start, i_end, j_start, j_end, k_start, k_end):
        """
        Schoolbook for A[i_start:i_end, j_start:j_end] * B[j_start:j_end, k_start:k_end]
        i range: i_start to i_end-1
        j range: j_start to j_end-1
        k range: k_start to k_end-1
        """
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                for k in range(k_start, k_end):
                    add_term({(i,j):1}, {(j,k):1}, {(k,i):1})

    # Block indices:
    # I1 = [0, 1] (2x2)
    # I2 = [2, 3, 4] (3x3)

    # 1. C_11 = A_11 B_11 (2x2 * 2x2)
    # Use Strassen
    strassen_2x2(0, 2, 0, 2, 0, 2) # i in 0,1; j in 0,1; k in 0,1

    # 2. C_22 = A_22 B_22 (3x3 * 3x3)
    # Use Schoolbook
    schoolbook_rect(2, 5, 2, 5, 2, 5)

    # 3. C_12 = A_11 B_12 + A_12 B_22
    # A_11 (2x2) * B_12 (2x3)
    # i in 0,1; j in 0,1; k in 2,3,4
    schoolbook_rect(0, 2, 0, 2, 2, 5)
    
    # A_12 (2x3) * B_22 (3x3)
    # i in 0,1; j in 2,3,4; k in 2,3,4
    schoolbook_rect(0, 2, 2, 5, 2, 5)

    # 4. C_21 = A_21 B_11 + A_22 B_21
    # A_21 (3x2) * B_11 (2x2)
    # i in 2,3,4; j in 0,1; k in 0,1
    schoolbook_rect(2, 5, 0, 2, 0, 2)

    # A_22 (3x3) * B_21 (3x2)
    # i in 2,3,4; j in 2,3,4; k in 0,1
    schoolbook_rect(2, 5, 2, 5, 0, 2)

    # 5. C_22 cross term: A_21 B_12
    # A_21 (3x2) * B_12 (2x3)
    # i in 2,3,4; j in 0,1; k in 2,3,4
    schoolbook_rect(2, 5, 0, 2, 2, 5)

    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T

    rank = U.shape[1]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
