import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = 140.0

    U_list = []
    V_list = []
    W_list = []

    def add_term(u_vals, v_vals, w_vals):
        U = np.zeros(n * m, dtype=float)
        V = np.zeros(m * p, dtype=float)
        W = np.zeros(p * n, dtype=float)
        for (i, j), val in u_vals.items():
            U[i * m + j] = val
        for (j, k), val in v_vals.items():
            V[j * p + k] = val
        for (k, i), val in w_vals.items():
            W[k * n + i] = val
        U_list.append(U)
        V_list.append(V)
        W_list.append(W)

    def schoolbook_block(i_start, i_end, j_start, j_end, k_start, k_end):
        """
        Adds terms for C[i,k] += A[i,j] * B[j,k]
        for i in [i_start, i_end), j in [j_start, j_end), k in [k_start, k_end)
        """
        for i in range(i_start, i_end):
            for k in range(k_start, k_end):
                for j in range(j_start, j_end):
                    add_term({(i, j): 1}, {(j, k): 1}, {(k, i): 1})

    def strassen_2x2_block(i0, i1, j0, j1, k0, k1):
        """
        Adds terms for 2x2 Strassen multiplication.
        A indices: (i0,i1) x (j0,j1)
        B indices: (j0,j1) x (k0,k1)
        C indices: (i0,i1) x (k0,k1)
        """
        # M1 = (A00+A11)(B00+B11)
        add_term({(i0, j0): 1, (i1, j1): 1},
                 {(j0, k0): 1, (j1, k1): 1},
                 {(k0, i0): 1, (k1, i1): 1})
        # M2 = (A10+A11)B00
        add_term({(i1, j0): 1, (i1, j1): 1},
                 {(j0, k0): 1},
                 {(k0, i1): 1, (k1, i1): -1})
        # M3 = A00(B01-B11)
        add_term({(i0, j0): 1},
                 {(j0, k1): 1, (j1, k1): -1},
                 {(k1, i0): 1, (k1, i1): 1})
        # M4 = A11(B10-B00)
        add_term({(i1, j1): 1},
                 {(j1, k0): 1, (j0, k0): -1},
                 {(k0, i0): 1, (k0, i1): 1})
        # M5 = (A00+A01)B11
        add_term({(i0, j0): 1, (i0, j1): 1},
                 {(j1, k1): 1},
                 {(k0, i0): -1, (k1, i0): 1})
        # M6 = (A10-A00)(B00+B01)
        add_term({(i1, j0): 1, (i0, j0): -1},
                 {(j0, k0): 1, (j0, k1): 1},
                 {(k1, i1): 1})
        # M7 = (A01-A11)(B10+B11)
        add_term({(i0, j1): 1, (i1, j1): -1},
                 {(j1, k0): 1, (j1, k1): 1},
                 {(k0, i0): 1})

    # Split 5x5 into 2+3 blocks
    # Rows/Cols 0,1 are "small" (size 2)
    # Rows/Cols 2,3,4 are "large" (size 3)
    
    # Block (0,0): A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # Dimensions: 2x2 * 2x2 -> 2x2
    # Use Strassen
    strassen_2x2_block(0, 1, 0, 1, 0, 1)
    
    # Block (0,1): A[0:2, 2:5] * B[2:5, 0:2] -> C[0:2, 0:2]
    # Dimensions: 2x3 * 3x2 -> 2x2
    # Use Schoolbook
    schoolbook_block(0, 2, 2, 5, 0, 2)
    
    # Block (0,2): A[0:2, 0:2] * B[0:2, 2:5] -> C[0:2, 2:5]
    # Dimensions: 2x2 * 2x3 -> 2x3
    # Use Schoolbook
    schoolbook_block(0, 2, 0, 2, 2, 5)
    
    # Block (0,3): A[0:2, 2:5] * B[2:5, 2:5] -> C[0:2, 2:5]
    # Dimensions: 2x3 * 3x3 -> 2x3
    # Use Schoolbook
    schoolbook_block(0, 2, 2, 5, 2, 5)
    
    # Block (1,0): A[2:5, 0:2] * B[0:2, 0:2] -> C[2:5, 0:2]
    # Dimensions: 3x2 * 2x2 -> 3x2
    # Use Schoolbook
    schoolbook_block(2, 5, 0, 2, 0, 2)
    
    # Block (1,1): A[2:5, 2:5] * B[2:5, 0:2] -> C[2:5, 0:2]
    # Dimensions: 3x3 * 3x2 -> 3x2
    # Use Schoolbook
    schoolbook_block(2, 5, 2, 5, 0, 2)
    
    # Block (1,2): A[2:5, 0:2] * B[0:2, 2:5] -> C[2:5, 2:5]
    # Dimensions: 3x2 * 2x3 -> 3x3
    # Use Schoolbook
    schoolbook_block(2, 5, 0, 2, 2, 5)
    
    # Block (1,3): A[2:5, 2:5] * B[2:5, 2:5] -> C[2:5, 2:5]
    # Dimensions: 3x3 * 3x3 -> 3x3
    # Use Schoolbook
    schoolbook_block(2, 5, 2, 5, 2, 5)

    if time.time() - t0 > deadline:
        # Emergency return, though construction is fast
        U = np.array(U_list).T
        V = np.array(V_list).T
        W = np.array(W_list).T
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
