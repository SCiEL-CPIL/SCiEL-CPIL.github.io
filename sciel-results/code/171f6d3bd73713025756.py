"""
5x5 matrix multiplication - correct rank decomposition.
Uses Strassen for 2x2 blocks (rank 7) and schoolbook for remaining blocks.
Block structure: split 5 as 2+3, then 3 as 2+1.
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
        U = np.zeros((n * m, 1), dtype=float)
        V = np.zeros((m * p, 1), dtype=float)
        W = np.zeros((p * n, 1), dtype=float)
        for (i, j), val in u_vals.items():
            U[i * m + j, 0] = val
        for (j, k), val in v_vals.items():
            V[j * p + k, 0] = val
        for (k, i), val in w_vals.items():
            W[k * n + i, 0] = val
        U_list.append(U.flatten())
        V_list.append(V.flatten())
        W_list.append(W.flatten())

    def schoolbook_block(i_list, j_list, k_list):
        """Add schoolbook terms for A[i_list, j_list] * B[j_list, k_list] -> C[i_list, k_list]"""
        for i in i_list:
            for j in j_list:
                for k in k_list:
                    add_term({(i, j): 1}, {(j, k): 1}, {(k, i): 1})

    def strassen_2x2(i0, i1, j0, j1, k0, k1):
        """
        Strassen decomposition for A[i0:i1, j0:j1] * B[j0:j1, k0:k1] -> C[i0:i1, k0:k1]
        where i0,i1 are the two row indices, j0,j1 the two col indices of A (and row indices of B),
        k0,k1 the two output column indices.
        """
        # M1 = (A00+A11)*(B00+B11)
        add_term({(i0, j0): 1, (i1, j1): 1},
                 {(j0, k0): 1, (j1, k1): 1},
                 {(k0, i0): 1, (k1, i1): 1})
        # M2 = (A10+A11)*B00
        add_term({(i1, j0): 1, (i1, j1): 1},
                 {(j0, k0): 1},
                 {(k0, i1): 1, (k1, i1): -1})
        # M3 = A00*(B01-B11)
        add_term({(i0, j0): 1},
                 {(j0, k1): 1, (j1, k1): -1},
                 {(k1, i0): 1, (k1, i1): 1})
        # M4 = A11*(B10-B00)
        add_term({(i1, j1): 1},
                 {(j1, k0): 1, (j0, k0): -1},
                 {(k0, i0): 1, (k0, i1): 1})
        # M5 = (A00+A01)*B11
        add_term({(i0, j0): 1, (i0, j1): 1},
                 {(j1, k1): 1},
                 {(k0, i0): -1, (k1, i0): 1})
        # M6 = (A10-A00)*(B00+B01)
        add_term({(i1, j0): 1, (i0, j0): -1},
                 {(j0, k0): 1, (j0, k1): 1},
                 {(k1, i1): 1})
        # M7 = (A01-A11)*(B10+B11)
        add_term({(i0, j1): 1, (i1, j1): -1},
                 {(j1, k0): 1, (j1, k1): 1},
                 {(k0, i0): 1})

    # Block structure:
    # Rows of C: {0,1} (top 2) and {2,3,4} (bottom 3)
    # Cols of C: {0,1} (left 2) and {2,3,4} (right 3)
    # A rows: {0,1} and {2,3,4}
    # B cols: {0,1} and {2,3,4}
    # A cols = B rows: {0,1} and {2,3,4}
    #
    # C[0:2, 0:2] = A[0:2, 0:2]*B[0:2, 0:2] + A[0:2, 2:5]*B[2:5, 0:2]
    # C[0:2, 2:5] = A[0:2, 0:2]*B[0:2, 2:5] + A[0:2, 2:5]*B[2:5, 2:5]
    # C[2:5, 0:2] = A[2:5, 0:2]*B[0:2, 0:2] + A[2:5, 2:5]*B[2:5, 0:2]
    # C[2:5, 2:5] = A[2:5, 0:2]*B[0:2, 2:5] + A[2:5, 2:5]*B[2:5, 2:5]

    # --- C[0:2, 0:2] = A[0:2, 0:2]*B[0:2, 0:2] + A[0:2, 2:5]*B[2:5, 0:2] ---
    # A[0:2,0:2] * B[0:2,0:2]: Strassen (7 terms)
    strassen_2x2(0, 1, 0, 1, 0, 1)
    # A[0:2,2:5] * B[2:5,0:2]: A is 2x3, B is 3x2
    # Split j={2,3,4} into {2,3} and {4}
    # A[0:2,2:3] (2x2) * B[2:3,0:2] (2x2): Strassen
    strassen_2x2(0, 1, 2, 3, 0, 1)
    # A[0:2,4] (2x1) * B[4,0:2] (1x2): schoolbook 2*1*2 = 4 terms
    schoolbook_block([0, 1], [4], [0, 1])

    # --- C[0:2, 2:5] = A[0:2, 0:2]*B[0:2, 2:5] + A[0:2, 2:5]*B[2:5, 2:5] ---
    # A[0:2,0:2] (2x2) * B[0:2,2:5] (2x3):
    #   B[0:2,2:3] (2x2) and B[0:2,4] (2x1)
    # A[0:2,0:2] * B[0:2,2:3]: Strassen (7 terms)
    strassen_2x2(0, 1, 0, 1, 2, 3)
    # A[0:2,0:2] * B[0:2,4]: schoolbook 2*2*1 = 4 terms
    schoolbook_block([0, 1], [0, 1], [4])
    # A[0:2,2:5] (2x3) * B[2:5,2:5] (3x3):
    #   A[0:2,2:3] (2x2), A[0:2,4] (2x1)
    #   B[2:3,2:3] (2x2), B[2:3,4] (2x1), B[4,2:3] (1x2), B[4,4] (1x1)
    # A[0:2,2:3] * B[2:3,2:3]: Strassen
    strassen_2x2(0, 1, 2, 3, 2, 3)
    # A[0:2,2:3] * B[2:3,4]: schoolbook 2*2*1 = 4
    schoolbook_block([0, 1], [2, 3], [4])
    # A[0:2,4] * B[4,2:3]: schoolbook 2*1*2 = 4
    schoolbook_block([0, 1], [4], [2, 3])
    # A[0:2,4] * B[4,4]: schoolbook 2*1*1 = 2
    schoolbook_block([0, 1], [4], [4])

    # --- C[2:5, 0:2] = A[2:5, 0:2]*B[0:2, 0:2] + A[2:5, 2:5]*B[2:5, 0:2] ---
    # A[2:5,0:2] (3x2) * B[0:2,0:2] (2x2):
    #   A[2:3,0:2] (2x2), A[4,0:2] (1x2)
    # A[2:3,0:2] * B[0:2,0:2]: Strassen
    strassen_2x2(2, 3, 0, 1, 0, 1)
    # A[4,0:2] * B[0:2,0:2]: schoolbook 1*2*2 = 4
    schoolbook_block([4], [0, 1], [0, 1])
    # A[2:5,2:5] (3x3) * B[2:5,0:2] (3x2):
    #   A[2:3,2:3] (2x2), A[2:3,4] (2x1), A[4,2:3] (1x2), A[4,4] (1x1)
    #   B[2:3,0:2] (2x2), B[4,0:2] (1x2)
    # A[2:3,2:3] * B[2:3,0:2]: Strassen
    strassen_2x2(2, 3, 2, 3, 0, 1)
    # A[2:3,4] * B[4,0:2]: schoolbook 2*1*2 = 4
    schoolbook_block([2, 3], [4], [0, 1])
    # A[4,2:3] * B[2:3,0:2]: schoolbook 1*2*2 = 4
    schoolbook_block([4], [2, 3], [0, 1])
    # A[4,4] * B[4,0:2]: schoolbook 1*1*2 = 2
    schoolbook_block([4], [4], [0, 1])

    # --- C[2:5, 2:5] = A[2:5, 0:2]*B[0:2, 2:5] + A[2:5, 2:5]*B[2:5, 2:5] ---
    # A[2:5,0:2] (3x2) * B[0:2,2:5] (2x3):
    #   A[2:3,0:2] (2x2), A[4,0:2] (1x2)
    #   B[0:2,2:3] (2x2), B[0:2,4] (2x1)
    # A[2:3,0:2] * B[0:2,2:3]: Strassen
    strassen_2x2(2, 3, 0, 1, 2, 3)
    # A[2:3,0:2] * B[0:2,4]: schoolbook 2*2*1 = 4
    schoolbook_block([2, 3], [0, 1], [4])
    # A[4,0:2] * B[0:2,2:3]: schoolbook 1*2*2 = 4
    schoolbook_block([4], [0, 1], [2, 3])
    # A[4,0:2] * B[0:2,4]: schoolbook 1*2*1 = 2
    schoolbook_block([4], [0, 1], [4])
    # A[2:5,2:5] (3x3) * B[2:5,2:5] (3x3):
    #   A[2:3,2:3] (2x2), A[2:3,4] (2x1), A[4,2:3] (1x2), A[4,4] (1x1)
    #   B[2:3,2:3] (2x2), B[2:3,4] (2x1), B[4,2:3] (1x2), B[4,4] (1x1)
    # A[2:3,2:3] * B[2:3,2:3]: Strassen
    strassen_2x2(2, 3, 2, 3, 2, 3)
    # A[2:3,2:3] * B[2:3,4]: schoolbook 2*2*1 = 4
    schoolbook_block([2, 3], [2, 3], [4])
    # A[2:3,4] * B[4,2:3]: schoolbook 2*1*2 = 4
    schoolbook_block([2, 3], [4], [2, 3])
    # A[2:3,4] * B[4,4]: schoolbook 2*1*1 = 2
    schoolbook_block([2, 3], [4], [4])
    # A[4,2:3] * B[2:3,2:3]: schoolbook 1*2*2 = 4
    schoolbook_block([4], [2, 3], [2, 3])
    # A[4,2:3] * B[2:3,4]: schoolbook 1*2*1 = 2
    schoolbook_block([4], [2, 3], [4])
    # A[4,4] * B[4,2:3]: schoolbook 1*1*2 = 2
    schoolbook_block([4], [4], [2, 3])
    # A[4,4] * B[4,4]: schoolbook 1*1*1 = 1
    schoolbook_block([4], [4], [4])

    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T

    rank = U.shape[1]

    if time.time() - t0 > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
