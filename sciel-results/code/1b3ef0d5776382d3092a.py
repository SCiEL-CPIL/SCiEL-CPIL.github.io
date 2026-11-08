"""
5x5 matrix multiplication - optimized rank decomposition (2+3 split).
Uses Strassen for 2x2 blocks and carefully optimized schoolbook/combined
decompositions for rectangular cross-blocks to minimize total rank.
"""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = 130.0  # Safety margin before 150s kill

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
        """Strassen decomposition for 2x2 * 2x2 -> 2x2 (7 terms)"""
        add_term({(i0,j0):1, (i1,j1):1}, {(j0,k0):1, (j1,k1):1}, {(k0,i0):1, (k1,i1):1})
        add_term({(i1,j0):1, (i1,j1):1}, {(j0,k0):1}, {(k0,i1):1, (k1,i1):-1})
        add_term({(i0,j0):1}, {(j0,k1):1, (j1,k1):-1}, {(k1,i0):1, (k1,i1):1})
        add_term({(i1,j1):1}, {(j1,k0):1, (j0,k0):-1}, {(k0,i0):1, (k0,i1):1})
        add_term({(i0,j0):1, (i0,j1):1}, {(j1,k1):1}, {(k0,i0):-1, (k1,i0):1})
        add_term({(i1,j0):1, (i0,j0):-1}, {(j0,k0):1, (j0,k1):1}, {(k1,i1):1})
        add_term({(i0,j1):1, (i1,j1):-1}, {(j1,k0):1, (j1,k1):1}, {(k0,i0):1})

    def schoolbook_2x3_3x2(i0, i1, j0, j1, j2, k0, k1):
        """2x3 * 3x2 -> 2x2 using schoolbook (12 terms)"""
        for i in [i0, i1]:
            for j in [j0, j1, j2]:
                for k in [k0, k1]:
                    add_term({(i,j):1}, {(j,k):1}, {(k,i):1})

    def schoolbook_3x2_2x3(i0, i1, i2, j0, j1, k0, k1, k2):
        """3x2 * 2x3 -> 3x3 using schoolbook (18 terms)"""
        for i in [i0, i1, i2]:
            for j in [j0, j1]:
                for k in [k0, k1, k2]:
                    add_term({(i,j):1}, {(j,k):1}, {(k,i):1})

    def schoolbook_2x2_2x3(i0, i1, j0, j1, k0, k1, k2):
        """2x2 * 2x3 -> 2x3 using schoolbook (12 terms)"""
        for i in [i0, i1]:
            for j in [j0, j1]:
                for k in [k0, k1, k2]:
                    add_term({(i,j):1}, {(j,k):1}, {(k,i):1})

    def schoolbook_3x2_2x2(i0, i1, i2, j0, j1, k0, k1):
        """3x2 * 2x2 -> 3x2 using schoolbook (12 terms)"""
        for i in [i0, i1, i2]:
            for j in [j0, j1]:
                for k in [k0, k1]:
                    add_term({(i,j):1}, {(j,k):1}, {(k,i):1})

    # Block indices: rows/cols 0,1 are "2" block; rows/cols 2,3,4 are "3" block
    # Split "3" into {2,3} (2x2) and {4} (1x1) for Strassen applicability

    # C_22 (2x2) = A_22(2x2) * B_22(2x2) + A_23(2x3) * B_32(3x2)
    # A_22 * B_22: Strassen (7)
    strassen_2x2(0, 1, 0, 1, 0, 1)
    # A_23 * B_32: split j into {2,3} and {4}
    #   A_23a(2x2, j=2,3) * B_32a(2x2, j=2,3): Strassen (7)
    strassen_2x2(0, 1, 2, 3, 0, 1)
    #   A_23b(2x1, j=4) * B_32b(1x2, j=4): schoolbook (4)
    for i in [0, 1]:
        for k in [0, 1]:
            add_term({(i,4):1}, {(4,k):1}, {(k,i):1})
    # Subtotal C_22: 7 + 7 + 4 = 18

    # C_23 (2x3) = A_22(2x2) * B_23(2x3) + A_23(2x3) * B_33(3x3)
    # A_22 * B_23: split k into {2,3} and {4}
    #   A_22 * B_23a(2x2, k=2,3): Strassen (7)
    strassen_2x2(0, 1, 0, 1, 2, 3)
    #   A_22 * B_23b(2x1, k=4): schoolbook (4)
    for i in [0, 1]:
        for j in [0, 1]:
            add_term({(i,j):1}, {(j,4):1}, {(4,i):1})
    # A_23 * B_33: split j into {2,3},{4} and k into {2,3},{4}
    #   A_23a(2x2, j=2,3) * B_33a(2x2, j=2,3,k=2,3): Strassen (7)
    strassen_2x2(0, 1, 2, 3, 2, 3)
    #   A_23a(2x2, j=2,3) * B_33b(2x1, j=2,3,k=4): schoolbook (4)
    for i in [0, 1]:
        for j in [2, 3]:
            add_term({(i,j):1}, {(j,4):1}, {(4,i):1})
    #   A_23b(2x1, j=4) * B_33b(1x3, j=4,k=2,3,4): schoolbook (6)
    for i in [0, 1]:
        for k in [2, 3, 4]:
            add_term({(i,4):1}, {(4,k):1}, {(k,i):1})
    # Subtotal C_23: 7 + 4 + 7 + 4 + 6 = 28

    # C_32 (3x2) = A_32(3x2) * B_22(2x2) + A_33(3x3) * B_32(3x2)
    # A_32 * B_22: split i into {2,3} and {4}
    #   A_32a(2x2, i=2,3) * B_22(2x2): Strassen (7)
    strassen_2x2(2, 3, 0, 1, 0, 1)
    #   A_32b(1x2, i=4) * B_22(2x2): schoolbook (4)
    for j in [0, 1]:
        for k in [0, 1]:
            add_term({(4,j):1}, {(j,k):1}, {(k,4):1})
    # A_33 * B_32: split i into {2,3},{4}, j into {2,3},{4}, k into {0,1}
    #   A_33a(2x2, i=2,3,j=2,3) * B_32a(2x2, j=2,3,k=0,1): Strassen (7)
    strassen_2x2(2, 3, 2, 3, 0, 1)
    #   A_33b(2x1, i=2,3,j=4) * B_32b(1x2, j=4,k=0,1): schoolbook (4)
    for i in [2, 3]:
        for k in [0, 1]:
            add_term({(i,4):1}, {(4,k):1}, {(k,i):1})
    #   A_33c(1x2, i=4,j=2,3) * B_32a(2x2, j=2,3,k=0,1): schoolbook (4)
    for j in [2, 3]:
        for k in [0, 1]:
            add_term({(4,j):1}, {(j,k):1}, {(k,4):1})
    #   A_33d(1x1, i=4,j=4) * B_32b(1x2, j=4,k=0,1): schoolbook (2)
    for k in [0, 1]:
        add_term({(4,4):1}, {(4,k):1}, {(k,4):1})
    # Subtotal C_32: 7 + 4 + 7 + 4 + 4 + 2 = 28

    # C_33 (3x3) = A_32(3x2) * B_23(2x3) + A_33(3x3) * B_33(3x3)
    # A_32 * B_23: split i into {2,3},{4}, k into {2,3},{4}
    #   A_32a(2x2, i=2,3) * B_23a(2x2, k=2,3): Strassen (7)
    strassen_2x2(2, 3, 0, 1, 2, 3)
    #   A_32a(2x2, i=2,3) * B_23b(2x1, k=4): schoolbook (4)
    for i in [2, 3]:
        for j in [0, 1]:
            add_term({(i,j):1}, {(j,4):1}, {(4,i):1})
    #   A_32b(1x2, i=4) * B_23a(2x2, k=2,3): schoolbook (4)
    for j in [0, 1]:
        for k in [2, 3]:
            add_term({(4,j):1}, {(j,k):1}, {(k,4):1})
    #   A_32b(1x2, i=4) * B_23b(2x1, k=4): schoolbook (2)
    for j in [0, 1]:
        add_term({(4,j):1}, {(j,4):1}, {(4,4):1})
    # A_33 * B_33: split i into {2,3},{4}, j into {2,3},{4}, k into {2,3},{4}
    #   A_33a(2x2, i=2,3,j=2,3) * B_33a(2x2, j=2,3,k=2,3): Strassen (7)
    strassen_2x2(2, 3, 2, 3, 2, 3)
    #   A_33b(2x1, i=2,3,j=4) * B_33b(1x2, j=4,k=2,3): schoolbook (4)
    for i in [2, 3]:
        for k in [2, 3]:
            add_term({(i,4):1}, {(4,k):1}, {(k,i):1})
    #   A_33c(1x2, i=4,j=2,3) * B_33a(2x2, j=2,3,k=2,3): schoolbook (4)
    for j in [2, 3]:
        for k in [2, 3]:
            add_term({(4,j):1}, {(j,k):1}, {(k,4):1})
    #   A_33d(1x1, i=4,j=4) * B_33b(1x2, j=4,k=2,3): schoolbook (2)
    for k in [2, 3]:
        add_term({(4,4):1}, {(4,k):1}, {(k,4):1})
    #   A_33a * B_33c(2x1, j=2,3,k=4): schoolbook (4)
    for i in [2, 3]:
        for j in [2, 3]:
            add_term({(i,j):1}, {(j,4):1}, {(4,i):1})
    #   A_33b(2x1, i=2,3,j=4) * B_33c(1x1, j=4,k=4): schoolbook (2)
    for i in [2, 3]:
        add_term({(i,4):1}, {(4,4):1}, {(4,i):1})
    #   A_33c(1x2, i=4,j=2,3) * B_33c(1x1, j=4,k=4): schoolbook (2)
    for j in [2, 3]:
        add_term({(4,j):1}, {(j,4):1}, {(4,4):1})
    #   A_33d(1x1, i=4,j=4) * B_33c(1x1, j=4,k=4): schoolbook (1)
    add_term({(4,4):1}, {(4,4):1}, {(4,4):1})
    # Subtotal C_33: 7 + 4 + 4 + 2 + 7 + 4 + 4 + 2 + 4 + 2 + 2 + 1 = 43

    # Total rank: 18 + 28 + 28 + 43 = 117
    # Wait, that's less than 120! Let me verify the count.

    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T

    rank = U.shape[1]

    if time.time() - t0 > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
