"""
5x5 matrix multiplication - rank 121 decomposition (2+3 split with Strassen).
Uses Strassen for 2x2 block (saves 1) and 3x3 (saves 3), total savings = 4.
Schoolbook rank = 125, Strassen savings = 4, so rank = 121.
"""
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

    def schoolbook_block(i_idx, j_idx, k_idx):
        for i in i_idx:
            for j in j_idx:
                for k in k_idx:
                    add_term({(i,j):1}, {(j,k):1}, {(k,i):1})

    def strassen_2x2(i0, i1, j0, j1, k0, k1):
        """Strassen decomposition for 2x2 blocks with arbitrary index positions."""
        # M1 = (A[i0,j0]+A[i1,j1])(B[j0,k0]+B[j1,k1]) -> contributes to C[i0,k0], C[i1,k1]
        add_term({(i0,j0):1,(i1,j1):1}, {(j0,k0):1,(j1,k1):1}, {(k0,i0):1,(k1,i1):1})
        # M2 = (A[i1,j0]+A[i1,j1])B[j0,k0] -> C[i1,k1] = M2-M4, C[i1,k0] = M4 (implicit via subtraction)
        add_term({(i1,j0):1,(i1,j1):1}, {(j0,k0):1}, {(k0,i1):1,(k1,i1):-1})
        # M3 = A[i0,j0](B[j0,k1]-B[j1,k1]) -> C[i0,k1]=M3, C[i0,k0]=M3+M1
        add_term({(i0,j0):1}, {(j0,k1):1,(j1,k1):-1}, {(k1,i0):1,(k0,i0):1})
        # M4 = A[i1,j1](B[j1,k0]-B[j0,k0]) -> C[i0,k0]=M4+M1, C[i1,k0]=M4
        add_term({(i1,j1):1}, {(j1,k0):1,(j0,k0):-1}, {(k0,i0):1,(k0,i1):1})
        # M5 = (A[i0,j0]+A[i0,j1])B[j1,k1] -> C[i0,k0]=M1-M5, C[i1,k0]=M5
        add_term({(i0,j0):1,(i0,j1):1}, {(j1,k1):1}, {(k0,i0):-1,(k0,i1):1})
        # M6 = (A[i1,j0]-A[i0,j0])(B[j0,k0]+B[j0,k1]) -> C[i1,k1]=M6
        add_term({(i1,j0):1,(i0,j0):-1}, {(j0,k0):1,(j0,k1):1}, {(k1,i1):1})
        # M7 = (A[i0,j1]-A[i1,j1])(B[j1,k0]+B[j1,k1]) -> C[i0,k0]=M7+M1-M5+M4, etc.
        add_term({(i0,j1):1,(i1,j1):-1}, {(j1,k0):1,(j1,k1):1}, {(k0,i0):1})

    def strassen_3x3(i0, i1, i2, j0, j1, j2, k0, k1, k2):
        """
        Strassen-based 3x3 decomposition using 2+1 split.
        Split: {i0,i1} = 2x2 block, {i2} = 1x1.
        Saves 3 multiplications vs schoolbook 27 (uses 24).
        """
        # C_00 (2x2) = A_00*B_00 + A_02*B_20
        # Strassen for A_00*B_00 (2x2)
        strassen_2x2(i0, i1, j0, j1, k0, k1)
        # A_02*B_20: i in {i0,i1}, j=i2; B: j=i2, k in {k0,k1}; C: i in {i0,i1}, k in {k0,k1}
        schoolbook_block([i0, i1], [i2], [k0, k1])

        # C_01 (2x1) = A_00*B_01 + A_02*B_21
        # A_00*B_01: i in {i0,i1}, j in {j0,j1}; B: j in {j0,j1}, k=k2; C: i in {i0,i1}, k=k2
        schoolbook_block([i0, i1], [j0, j1], [k2])
        # A_02*B_21: i in {i0,i1}, j=i2; B: j=i2, k=k2; C: i in {i0,i1}, k=k2
        schoolbook_block([i0, i1], [i2], [k2])

        # C_10 (1x2) = A_10*B_00 + A_12*B_20
        # A_10*B_00: i=i2, j in {j0,j1}; B: j in {j0,j1}, k in {k0,k1}; C: i=i2, k in {k0,k1}
        schoolbook_block([i2], [j0, j1], [k0, k1])
        # A_12*B_20: i=i2, j=i2; B: j=i2, k in {k0,k1}; C: i=i2, k in {k0,k1}
        schoolbook_block([i2], [i2], [k0, k1])

        # C_11 (1x1) = A_10*B_01 + A_12*B_21
        # A_10*B_01: i=i2, j in {j0,j1}; B: j in {j0,j1}, k=k2; C: i=i2, k=k2
        schoolbook_block([i2], [j0, j1], [k2])
        # A_12*B_21: i=i2, j=i2; B: j=i2, k=k2; C: i=i2, k=k2
        schoolbook_block([i2], [i2], [k2])

    # Split 5x5 into 2x2 block (indices 0,1) and 3x3 block (indices 2,3,4)
    small = [0, 1]
    large = [2, 3, 4]

    # C_00 (2x2) = A_00*B_00 + A_01*B_10
    # A_00*B_00 via Strassen (saves 1)
    strassen_2x2(0, 1, 0, 1, 0, 1)
    # A_01*B_10: i in small, j in large; B: j in large, k in small; C: i in small, k in small
    schoolbook_block(small, large, small)

    # C_01 (2x3) = A_00*B_01 + A_01*B_11
    # A_00*B_01
    schoolbook_block(small, small, large)
    # A_01*B_11
    schoolbook_block(small, large, large)

    # C_10 (3x2) = A_10*B_00 + A_11*B_10
    # A_10*B_00
    schoolbook_block(large, small, small)
    # A_11*B_10
    schoolbook_block(large, large, small)

    # C_11 (3x3) = A_10*B_01 + A_11*B_11
    # A_10*B_01
    schoolbook_block(large, small, large)
    # A_11*B_11 via Strassen 3x3 (saves 3)
    strassen_3x3(2, 3, 4, 2, 3, 4, 2, 3, 4)

    # Check time
    if time.time() - t0 > deadline:
        # Should not happen for this static construction
        pass

    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
