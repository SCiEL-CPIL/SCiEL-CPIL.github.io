"""
5x5 matrix multiplication - rank 120 block decomposition (2+3 split).
Uses Strassen for 2x2 blocks and optimized 2+1 split for 3x3 blocks.
"""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = 130.0  # Safety margin: evaluator kills at 150s

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

    # 1. C_22 = A_22 B_22 + A_23 B_32
    # Strassen for A_22 B_22 (2x2 * 2x2)
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

    strassen_2x2(0, 1, 0, 1, 0, 1)

    # A_23 B_32: A is 2x3 (i=0,1; j=2,3,4), B is 3x2 (j=2,3,4; k=0,1)
    # Use 2+1 split on j: j={2,3} and j={4}
    # A_23a (2x2) * B_32a (2x2): Strassen
    strassen_2x2(0, 1, 2, 3, 0, 1)
    # A_23b (2x1, j=4) * B_32b (1x2, j=4): schoolbook 2*1*2 = 4
    for i in [0, 1]:
        for k in [0, 1]:
            add_term({(i,4):1}, {(4,k):1}, {(k,i):1})

    # 2. C_23 = A_22 B_23 + A_23 B_33
    # A_22 (2x2) * B_23 (2x3): Strassen on 2x2 part + schoolbook
    strassen_2x2(0, 1, 0, 1, 2, 3)
    # A_22 * B_23b (j=0,1; k=4): schoolbook 2*2*1 = 4
    for i in [0, 1]:
        for j in [0, 1]:
            add_term({(i,j):1}, {(j,4):1}, {(4,i):1})
    
    # A_23 (2x3) * B_33 (3x3): 
    # A_23a (2x2, j=2,3) * B_33a (2x2, j=2,3,k=2,3): Strassen
    strassen_2x2(0, 1, 2, 3, 2, 3)
    # A_23a * B_33b (j=2,3; k=4): schoolbook 2*2*1 = 4
    for i in [0, 1]:
        for j in [2, 3]:
            add_term({(i,j):1}, {(j,4):1}, {(4,i):1})
    # A_23b (2x1, j=4) * B_33b (1x3, j=4; k=2,3,4): schoolbook 2*1*3 = 6
    for i in [0, 1]:
        for k in [2, 3, 4]:
            add_term({(i,4):1}, {(4,k):1}, {(k,i):1})

    # 3. C_32 = A_32 B_22 + A_33 B_32
    # A_32 (3x2) * B_22 (2x2):
    # A_32a (2x2, i=2,3) * B_22: Strassen
    strassen_2x2(2, 3, 0, 1, 0, 1)
    # A_32b (1x2, i=4) * B_22: schoolbook 1*2*2 = 4
    for j in [0, 1]:
        for k in [0, 1]:
            add_term({(4,j):1}, {(j,k):1}, {(k,4):1})
    
    # A_33 (3x3) * B_32 (3x2):
    # A_33a (2x2, i=2,3; j=2,3) * B_32a (2x2, j=2,3; k=0,1): Strassen
    strassen_2x2(2, 3, 2, 3, 0, 1)
    # A_33b (2x1, i=2,3; j=4) * B_32b (1x2, j=4; k=0,1): schoolbook 2*1*2 = 4
    for i in [2, 3]:
        for k in [0, 1]:
            add_term({(i,4):1}, {(4,k):1}, {(k,i):1})
    # A_33c (1x2, i=4; j=2,3) * B_32a (2x2, j=2,3; k=0,1): schoolbook 1*2*2 = 4
    for j in [2, 3]:
        for k in [0, 1]:
            add_term({(4,j):1}, {(j,k):1}, {(k,4):1})
    # A_33d (1x1, i=4; j=4) * B_32b (1x2, j=4; k=0,1): schoolbook 1*1*2 = 2
    for k in [0, 1]:
        add_term({(4,4):1}, {(4,k):1}, {(k,4):1})

    # 4. C_33 = A_32 B_23 + A_33 B_33
    # A_32 (3x2) * B_23 (2x3):
    # A_32a (2x2, i=2,3) * B_23a (2x2, k=2,3): Strassen
    strassen_2x2(2, 3, 0, 1, 2, 3)
    # A_32a * B_23b (k=4): schoolbook 2*2*1 = 4
    for i in [2, 3]:
        for j in [0, 1]:
            add_term({(i,j):1}, {(j,4):1}, {(4,i):1})
    # A_32b (1x2, i=4) * B_23a (2x2, k=2,3): schoolbook 1*2*2 = 4
    for j in [0, 1]:
        for k in [2, 3]:
            add_term({(4,j):1}, {(j,k):1}, {(k,4):1})
    # A_32b * B_23b (k=4): schoolbook 1*2*1 = 2
    for j in [0, 1]:
        add_term({(4,j):1}, {(j,4):1}, {(4,4):1})
    
    # A_33 B_33 (3x3 * 3x3) with 2+1 split
    # C22 = A22*B22 + A21*B12
    strassen_2x2(2, 3, 2, 3, 2, 3)
    # A21*B12: i in {2,3}, j=4; k in {2,3}
    for i in [2, 3]:
        for k in [2, 3]:
            add_term({(i,4):1}, {(4,k):1}, {(k,i):1})
    
    # C23 = A22*B23 + A21*B13
    # A22*B23: i in {2,3}, j in {2,3}; k=4
    for i in [2, 3]:
        for j in [2, 3]:
            add_term({(i,j):1}, {(j,4):1}, {(4,i):1})
    # A21*B13: i in {2,3}, j=4; k=4
    for i in [2, 3]:
        add_term({(i,4):1}, {(4,4):1}, {(4,i):1})
    
    # C32 = A12*B22 + A11*B32
    # A12*B22: i=4, j in {2,3}; k in {2,3}
    for j in [2, 3]:
        for k in [2, 3]:
            add_term({(4,j):1}, {(j,k):1}, {(k,4):1})
    # A11*B32: i=4, j=4; k in {2,3}
    for k in [2, 3]:
        add_term({(4,4):1}, {(4,k):1}, {(k,4):1})
    
    # C33 = A12*B23 + A11*B33
    # A12*B23: i=4, j in {2,3}; k=4
    for j in [2, 3]:
        add_term({(4,j):1}, {(j,4):1}, {(4,4):1})
    # A11*B33: i=4, j=4; k=4
    add_term({(4,4):1}, {(4,4):1}, {(4,4):1})

    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T

    rank = U.shape[1]

    # Quick verification with a small random test
    if time.time() - t0 < deadline - 10.0:
        np.random.seed(123)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test

        # Build the tensor contraction efficiently
        # T[i,j,k] = sum_r U[r, i*m+j] * V[r, j*p+k] * W[r, k*n+i]
        # C[i,k] = sum_j T[i,j,k]
        C_computed = np.zeros((n, p))
        for r in range(rank):
            u_r = U[:, r].reshape(n, m)
            v_r = V[:, r].reshape(m, p)
            w_r = W[:, r].reshape(p, n)
            # C[i,k] += sum_j u_r[i,j] * v_r[j,k] * w_r[k,i]
            for i in range(n):
                for k in range(p):
                    s = 0.0
                    for j in range(m):
                        s += u_r[i,j] * v_r[j,k] * w_r[k,i]
                    C_computed[i,k] += s

        if not np.allclose(C_computed, C_expected, atol=1e-10):
            # Fallback: return schoolbook (rank 125)
            U_list2 = []
            V_list2 = []
            W_list2 = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U2 = np.zeros(n*m)
                        V2 = np.zeros(m*p)
                        W2 = np.zeros(p*n)
                        U2[i*m+j] = 1
                        V2[j*p+k] = 1
                        W2[k*n+i] = 1
                        U_list2.append(U2)
                        V_list2.append(V2)
                        W_list2.append(W2)
            U = np.array(U_list2).T
            V = np.array(V_list2).T
            W = np.array(W_list2).T

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
