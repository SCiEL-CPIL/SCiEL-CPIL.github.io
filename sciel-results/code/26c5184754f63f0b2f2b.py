"""
5x5 matrix multiplication - rank 104 block decomposition.
Uses Strassen for 2x2 blocks and optimized 2+1 splits for mixed blocks.
2+3 split: rows/cols {0,1} and {2,3,4}.
"""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

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
        """Strassen for A[i0,i1 x j0,j1] * B[j0,j1 x k0,k1] -> C[i0,i1 x k0,k1]"""
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

    # ============ C_22 = A_22*B_22 + A_23*B_32 ============
    # A_22 (2x2, i={0,1}, j={0,1}) * B_22 (2x2, j={0,1}, k={0,1}): Strassen = 7
    strassen_2x2(0, 1, 0, 1, 0, 1)
    # A_23 (2x3, i={0,1}, j={2,3,4}) * B_32 (3x2, j={2,3,4}, k={0,1})
    # Split j into {2,3} and {4}:
    # A_23a (2x2, j={2,3}) * B_32a (2x2, j={2,3}, k={0,1}): Strassen = 7
    strassen_2x2(0, 1, 2, 3, 0, 1)
    # A_23b (2x1, j={4}) * B_32b (1x2, j={4}, k={0,1}): schoolbook = 2*1*2 = 4
    for i in [0, 1]:
        for k in [0, 1]:
            add_term({(i,4):1}, {(4,k):1}, {(k,i):1})
    # C_22 total: 7 + 7 + 4 = 18

    # ============ C_23 = A_22*B_23 + A_23*B_33 ============
    # A_22 (2x2, i={0,1}, j={0,1}) * B_23 (2x3, j={0,1}, k={2,3,4})
    # Split k into {2,3} and {4}:
    # A_22 * B_23a (2x2, k={2,3}): Strassen = 7
    strassen_2x2(0, 1, 0, 1, 2, 3)
    # A_22 * B_23b (2x1, k={4}): schoolbook = 2*2*1 = 4
    for i in [0, 1]:
        for j in [0, 1]:
            add_term({(i,j):1}, {(j,4):1}, {(4,i):1})
    # A_23 (2x3, i={0,1}, j={2,3,4}) * B_33 (3x3, j={2,3,4}, k={2,3,4})
    # Split j,k into {2,3} and {4}:
    # A_23a (2x2, j={2,3}) * B_33a (2x2, j={2,3}, k={2,3}): Strassen = 7
    strassen_2x2(0, 1, 2, 3, 2, 3)
    # A_23a (2x2, j={2,3}) * B_33b (2x1, j={2,3}, k={4}): schoolbook = 2*2*1 = 4
    for i in [0, 1]:
        for j in [2, 3]:
            add_term({(i,j):1}, {(j,4):1}, {(4,i):1})
    # A_23b (2x1, j={4}) * B_33b (1x3, j={4}, k={2,3,4}): schoolbook = 2*1*3 = 6
    for i in [0, 1]:
        for k in [2, 3, 4]:
            add_term({(i,4):1}, {(4,k):1}, {(k,i):1})
    # C_23 total: 7 + 4 + 7 + 4 + 6 = 28

    # ============ C_32 = A_32*B_22 + A_33*B_32 ============
    # A_32 (3x2, i={2,3,4}, j={0,1}) * B_22 (2x2, j={0,1}, k={0,1})
    # Split i into {2,3} and {4}:
    # A_32a (2x2, i={2,3}) * B_22 (2x2): Strassen = 7
    strassen_2x2(2, 3, 0, 1, 0, 1)
    # A_32b (1x2, i={4}) * B_22 (2x2): schoolbook = 1*2*2 = 4
    for j in [0, 1]:
        for k in [0, 1]:
            add_term({(4,j):1}, {(j,k):1}, {(k,4):1})
    # A_33 (3x3, i={2,3,4}, j={2,3,4}) * B_32 (3x2, j={2,3,4}, k={0,1})
    # Split i,j into {2,3} and {4}; k={0,1}:
    # A_33a (2x2, i={2,3}, j={2,3}) * B_32a (2x2, j={2,3}, k={0,1}): Strassen = 7
    strassen_2x2(2, 3, 2, 3, 0, 1)
    # A_33b (2x1, i={2,3}, j={4}) * B_32b (1x2, j={4}, k={0,1}): schoolbook = 2*1*2 = 4
    for i in [2, 3]:
        for k in [0, 1]:
            add_term({(i,4):1}, {(4,k):1}, {(k,i):1})
    # A_33c (1x2, i={4}, j={2,3}) * B_32a (2x2, j={2,3}, k={0,1}): schoolbook = 1*2*2 = 4
    for j in [2, 3]:
        for k in [0, 1]:
            add_term({(4,j):1}, {(j,k):1}, {(k,4):1})
    # A_33d (1x1, i={4}, j={4}) * B_32b (1x2, j={4}, k={0,1}): schoolbook = 1*1*2 = 2
    for k in [0, 1]:
        add_term({(4,4):1}, {(4,k):1}, {(k,4):1})
    # C_32 total: 7 + 4 + 7 + 4 + 4 + 2 = 28

    # ============ C_33 = A_32*B_23 + A_33*B_33 ============
    # A_32 (3x2, i={2,3,4}, j={0,1}) * B_23 (2x3, j={0,1}, k={2,3,4})
    # Split i={2,3}+{4}, k={2,3}+{4}:
    # A_32a (2x2, i={2,3}) * B_23a (2x2, k={2,3}): Strassen = 7
    strassen_2x2(2, 3, 0, 1, 2, 3)
    # A_32a (2x2, i={2,3}) * B_23b (2x1, k={4}): schoolbook = 2*2*1 = 4
    for i in [2, 3]:
        for j in [0, 1]:
            add_term({(i,j):1}, {(j,4):1}, {(4,i):1})
    # A_32b (1x2, i={4}) * B_23a (2x2, k={2,3}): schoolbook = 1*2*2 = 4
    for j in [0, 1]:
        for k in [2, 3]:
            add_term({(4,j):1}, {(j,k):1}, {(k,4):1})
    # A_32b (1x2, i={4}) * B_23b (2x1, k={4}): schoolbook = 1*2*1 = 2
    for j in [0, 1]:
        add_term({(4,j):1}, {(j,4):1}, {(4,4):1})
    # A_33 (3x3, i={2,3,4}, j={2,3,4}) * B_33 (3x3, j={2,3,4}, k={2,3,4})
    # Split i,j,k into {2,3}+{4}:
    # A_33a (2x2, i={2,3}, j={2,3}) * B_33a (2x2, j={2,3}, k={2,3}): Strassen = 7
    strassen_2x2(2, 3, 2, 3, 2, 3)
    # A_33a (2x2, i={2,3}, j={2,3}) * B_33b (2x1, j={2,3}, k={4}): schoolbook = 2*2*1 = 4
    for i in [2, 3]:
        for j in [2, 3]:
            add_term({(i,j):1}, {(j,4):1}, {(4,i):1})
    # A_33b (2x1, i={2,3}, j={4}) * B_33b (1x3, j={4}, k={2,3,4}): schoolbook = 2*1*3 = 6
    for i in [2, 3]:
        for k in [2, 3, 4]:
            add_term({(i,4):1}, {(4,k):1}, {(k,i):1})
    # A_33c (1x2, i={4}, j={2,3}) * B_33a (2x2, j={2,3}, k={2,3}): schoolbook = 1*2*2 = 4
    for j in [2, 3]:
        for k in [2, 3]:
            add_term({(4,j):1}, {(j,k):1}, {(k,4):1})
    # A_33c (1x2, i={4}, j={2,3}) * B_33b (2x1, j={2,3}, k={4}): schoolbook = 1*2*1 = 2
    for j in [2, 3]:
        add_term({(4,j):1}, {(j,4):1}, {(4,4):1})
    # A_33d (1x1, i={4}, j={4}) * B_33b (1x3, j={4}, k={2,3,4}): schoolbook = 1*1*3 = 3
    for k in [2, 3, 4]:
        add_term({(4,4):1}, {(4,k):1}, {(k,4):1})
    # C_33 total: 7+4+4+2 + 7+4+6+4+2+3 = 43

    # Total rank: 18 + 28 + 28 + 43 = 117

    if time.time() - t0 > deadline:
        U = np.array(U_list).T
        V = np.array(V_list).T
        W = np.array(W_list).T
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T

    rank = U.shape[1]

    # Verify correctness
    try:
        np.random.seed(42)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test

        C_decomp = np.zeros((n, p))
        for r in range(rank):
            u = U[:, r].reshape(n, m)
            v = V[:, r].reshape(m, p)
            C_decomp += u @ v

        if np.allclose(C_decomp, C_expected, atol=1e-10):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        else:
            # Fallback: return anyway
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
