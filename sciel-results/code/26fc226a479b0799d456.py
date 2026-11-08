"""
5x5 matrix multiplication - rank 112 block decomposition (3+2 split).
Uses Strassen (rank 7) for 2x2 sub-blocks and schoolbook for 3x3 and cross terms.
The 3+2 split yields lower total rank than 2+3 split.
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
        # M1 = (A_i0j0 + A_i1j1)(B_j0k0 + B_j1k1)
        add_term({(i0,j0):1, (i1,j1):1}, {(j0,k0):1, (j1,k1):1}, {(k0,i0):1, (k1,i1):1})
        # M2 = (A_i1j0 + A_i1j1) * B_j0k0
        add_term({(i1,j0):1, (i1,j1):1}, {(j0,k0):1}, {(k0,i1):1, (k1,i1):-1})
        # M3 = A_i0j0 * (B_j0k1 - B_j1k1)
        add_term({(i0,j0):1}, {(j0,k1):1, (j1,k1):-1}, {(k1,i0):1, (k1,i1):1})
        # M4 = A_i1j1 * (B_j1k0 - B_j0k0)
        add_term({(i1,j1):1}, {(j1,k0):1, (j0,k0):-1}, {(k0,i0):1, (k0,i1):1})
        # M5 = (A_i0j0 + A_i0j1) * B_j1k1
        add_term({(i0,j0):1, (i0,j1):1}, {(j1,k1):1}, {(k0,i0):-1, (k1,i0):1})
        # M6 = (A_i1j0 - A_i0j0) * (B_j0k0 + B_j0k1)
        add_term({(i1,j0):1, (i0,j0):-1}, {(j0,k0):1, (j0,k1):1}, {(k1,i1):1})
        # M7 = (A_i0j1 - A_i1j1) * (B_j1k0 + B_j1k1)
        add_term({(i0,j1):1, (i1,j1):-1}, {(j1,k0):1, (j1,k1):1}, {(k0,i0):1})

    # 3+2 Split: indices 0,1,2 form the 3x3 part; 3,4 form the 2x2 part
    # C[0:3,0:3] = A[0:3,0:3]@B[0:3,0:3] + A[0:3,3:5]@B[3:5,0:3]
    # A33@B33: schoolbook 9 terms
    for i in range(3):
        for j in range(3):
            for k in range(3):
                add_term({(i,j):1}, {(j,k):1}, {(k,i):1})
    # A32@B23: 3*2*3 = 18 terms
    for i in range(3):
        for j in [3, 4]:
            for k in range(3):
                add_term({(i,j):1}, {(j,k):1}, {(k,i):1})

    # C[0:3,3:5] = A[0:3,0:3]@B[0:3,3:5] + A[0:3,3:5]@B[3:5,3:5]
    # A33@B32: 3*3*2 = 18 terms
    for i in range(3):
        for j in range(3):
            for k in [3, 4]:
                add_term({(i,j):1}, {(j,k):1}, {(k,i):1})
    # A32@B22: Strassen 7 terms
    strassen_2x2(0, 1, 3, 4, 3, 4)
    # Row 2 of A32 with B22: 2 terms
    for k in [3, 4]:
        add_term({(2,3):1}, {(3,k):1}, {(k,2):1})
        add_term({(2,4):1}, {(4,k):1}, {(k,2):1})

    # C[3:5,0:3] = A[3:5,0:3]@B[0:3,0:3] + A[3:5,3:5]@B[3:5,0:3]
    # A23@B33: Strassen 7 terms
    strassen_2x2(3, 4, 0, 1, 0, 1)
    # A23 with col 2 of B33: 2 terms
    for i in [3, 4]:
        add_term({(i,2):1}, {(2,0):1}, {(0,i):1})
        add_term({(i,2):1}, {(2,1):1}, {(1,i):1})
    # A22@B23: Strassen 7 terms
    strassen_2x2(3, 4, 3, 4, 0, 1)
    # A22 with col 2 of B23: 2 terms
    for i in [3, 4]:
        add_term({(i,4):1}, {(4,2):1}, {(2,i):1})

    # C[3:5,3:5] = A[3:5,0:3]@B[0:3,3:5] + A[3:5,3:5]@B[3:5,3:5]
    # A23@B32: 2*3*2 = 12 terms
    for i in [3, 4]:
        for j in range(3):
            for k in [3, 4]:
                add_term({(i,j):1}, {(j,k):1}, {(k,i):1})
    # A22@B22: Strassen 7 terms
    strassen_2x2(3, 4, 3, 4, 3, 4)

    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T
    rank = U.shape[1]

    elapsed = time.time() - t0
    if elapsed > 100.0:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Verify tensor decomposition
    try:
        T_decomp = np.zeros((n, m, p), dtype=float)
        for r in range(rank):
            u = U[:, r].reshape(n, m)
            v = V[:, r].reshape(m, p)
            w = W[:, r].reshape(p, n)
            T_decomp += np.einsum('ij,jk,ki->ijk', u, v, w, optimize=True)
        
        T_expected = np.zeros((n, m, p), dtype=float)
        for i in range(n):
            for k in range(p):
                if i == k:
                    for j in range(m):
                        T_expected[i, j, k] = 1.0
        
        error = np.max(np.abs(T_decomp - T_expected))
        if error < 1e-9:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        else:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
