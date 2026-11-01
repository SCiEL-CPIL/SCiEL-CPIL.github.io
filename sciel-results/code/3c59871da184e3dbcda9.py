import numpy as np
import time
from itertools import product

def construct():
    n, m, p = 2, 3, 5
    deadline = time.time() + 130

    # Build target tensor T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    # 28-rank construction:
    # Split m=3 into m1=2, m2=1
    # Split p=5 into p1=4, p2=1
    # A1*B1a: (2x2)*(2x2) Strassen rank 7
    # A1*B1b: (2x2)*(2x2) Strassen rank 7
    # A1*B1c: (2x2)*(2x1) schoolbook rank 4
    # A2*B2:   (2x1)*(1x5) schoolbook rank 10
    # Total: 28

    def strassen_2x2x2():
        """Return U,V,W for 2x2 x 2x2 multiplication, rank 7.
        Indices: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
        i,k in {0,1}, j in {0,1}
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))

        # M1 = (a+d)(e+h)
        U[0,0] = 1; U[3,0] = 1
        V[0,0] = 1; V[3,0] = 1
        W[0,0] = 1; W[1,0] = 1; W[2,0] = -1; W[3,0] = 1

        # M2 = (c+d)*e
        U[2,1] = 1; U[3,1] = 1
        V[0,1] = 1
        W[2,1] = 1; W[3,1] = -1

        # M3 = a*(f-h)
        U[0,2] = 1
        V[1,2] = 1; V[3,2] = -1
        W[1,2] = 1; W[3,2] = 1

        # M4 = d*(g-e)
        U[3,3] = 1
        V[2,3] = 1; V[0,3] = -1
        W[0,3] = 1; W[2,3] = 1

        # M5 = (a+b)*h
        U[0,4] = 1; U[1,4] = 1
        V[3,4] = 1
        W[0,4] = -1; W[1,4] = 1

        # M6 = (c-a)*(e+f)
        U[2,5] = 1; U[0,5] = -1
        V[0,5] = 1; V[1,5] = 1
        W[3,5] = 1

        # M7 = (b-d)*(g+h)
        U[1,6] = 1; U[3,6] = -1
        V[2,6] = 1; V[3,6] = 1
        W[0,6] = 1

        return U, V, W

    # Build the 28-rank decomposition
    U_full = np.zeros((6, 28))
    V_full = np.zeros((15, 28))
    W_full = np.zeros((10, 28))

    col = 0

    def add_strassen(i0, j0, k0):
        nonlocal col
        U_s, V_s, W_s = strassen_2x2x2()
        for r in range(7):
            for a_local in range(4):
                i_l, j_l = divmod(a_local, 2)
                U_full[(i0+i_l)*3 + (j0+j_l), col] = U_s[a_local, r]
            for b_local in range(4):
                j_l, k_l = divmod(b_local, 2)
                V_full[(j0+j_l)*5 + (k0+k_l), col] = V_s[b_local, r]
            for c_local in range(4):
                k_l, i_l = divmod(c_local, 2)
                W_full[(k0+k_l)*2 + (i0+i_l), col] = W_s[c_local, r]
            col += 1

    def add_schoolbook(i0, j0, k0, ni, mj, pk):
        nonlocal col
        for i_l in range(ni):
            for j_l in range(mj):
                for k_l in range(pk):
                    U_full[(i0+i_l)*3 + (j0+j_l), col] = 1
                    V_full[(j0+j_l)*5 + (k0+k_l), col] = 1
                    W_full[(k0+k_l)*2 + (i0+i_l), col] = 1
                    col += 1

    # Block 1a: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    add_strassen(0, 0, 0)

    # Block 1b: A[0:2, 0:2] * B[0:2, 2:4] -> C[2:4, 0:2]
    add_strassen(0, 0, 2)

    # Block 2: A[0:2, 0:2] * B[0:2, 4:5] -> C[4:5, 0:2]
    add_schoolbook(0, 0, 4, 2, 2, 1)

    # Block 3: A[0:2, 2:3] * B[2:3, 0:5] -> C[0:5, 0:2]
    add_schoolbook(0, 2, 0, 2, 1, 5)

    assert col == 28, f"Expected 28, got {col}"

    # Verify
    S = np.einsum('ar,br,cr->abc', U_full, V_full, W_full)
    if np.allclose(S, T, atol=1e-10):
        return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}

    # Fallback: schoolbook rank 30
    rank = 30
    U = np.zeros((6, rank))
    V = np.zeros((15, rank))
    W = np.zeros((10, rank))
    col2 = 0
    for i in range(2):
        for j in range(3):
            for k in range(5):
                U[i*3+j, col2] = 1
                V[j*5+k, col2] = 1
                W[k*2+i, col2] = 1
                col2 += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
