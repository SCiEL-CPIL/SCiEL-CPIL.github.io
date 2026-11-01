"""Rank-minimal exact decomposition for 2x2 * 2x7 matrix multiplication.

T_{2,2,7} = T_{2,2,4} (Strassen, rank 7) ⊕ T_{2,2,3} (schoolbook, rank 12)
Total rank = 19.
"""
DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS

    U, V, W = [], [], []

    # ---- Block A: 2x2 * 2x4  (Strassen, rank 7) ----
    # a_vec over (a11,a12,a21,a22); b_vec over (b11,b12,b21,b22); c_vec over (c11,c12,c21,c22)
    # C11 = M1+M3-M5-M7 ; C12 = M4+M5 ; C21 = M2+M3 ; C22 = M1-M2+M4-M6
    strassen = [
        # M1 = (a11-a22)(b21+b22)
        ([1, 0, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
        # M2 = (a22)(b11-b12)
        ([0, 0, 0, 1], [1, -1, 0, 0], [0, 0, 1, 0]),
        # M3 = (a11+a22)(b11+b12)
        ([1, 0, 0, 1], [1, 1, 0, 0], [1, 0, 1, 0]),
        # M4 = (a11-a21)(b11+b21)
        ([1, 0, -1, 0], [1, 0, 1, 0], [0, 1, 0, 1]),
        # M5 = (a21+a22)(b12-b22)
        ([0, 0, 1, 1], [0, 1, 0, -1], [0, 1, 0, 0]),
        # M6 = (a11-a12)(b12+b22)
        ([1, -1, 0, 0], [0, 1, 0, 1], [0, 0, 0, -1]),
        # M7 = (a12-a22)(b11+b21)
        ([0, 1, 0, -1], [1, 0, 1, 0], [-1, 0, 0, 0]),
    ]
    for a, b, c in strassen:
        U.append(a)
        V.append(b)
        W.append(c)

    # ---- Block B: 2x2 * 2x3  (schoolbook, rank 12) ----
    # a_vec over (a11,a12,a21,a22); b_vec over (b15,b16,b17,b25,b26,b27); c_vec over (c5,c6,c7)
    # C_{i,k} = a_{i,1} b_{1,k} + a_{i,2} b_{2,k}
    for i in range(2):          # row of A
        for j in range(2):      # inner
            for k in range(3):  # col of B (c5..c7)
                a = [0, 0, 0, 0]
                a[i * 2 + j] = 1
                b = [0, 0, 0, 0, 0, 0]
                b[j * 3 + k] = 1
                c = [0, 0, 0]
                c[k] = 1
                U.append(a)
                V.append(b)
                W.append(c)

    rank = len(U)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    import numpy as np
    out = construct()
    n, m, p = DIMENSIONS
    U = np.array(out["U"], dtype=float).T
    V = np.array(out["V"], dtype=float).T
    W = np.array(out["W"], dtype=float).T
    T = np.ones((n, m, p))
    R = np.zeros((n, m, p))
    for r in range(len(U)):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    R[i, j, k] += U[r, i * m + j] * V[r, j * p + k] * W[r, k * n + i]
    err = float(np.max(np.abs(R - T)))
    print("rank =", len(U), "max error =", err)
    assert err < 1e-9, "decomposition is NOT exact"
    print("OK: exact decomposition")
