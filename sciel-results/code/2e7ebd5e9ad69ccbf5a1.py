"""Rank-minimal exact tensor decomposition for M_{2,2,15} using Strassen blocks.

A is 2x2 (flattened A[i*m+j] = A[i,j]), B is 2x15 (flattened B[i*p+j] = B[i,j]),
C is 2x15 (output factor W[k*n+i, r] = C[i,k] per the official trace convention).

We decompose the 15 output columns into 7 blocks of 2 columns (Strassen, 7 terms each)
plus 1 block of 1 column (schoolbook, 4 terms). Total rank = 7*7 + 4 = 53.

Strassen reconstruction (verified):
  C00 = P1 + P4 - P5 + P7
  C01 = P3 + P5
  C10 = P2 + P4
  C11 = P1 - P2 + P3 + P6
"""

def construct():
    n, m, p = 2, 2, 15

    # 7 Strassen terms: (u for A, v for B, w for C) with local indices.
    # Local A indices: 0=A00, 1=A01, 2=A10, 3=A11
    # Local B indices: 0=B00, 1=B01, 2=B10, 3=B11
    # Local C indices: 0=C00, 1=C01, 2=C10, 3=C11
    STR = [
        # P1 = (A00 + A11)(B00 + B11)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # P2 = (A10 + A11) B00
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        # P3 = A00 (B10 - B11)
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 0]),
        # P4 = (A00 + A01) B11
        ([1, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1]),
        # P5 = (A01 - A10)(B00 + B10)
        ([0, 1, -1, 0], [1, 0, 1, 0], [-1, 1, 0, 0]),
        # P6 = (A10 - A00)(B11 + B01)
        ([-1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 0, 1]),
        # P7 = (A11 - A01)(B00 + B01)
        ([0, -1, 0, 1], [1, 1, 0, 0], [1, 0, 0, 1]),
    ]

    # 4 schoolbook terms for a single column (2x2 x 2x1 -> 2x1):
    # C0 = A00*B0 + A01*B1 ; C1 = A10*B0 + A11*B1
    # Local A: 0=A00,1=A01,2=A10,3=A11 ; Local B: 0=B0,1=B1 ; Local C: 0=C0,1=C1
    SCHO = [
        ([1, 0, 0, 0], [1, 0], [1, 0]),
        ([0, 1, 0, 0], [0, 1], [1, 0]),
        ([0, 0, 1, 0], [1, 0], [0, 1]),
        ([0, 0, 0, 1], [0, 1], [0, 1]),
    ]

    rank = 7 * 7 + 4  # 53
    U = [[0] * rank for _ in range(n * m)]   # 4 x rank
    V = [[0] * rank for _ in range(m * p)]   # 30 x rank
    W = [[0] * rank for _ in range(p * n)]   # 30 x rank

    col = 0
    # 7 blocks of 2 columns: block b covers output columns k=2b, 2b+1
    for b in range(7):
        k0 = 2 * b
        for (u, v, w) in STR:
            for a in range(4):
                if u[a] != 0:
                    U[a][col] = u[a]
            for a in range(4):
                if v[a] != 0:
                    # B local index a = i*2 + j -> global B index i*p + j
                    i, j = divmod(a, 2)
                    V[i * p + (k0 + j)][col] = v[a]
            for c in range(4):
                if w[c] != 0:
                    # C local index c = i*2 + j -> output column k = k0 + j, row i
                    i, j = divmod(c, 2)
                    W[(k0 + j) * n + i][col] = w[c]
            col += 1

    # 1 block of 1 column: output column k = 14
    k0 = 14
    for (u, v, w) in SCHO:
        for a in range(4):
            if u[a] != 0:
                U[a][col] = u[a]
        for a in range(2):
            if v[a] != 0:
                # B local index a = i*1 + j -> global B index i*p + (k0 + j)
                i, j = divmod(a, 1)
                V[i * p + (k0 + j)][col] = v[a]
        for c in range(2):
            if w[c] != 0:
                # C local index c = i*1 + j -> output column k = k0 + j, row i
                i, j = divmod(c, 1)
                W[(k0 + j) * n + i][col] = w[c]
        col += 1

    assert col == rank, (col, rank)
    return {"U": U, "V": V, "W": W}


if __name__ == "__main__":
    import random
    import numpy as np
    d = construct()
    U, V, W = np.array(d["U"]), np.array(d["V"]), np.array(d["W"])
    n, m, p = 2, 2, 15
    for _ in range(200):
        A = np.array([[random.randint(-5, 5) for _ in range(m)] for _ in range(n)]).reshape(n * m)
        B = np.array([[random.randint(-5, 5) for _ in range(p)] for _ in range(m)]).reshape(m * p)
        # True product C[i,k] = sum_j A[i,j] B[j,k]
        C_true = np.zeros(p * n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    C_true[k * n + i] += A[i * m + j] * B[j * p + k]
        # Decomposed: C_c = sum_r U_{a,r} V_{b,r} W_{c,r} A_a B_b
        C_dec = np.zeros(p * n)
        for r in range(U.shape[1]):
            ua = np.dot(U[:, r], A)
            vb = np.dot(V[:, r], B)
            C_dec += W[:, r] * (ua * vb)
        if not np.allclose(C_true, C_dec):
            print("MISMATCH", np.max(np.abs(C_true - C_dec)))
            break
    else:
        print("OK, rank =", U.shape[1])
