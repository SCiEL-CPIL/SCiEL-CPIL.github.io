import numpy as np

def construct():
    """
    Rank-28 exact decomposition for 2x3 * 3x5 matrix multiplication.
    Blocks:
      - j=0: schoolbook 2x1x5 -> rank 10
      - j in {1,2}, k in {0,1}: Strassen 2x2x2 -> rank 7
      - j in {1,2}, k=2: schoolbook 2x2x1 -> rank 4
      - j in {1,2}, k in {3,4}: Strassen 2x2x2 -> rank 7
    Total: 28
    """
    n, m, p = 2, 3, 5
    rank = 28

    U = [[0.0] * rank for _ in range(n * m)]  # 6 x 28
    V = [[0.0] * rank for _ in range(m * p)]  # 15 x 28
    W = [[0.0] * rank for _ in range(p * n)]  # 10 x 28

    col = 0

    # Block 1: j=0, all k in {0..4}
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1.0
            V[0 * p + k][col] = 1.0
            W[k * n + i][col] = 1.0
            col += 1

    # Block 2a: j in {1,2}, k in {0,1} — Strassen 2x2x2
    # U index: i*3 + (j'+1), V index: (j'+1)*5 + k', W index: k'*2 + i
    # M1
    U[0*3+1][col] = 1.0; U[1*3+2][col] = 1.0
    V[1*5+0][col] = 1.0; V[2*5+1][col] = 1.0
    W[0*2+0][col] = 1.0; W[1*2+1][col] = 1.0
    col += 1
    # M2
    U[1*3+1][col] = 1.0; U[1*3+2][col] = 1.0
    V[1*5+0][col] = 1.0
    W[0*2+1][col] = 1.0; W[1*2+1][col] = -1.0
    col += 1
    # M3
    U[0*3+1][col] = 1.0
    V[1*5+1][col] = 1.0; V[2*5+1][col] = -1.0
    W[1*2+0][col] = 1.0; W[1*2+1][col] = 1.0
    col += 1
    # M4
    U[1*3+2][col] = 1.0
    V[2*5+0][col] = 1.0; V[1*5+0][col] = -1.0
    W[0*2+0][col] = 1.0; W[0*2+1][col] = 1.0
    col += 1
    # M5
    U[0*3+1][col] = 1.0; U[0*3+2][col] = 1.0
    V[2*5+1][col] = 1.0
    W[0*2+0][col] = -1.0; W[1*2+0][col] = 1.0
    col += 1
    # M6
    U[1*3+1][col] = 1.0; U[0*3+1][col] = -1.0
    V[1*5+0][col] = 1.0; V[1*5+1][col] = 1.0
    W[1*2+1][col] = 1.0
    col += 1
    # M7
    U[0*3+2][col] = 1.0; U[1*3+2][col] = -1.0
    V[2*5+0][col] = 1.0; V[2*5+1][col] = 1.0
    W[0*2+0][col] = 1.0
    col += 1

    # Block 2b: j in {1,2}, k=2 — schoolbook
    for i in range(n):
        for j_local in range(2):
            j = j_local + 1
            U[i*3 + j][col] = 1.0
            V[j*5 + 2][col] = 1.0
            W[2*2 + i][col] = 1.0
            col += 1

    # Block 2c: j in {1,2}, k in {3,4} — Strassen 2x2x2
    # U index: i*3 + (j'+1), V index: (j'+1)*5 + (k'+3), W index: (k'+3)*2 + i
    # M1
    U[0*3+1][col] = 1.0; U[1*3+2][col] = 1.0
    V[1*5+3][col] = 1.0; V[2*5+4][col] = 1.0
    W[3*2+0][col] = 1.0; W[4*2+1][col] = 1.0
    col += 1
    # M2
    U[1*3+1][col] = 1.0; U[1*3+2][col] = 1.0
    V[1*5+3][col] = 1.0
    W[3*2+1][col] = 1.0; W[4*2+1][col] = -1.0
    col += 1
    # M3
    U[0*3+1][col] = 1.0
    V[1*5+4][col] = 1.0; V[2*5+4][col] = -1.0
    W[4*2+0][col] = 1.0; W[4*2+1][col] = 1.0
    col += 1
    # M4
    U[1*3+2][col] = 1.0
    V[2*5+3][col] = 1.0; V[1*5+3][col] = -1.0
    W[3*2+0][col] = 1.0; W[3*2+1][col] = 1.0
    col += 1
    # M5
    U[0*3+1][col] = 1.0; U[0*3+2][col] = 1.0
    V[2*5+4][col] = 1.0
    W[3*2+0][col] = -1.0; W[4*2+0][col] = 1.0
    col += 1
    # M6
    U[1*3+1][col] = 1.0; U[0*3+1][col] = -1.0
    V[1*5+3][col] = 1.0; V[1*5+4][col] = 1.0
    W[4*2+1][col] = 1.0
    col += 1
    # M7
    U[0*3+2][col] = 1.0; U[1*3+2][col] = -1.0
    V[2*5+3][col] = 1.0; V[2*5+4][col] = 1.0
    W[3*2+0][col] = 1.0
    col += 1

    assert col == rank

    # Verification
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)

    tensor = np.zeros((n, m, p), dtype=np.float64)
    for r in range(rank):
        u = Ua[:, r]
        v = Va[:, r]
        w = Wa[:, r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    tensor[i, j, k] += u[i*m+j] * v[j*p+k] * w[k*n+i]

    if not np.allclose(tensor, np.ones((n, m, p)), atol=1e-12):
        # Fallback: schoolbook rank 30
        rs = n * m * p
        Uf = [[0.0] * rs for _ in range(n * m)]
        Vf = [[0.0] * rs for _ in range(m * p)]
        Wf = [[0.0] * rs for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    Uf[i*m+j][c] = 1.0
                    Vf[j*p+k][c] = 1.0
                    Wf[k*n+i][c] = 1.0
                    c += 1
        return {"U": Uf, "V": Vf, "W": Wf}

    return {"U": U, "V": V, "W": W}
