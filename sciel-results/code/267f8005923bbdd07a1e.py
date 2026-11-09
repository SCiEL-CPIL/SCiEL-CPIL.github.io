import time
import numpy as np

def construct():
    start = time.time()
    deadline = start + 110  # safety margin before 120s

    n, m, p = 2, 4, 5

    # Strassen factors for 2x2x2 (indices i,j,k in {0,1})
    SU = np.array([
        [1,0,0,1],[0,0,1,1],[1,0,0,0],[0,0,0,1],
        [1,1,0,0],[-1,0,1,0],[0,1,0,-1]
    ], dtype=float)  # 7 x 4 (i*2+j)
    SV = np.array([
        [1,0,0,1],[1,0,0,0],[0,1,0,-1],[-1,0,1,0],
        [0,0,0,1],[1,1,0,0],[0,0,1,1]
    ], dtype=float)  # 7 x 4 (j*2+k)
    SW = np.array([
        [1,0,0,1],[0,1,0,-1],[0,0,1,1],[1,1,0,0],
        [-1,0,1,0],[0,0,0,1],[1,0,0,0]
    ], dtype=float)  # 7 x 4 (k*2+i)

    rank = 4 * 7 + 8
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    col = 0
    # 4 Strassen blocks for the k in {0,1,2,3} portion (4 columns of B)
    for k0 in (0, 2):
        for r in range(7):
            # U: 2x2 A-block (rows 0-1, cols 0-3 -> all of m)
            U[0:m, col] = SU[r].reshape(2, 2).flatten()
            # V: 2x2 B-block (rows 0-1, cols k0..k0+1)
            V[0:2, k0*1 + k0*0:0]  # placeholder, filled below
            # place V entries explicitly
            for lj in range(2):
                for lk in range(2):
                    V[lj * p + (k0 + lk), col] = SV[r, lj*2 + lk]
            # W: 2x2 C-block (rows k0..k0+1, cols 0-1)
            for lk in range(2):
                for li in range(2):
                    W[(k0 + lk) * n + li, col] = SW[r, lk*2 + li]
            col += 1

    # Schoolbook for the last column of B (k=4)
    for i in range(n):
        for j in range(m):
            U[i * m + j, col] = 1.0
            V[j * p + 4, col] = 1.0
            W[4 * n + i, col] = 1.0
            col += 1

    # Round to integers (values are already integer)
    U = U.astype(int)
    V = V.astype(int)
    W = W.astype(int)

    # Verify exactness with random integer matrices (cheap, well within budget)
    if time.time() < deadline:
        rng = np.random.default_rng(12345)
        A = rng.integers(-3, 4, size=(n, m))
        B = rng.integers(-3, 4, size=(m, p))
        C_true = A @ B
        C_rec = np.zeros((n, p))
        for r in range(rank):
            C_rec += (A @ V[:, r]) * W[:, r]
        if not np.array_equal(C_rec, C_true):
            # Fallback: pure schoolbook (rank 40) guaranteed correct
            U = np.zeros((n*m, n*m*p), dtype=int)
            V = np.zeros((m*p, n*m*p), dtype=int)
            W = np.zeros((p*n, n*m*p), dtype=int)
            c = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i*m+j, c] = 1
                        V[j*p+k, c] = 1
                        W[k*n+i, c] = 1
                        c += 1
            return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
