import numpy as np
import time

def construct():
    _t0 = time.time()
    _deadline = _t0 + 115.0  # safety margin before 120s limit

    n, m, p = 2, 3, 8

    # Rank-44 construction: 4 blocks of 3x2, each using Strassen(7) + 4 extras
    rank_per_block = 11
    num_blocks = p // 2  # 4 blocks
    total_rank = rank_per_block * num_blocks  # 44

    U = np.zeros((n * m, total_rank))  # (6, 44)
    V = np.zeros((m * p, total_rank))  # (24, 44)
    W = np.zeros((p * n, total_rank))  # (16, 44)

    for blk in range(num_blocks):
        base = blk * rank_per_block
        # For block blk, B columns are 2*blk and 2*blk+1
        # C columns (output) are 2*blk and 2*blk+1

        # Strassen for 2x2 core (A[:,0:2] * B[0:2,:]):
        # m1 = (A00+A11)(B00+B11)
        U[0, base + 0] = 1   # A[0,0]
        U[4, base + 0] = 1   # A[1,1]
        V[2 * blk, base + 0] = 1           # B[0, 2*blk]
        V[p + 2 * blk + 1, base + 0] = 1   # B[1, 2*blk+1]
        W[4 * blk, base + 0] = 1           # C[0, 2*blk]
        W[4 * blk + 3, base + 0] = 1       # C[1, 2*blk+1]

        # m2 = (A10+A11)B00
        U[3, base + 1] = 1   # A[1,0]
        U[4, base + 1] = 1   # A[1,1]
        V[2 * blk, base + 1] = 1           # B[0, 2*blk]
        W[4 * blk + 1, base + 1] = 1       # C[1, 2*blk]
        W[4 * blk + 3, base + 1] = -1      # C[1, 2*blk+1]

        # m3 = A00(B01-B11)
        U[0, base + 2] = 1   # A[0,0]
        V[2 * blk + 1, base + 2] = 1      # B[0, 2*blk+1]
        V[p + 2 * blk + 1, base + 2] = -1 # B[1, 2*blk+1]
        W[4 * blk + 2, base + 2] = 1      # C[0, 2*blk+1]
        W[4 * blk + 3, base + 2] = 1      # C[1, 2*blk+1]

        # m4 = A11(B10-B00)
        U[4, base + 3] = 1   # A[1,1]
        V[p + 2 * blk, base + 3] = 1     # B[1, 2*blk]
        V[2 * blk, base + 3] = -1        # B[0, 2*blk]
        W[4 * blk, base + 3] = 1         # C[0, 2*blk]
        W[4 * blk + 1, base + 3] = 1     # C[1, 2*blk]

        # m5 = (A00+A01)B11
        U[0, base + 4] = 1   # A[0,0]
        U[1, base + 4] = 1   # A[0,1]
        V[p + 2 * blk + 1, base + 4] = 1 # B[1, 2*blk+1]
        W[4 * blk, base + 4] = -1        # C[0, 2*blk]
        W[4 * blk + 2, base + 4] = 1     # C[0, 2*blk+1]

        # m6 = (A10-A00)(B00+B01)
        U[3, base + 5] = 1   # A[1,0]
        U[0, base + 5] = -1  # A[0,0]
        V[2 * blk, base + 5] = 1          # B[0, 2*blk]
        V[2 * blk + 1, base + 5] = 1      # B[0, 2*blk+1]
        W[4 * blk + 3, base + 5] = 1      # C[1, 2*blk+1]

        # m7 = (A01-A11)(B10+B11)
        U[1, base + 6] = 1   # A[0,1]
        U[4, base + 6] = -1  # A[1,1]
        V[p + 2 * blk, base + 6] = 1     # B[1, 2*blk]
        V[p + 2 * blk + 1, base + 6] = 1 # B[1, 2*blk+1]
        W[4 * blk, base + 6] = 1         # C[0, 2*blk]

        # Extra part: A[i,2] * B[2,k] for i in {0,1}, k in {2*blk, 2*blk+1}
        # 4 rank-1 terms
        U[2, base + 7] = 1
        V[2 * p + 2 * blk, base + 7] = 1
        W[4 * blk, base + 7] = 1

        U[2, base + 8] = 1
        V[2 * p + 2 * blk + 1, base + 8] = 1
        W[4 * blk + 2, base + 8] = 1

        U[5, base + 9] = 1
        V[2 * p + 2 * blk, base + 9] = 1
        W[4 * blk + 1, base + 9] = 1

        U[5, base + 10] = 1
        V[2 * p + 2 * blk + 1, base + 10] = 1
        W[4 * blk + 3, base + 10] = 1

    # Quick verification
    T = np.einsum('ar,br,cr->abc', U, V, W)
    target = np.zeros((6, 24, 16))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1.0
    if np.max(np.abs(T - target)) > 1e-10:
        # Fallback: schoolbook rank 48
        U = np.zeros((6, 48))
        V = np.zeros((24, 48))
        W = np.zeros((16, 48))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, c] = 1
                    V[j*p+k, c] = 1
                    W[k*n+i, c] = 1
                    c += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
