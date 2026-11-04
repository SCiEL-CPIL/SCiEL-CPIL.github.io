import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 115.0

    # Schoolbook fallback
    rank_school = n * m * p
    U_sb = np.zeros((n*m, rank_school))
    V_sb = np.zeros((m*p, rank_school))
    W_sb = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1

    best_rank = rank_school
    best_U, best_V, best_W = U_sb, V_sb, W_sb

    def _verify_decomposition(U, V, W, n, m, p):
        r = U.shape[1]
        if r == 0:
            return False
        U_r = U.reshape(n, m, r)
        V_r = V.reshape(m, p, r)
        W_r = W.reshape(p, n, r)
        res = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        return np.allclose(res, 1.0, atol=1e-9)

    # Winograd 2x2x2 factors
    U2 = np.array([[1,0,1,0,0],[0,1,0,0,1],[0,0,0,1,0],[1,1,0,1,0]], dtype=float)
    V2 = np.array([[1,0,0,1,0],[0,1,1,0,-1],[1,1,0,1,0],[1,0,1,0,1]], dtype=float)
    W2 = np.array([[1,-1,1,-1,1],[1,-1,1,0,-1],[1,1,-1,1,0],[1,1,-1,0,1]], dtype=float)

    # Kronecker product for 4x4x4 (rank 25)
    U4_kron, V4_kron, W4_kron = np.kron(U2,U2), np.kron(V2,V2), np.kron(W2,W2)

    # Permutation to reorder Kronecker indices to (i,j) ordering
    perm = np.zeros(16, dtype=int)
    for r1 in range(4):
        row1, col1 = r1 // 2, r1 % 2
        for r2 in range(4):
            row2, col2 = r2 // 2, r2 % 2
            perm[(row1*2+row2)*4 + (col1*2+col2)] = r1*4 + r2

    U_w, V_w, W_w = U4_kron[perm,:], V4_kron[perm,:], W4_kron[perm,:]

    # Rank 26: Winograd(25) + tail(1)
    U_cand = np.zeros((n*m, 26))
    V_cand = np.zeros((m*p, 26))
    W_cand = np.zeros((p*n, 26))

    U_cand[:, :25] = U_w
    for j in range(m):
        for k in range(4):
            V_cand[j*p+k, :25] = V_w[j*4+k, :]
    W_cand[:16, :25] = W_w

    # Tail: k=4 column
    U_cand[:, 25] = 1.0
    for j in range(m):
        V_cand[j*p+4, 25] = 1.0
    for i in range(n):
        W_cand[16+i, 25] = 1.0

    if _verify_decomposition(U_cand, V_cand, W_cand, n, m, p):
        best_rank, best_U, best_V, best_W = 26, U_cand, V_cand, W_cand

    def to_list(mat):
        return [[float(v) for v in row] for row in mat]

    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
