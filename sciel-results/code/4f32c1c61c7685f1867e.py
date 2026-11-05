import numpy as np
import time

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Exact rank-56 tensor decomposition for 3x4x5 matrix multiplication.
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1.
    
    Strategy: Split A into A1 (rows 0,1) and A2 (row 2).
    - A1 part (2x4x5): Split into two 2x2x5 blocks, each using Strassen(7)+Strassen(7)+schoolbook(4) = 18
    - A2 part (1x4x5): Split into two 1x2x5 blocks, each using schoolbook(4)+schoolbook(4)+schoolbook(2) = 10
    Total: 2*18 + 2*10 = 56
    """
    n, m, p = DIMENSIONS
    N1, N2, N3 = n * m, m * p, p * n
    start_time = time.time()
    deadline = start_time + 130

    U_parts = []
    V_parts = []
    W_parts = []

    def add_block(u_idx, v_idx, w_idx, U_blk, V_blk, W_blk):
        for r in range(U_blk.shape[1]):
            u = np.zeros(N1)
            v = np.zeros(N2)
            w = np.zeros(N3)
            for i, idx in enumerate(u_idx):
                u[idx] = U_blk[i, r]
            for i, idx in enumerate(v_idx):
                v[idx] = V_blk[i, r]
            for i, idx in enumerate(w_idx):
                w[idx] = W_blk[i, r]
            U_parts.append(u)
            V_parts.append(v)
            W_parts.append(w)

    def strassen_2x2x2():
        U = np.zeros((4, 7)); V = np.zeros((4, 7)); W = np.zeros((4, 7))
        U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
        U[2,1]=1; U[3,1]=1; V[0,1]=1; W[2,1]=1; W[3,1]=-1
        U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[1,2]=1; W[3,2]=1
        U[3,3]=1; V[2,3]=1; V[0,3]=-1; W[0,3]=1; W[2,3]=1
        U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[1,4]=1
        U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[3,5]=1
        U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
        return U, V, W

    def schoolbook_2x2x1():
        U = np.zeros((4,4)); V = np.zeros((2,4)); W = np.zeros((2,4))
        U[0,0]=1; V[0,0]=1; W[0,0]=1
        U[1,1]=1; V[1,1]=1; W[0,1]=1
        U[2,2]=1; V[0,2]=1; W[1,2]=1
        U[3,3]=1; V[1,3]=1; W[1,3]=1
        return U, V, W

    def schoolbook_1x2x2():
        U = np.zeros((2,4)); V = np.zeros((4,4)); W = np.zeros((2,4))
        U[0,0]=1; V[0,0]=1; W[0,0]=1
        U[1,1]=1; V[2,1]=1; W[0,1]=1
        U[0,2]=1; V[1,2]=1; W[1,2]=1
        U[1,3]=1; V[3,3]=1; W[1,3]=1
        return U, V, W

    def schoolbook_1x2x1():
        U = np.zeros((2,2)); V = np.zeros((2,2)); W = np.zeros((1,2))
        U[0,0]=1; V[0,0]=1; W[0,0]=1
        U[1,1]=1; V[1,1]=1; W[0,1]=1
        return U, V, W

    U_top = [0,1,4,5]
    U_bot = [2,3,6,7]
    V_tl = [0,1,5,6]
    V_tm = [2,3,7,8]
    V_tr = [4,9]
    V_bl = [10,11,15,16]
    V_bm = [12,13,17,18]
    V_br = [14,19]
    W_c01 = [0,3,1,4]
    W_c23 = [6,9,7,10]
    W_c4 = [12,13]

    blocks_p1 = [
        (U_top, V_tl, W_c01, strassen_2x2x2()),
        (U_top, V_tm, W_c23, strassen_2x2x2()),
        (U_top, V_tr, W_c4,  schoolbook_2x2x1()),
        (U_bot, V_bl, W_c01, strassen_2x2x2()),
        (U_bot, V_bm, W_c23, strassen_2x2x2()),
        (U_bot, V_br, W_c4,  schoolbook_2x2x1()),
    ]

    for u_idx, v_idx, w_idx, blk in blocks_p1:
        add_block(u_idx, v_idx, w_idx, *blk)

    if time.time() > deadline:
        rank = n * m * p
        Us = np.zeros((n*m, rank)); Vs = np.zeros((m*p, rank)); Ws = np.zeros((p*n, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    Us[i*m+j, c] = 1; Vs[j*p+k, c] = 1; Ws[k*n+i, c] = 1; c += 1
        return {"U": Us.tolist(), "V": Vs.tolist(), "W": Ws.tolist()}

    U_at = [8,9]
    U_ab = [10,11]
    W_c01_2 = [2,5]
    W_c23_2 = [8,11]
    W_c4_2 = [14]

    blocks_p2 = [
        (U_at, V_tl, W_c01_2, schoolbook_1x2x2()),
        (U_at, V_tm, W_c23_2, schoolbook_1x2x2()),
        (U_at, V_tr, W_c4_2,  schoolbook_1x2x1()),
        (U_ab, V_bl, W_c01_2, schoolbook_1x2x2()),
        (U_ab, V_bm, W_c23_2, schoolbook_1x2x2()),
        (U_ab, V_br, W_c4_2,  schoolbook_1x2x1()),
    ]

    for u_idx, v_idx, w_idx, blk in blocks_p2:
        add_block(u_idx, v_idx, w_idx, *blk)

    U = np.array(U_parts).T
    V = np.array(V_parts).T
    W = np.array(W_parts).T

    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = np.sum(W[k*n+i, :] * U[i*m+j, :] * V[j*p+k, :])

    if not np.allclose(T, np.ones((n, m, p)), atol=1e-9):
        rank = n * m * p
        Us = np.zeros((n*m, rank)); Vs = np.zeros((m*p, rank)); Ws = np.zeros((p*n, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    Us[i*m+j, c] = 1; Vs[j*p+k, c] = 1; Ws[k*n+i, c] = 1; c += 1
        return {"U": Us.tolist(), "V": Vs.tolist(), "W": Ws.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
