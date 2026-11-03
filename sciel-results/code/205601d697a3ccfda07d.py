import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """Construct a rank-minimal (or lower-rank) tensor decomposition for 4x4 * 4x7 matrix multiplication."""
    n, m, p = 4, 4, 7
    deadline = time.time() + 105
    
    def verify(U, V, W, n, m, p, tol=1e-7):
        U = np.asarray(U, dtype=np.float64)
        V = np.asarray(V, dtype=np.float64)
        W = np.asarray(W, dtype=np.float64)
        rank = U.shape[1]
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        return np.allclose(T, 1.0, atol=tol)

    def schoolbook_full(n, m, p):
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def strassen_2x2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
        U[2,1]=1; U[3,1]=1; V[0,1]=1; W[2,1]=-1; W[3,1]=1
        U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[1,2]=1; W[3,2]=1
        U[3,3]=1; V[2,3]=1; V[0,3]=-1; W[0,3]=-1; W[2,3]=1
        U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[1,4]=1
        U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[3,5]=-1
        U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
        return U, V, W

    def kronecker_strassen_4x4x4():
        U2, V2, W2 = strassen_2x2x2()
        A_bc = [[1,0,0,1],[0,0,1,1],[1,0,0,0],[0,0,0,1],[1,1,0,0],[-1,0,1,0],[0,1,0,-1]]
        B_bc = [[1,0,0,1],[1,0,0,0],[0,1,0,-1],[0,0,1,-1],[0,0,0,1],[1,1,0,0],[0,0,1,1]]
        C_oc = [[1,0,0,0,-1,0,1],[0,0,1,0,1,0,0],[0,1,0,1,0,0,0],[1,-1,1,0,0,-1,0]]
        
        rank = 49
        U4 = np.zeros((16, rank))
        V4 = np.zeros((16, rank))
        W4 = np.zeros((16, rank))
        
        for r in range(7):
            for s in range(7):
                col = r * 7 + s
                for ab in range(4):
                    if A_bc[r][ab] == 0: continue
                    ab_r, ab_c = divmod(ab, 2)
                    for lr in range(2):
                        for lc in range(2):
                            idx_a = (ab_r*2+lr)*4 + (ab_c*2+lc)
                            local_i = lr*2+lc
                            U4[idx_a, col] += A_bc[r][ab] * U2[local_i, s]
                for bb in range(4):
                    if B_bc[r][bb] == 0: continue
                    bb_r, bb_c = divmod(bb, 2)
                    for lr in range(2):
                        for lc in range(2):
                            idx_b = (bb_r*2+lr)*4 + (bb_c*2+lc)
                            local_j = lr*2+lc
                            V4[idx_b, col] += B_bc[r][bb] * V2[local_j, s]
                for cb in range(4):
                    if C_oc[cb][r] == 0: continue
                    cb_r, cb_c = divmod(cb, 2)
                    for lr in range(2):
                        for lc in range(2):
                            idx_c = (cb_c*2+lc)*4 + (cb_r*2+lr)
                            local_k = lr*2+lc
                            W4[idx_c, col] += C_oc[cb][r] * W2[local_k, s]
        return U4, V4, W4

    def schoolbook_4x4x3():
        rank = 4 * 4 * 3
        U = np.zeros((16, rank))
        V = np.zeros((12, rank))
        W = np.zeros((12, rank))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(3):
                    U[i * 4 + j, col] = 1
                    V[j * 3 + k, col] = 1
                    W[k * 4 + i, col] = 1
                    col += 1
        return U, V, W

    def combine(U1, V1, W1, U2, V2, W2):
        r1 = U1.shape[1]
        r2 = U2.shape[1]
        U = np.zeros((16, r1+r2))
        V = np.zeros((28, r1+r2))
        W = np.zeros((28, r1+r2))
        U[:, :r1] = U1
        V[:16, :r1] = V1
        W[:16, :r1] = W1
        U[:, r1:] = U2
        V[16:, r1:] = V2
        W[16:, r1:] = W2
        return U, V, W

    try:
        U44, V44, W44 = kronecker_strassen_4x4x4()
        U43, V43, W43 = schoolbook_4x4x3()
        U, V, W = combine(U44, V44, W44, U43, V43, W43)
        
        if not verify(U, V, W, 4, 4, 7):
             # Fallback
             U, V, W = schoolbook_full(4, 4, 7)
             
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except:
        U, V, W = schoolbook_full(4, 4, 7)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
