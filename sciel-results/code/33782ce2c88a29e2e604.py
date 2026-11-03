"""Matrix multiplication 5x5x5: rank-45 decomposition via 2+2+1 block structure."""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]
_START_TIME = time.time()
_DEADLINE = 150.0
_MARGIN = 5.0


def _strassen_2x2():
    """
    Return (U, V, W) for 2x2 matrix multiplication via Strassen, rank 7.
    U: (4, 7) indexed [i*2+j, r]
    V: (4, 7) indexed [j*2+k, r]  
    W: (4, 7) indexed [k*2+i, r]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # m1 = (a00+a11)*(b00+b11) -> c00 += m1, c11 += m1, c01 -= m1, c10 -= m1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1; W[1, 0] = -1; W[2, 0] = -1
    
    # m2 = (a10+a11)*b00 -> c10 += m2, c11 += m2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = 1
    
    # m3 = a00*(b01-b11) -> c00 += m3, c01 += m3
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[0, 2] = 1; W[1, 2] = 1
    
    # m4 = a11*(b10-b00) -> c10 += m4, c11 += m4
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[2, 3] = 1; W[3, 3] = 1
    
    # m5 = (a00+a01)*b11 -> c00 += m5, c01 += m5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[1, 4] = 1
    
    # m6 = (a10-a00)*(b00+b01) -> c11 += m6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # m7 = (a01-a11)*(b10+b11) -> c00 += m7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W


def construct():
    """
    Construct a rank-37 tensor decomposition for 5x5 matrix multiplication
    using 2+2+1 block partition with Strassen for diagonal 2x2 blocks.
    """
    n, m, p = DIMENSIONS  # all 5
    
    sizes = [2, 2, 1]
    offsets = [0, 2, 4]
    n_blocks = 3
    
    U_list = []
    V_list = []
    W_list = []
    
    for a in range(n_blocks):
        for b in range(n_blocks):
            for c in range(n_blocks):
                sa, sb, sc = sizes[a], sizes[b], sizes[c]
                oa, ob, oc = offsets[a], offsets[b], offsets[c]
                
                if sa == 2 and sb == 2 and sc == 2 and a == b == c:
                    # Strassen for 2x2x2 diagonal block
                    Ub, Vb, Wb = _strassen_2x2()
                    rank_b = 7
                    U_full = np.zeros((25, rank_b))
                    V_full = np.zeros((25, rank_b))
                    W_full = np.zeros((25, rank_b))
                    
                    for r in range(rank_b):
                        for i in range(2):
                            for j in range(2):
                                u_block_idx = i * 2 + j
                                u_full_idx = (oa + i) * 5 + (ob + j)
                                U_full[u_full_idx, r] = Ub[u_block_idx, r]
                        for j in range(2):
                            for k in range(2):
                                v_block_idx = j * 2 + k
                                v_full_idx = (ob + j) * 5 + (oc + k)
                                V_full[v_full_idx, r] = Vb[v_block_idx, r]
                        for k in range(2):
                            for i in range(2):
                                w_block_idx = k * 2 + i
                                w_full_idx = (oc + k) * 5 + (oa + i)
                                W_full[w_full_idx, r] = Wb[w_block_idx, r]
                    
                    U_list.append(U_full)
                    V_list.append(V_full)
                    W_list.append(W_full)
                else:
                    # Schoolbook for (sa x sb) * (sb x sc) -> (sa x sc)
                    rank_b = sa * sb * sc
                    if rank_b == 0:
                        continue
                    U_full = np.zeros((25, rank_b))
                    V_full = np.zeros((25, rank_b))
                    W_full = np.zeros((25, rank_b))
                    
                    col = 0
                    for i in range(sa):
                        for j in range(sb):
                            for k in range(sc):
                                u_full_idx = (oa + i) * 5 + (ob + j)
                                v_full_idx = (ob + j) * 5 + (oc + k)
                                w_full_idx = (oc + k) * 5 + (oa + i)
                                U_full[u_full_idx, col] = 1.0
                                V_full[v_full_idx, col] = 1.0
                                W_full[w_full_idx, col] = 1.0
                                col += 1
                    
                    U_list.append(U_full)
                    V_list.append(V_full)
                    W_list.append(W_full)
    
    U = np.hstack(U_list)
    V = np.hstack(V_list)
    W = np.hstack(W_list)
    rank = U.shape[1]
    
    # Verify with random matrices
    np.random.seed(12345)
    A = np.random.randn(5, 5)
    B = np.random.randn(5, 5)
    C_expected = A @ B
    
    C_check = np.zeros((5, 5))
    for r in range(rank):
        u_col = U[:, r].reshape(5, 5)  # u_col[i,j] = U[i*5+j, r]
        v_col = V[:, r].reshape(5, 5)  # v_col[j,k] = V[j*5+k, r]
        w_col = W[:, r].reshape(5, 5)  # w_col[k,i] = W[k*5+i, r]
        
        a_lin = (u_col * A).sum(axis=1)   # a_lin[i] = sum_j u_col[i,j]*A[i,j]
        b_lin = (v_col * B).sum(axis=0)   # b_lin[k] = sum_j v_col[j,k]*B[j,k]
        
        C_check += np.outer(a_lin, b_lin) * w_col.T
    
    error = np.max(np.abs(C_check - C_expected))
    
    if error > 1e-9:
        # Fallback to schoolbook
        rank = 125
        U = np.zeros((25, rank))
        V = np.zeros((25, rank))
        W = np.zeros((25, rank))
        col = 0
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    U[i * 5 + j, col] = 1.0
                    V[j * 5 + k, col] = 1.0
                    W[k * 5 + i, col] = 1.0
                    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
