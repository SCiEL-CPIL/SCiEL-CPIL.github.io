import time
import numpy as np

DIMENSIONS = [4, 4, 6]

def _get_strassen_2x2():
    """
    Returns U, V, W matrices for 2x2x2 Strassen decomposition (rank 7).
    Indices: row = i*2 + j for U/V, W row = k*2 + i.
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1 = (a00+a11)(b00+b11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (a10+a11)b00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # M3 = a00(b01-b11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # M4 = a11(b10-b00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5 = (a00+a01)b11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6 = (a10-a00)(b00+b01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (a01-a11)(b10+b11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _get_2x2x3_rank10():
    """
    Returns U, V, W for 2x2x3 multiplication with rank 10.
    U: 4x10, V: 6x10, W: 6x10.
    """
    U1, V1, W1 = _get_strassen_2x2()
    
    U2 = np.zeros((4, 10), dtype=float)
    V2 = np.zeros((6, 10), dtype=float)
    W2 = np.zeros((6, 10), dtype=float)
    
    # Copy Strassen part (cols 0,1 of B)
    for c in range(7):
        U2[:, c] = U1[:, c]
        V2[0, c] = V1[0, c]
        V2[1, c] = V1[1, c]
        V2[3, c] = V1[2, c]
        V2[4, c] = V1[3, c]
        
        W2[0, c] = W1[0, c]
        W2[1, c] = W1[1, c]
        W2[2, c] = W1[2, c]
        W2[3, c] = W1[3, c]
        
    # Term 8: a00 b02
    U2[0, 7] = 1
    V2[2, 7] = 1 # j=0, k=2 -> 0*3+2=2
    W2[4, 7] = 1 # k=2, i=0 -> 2*2+0=4
    
    # Term 9: a11 b12
    U2[3, 8] = 1
    V2[5, 8] = 1 # j=1, k=2 -> 1*3+2=5
    W2[5, 8] = 1 # k=2, i=1 -> 2*2+1=5
    
    # Term 10: (a01+a10)(b02+b12)
    U2[1, 9] = 1
    U2[2, 9] = 1
    V2[2, 9] = 1
    V2[5, 9] = 1
    W2[4, 9] = 1
    W2[5, 9] = 1
    
    return U2, V2, W2

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verifies the decomposition using random test cases.
    """
    np.random.seed(12345)
    for _ in range(8):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for r in range(U.shape[1]):
            alpha = np.sum(A * U[:, r].reshape(n, m))
            beta = np.sum(B * V[:, r].reshape(m, p))
            if alpha != 0 and beta != 0:
                C_computed += alpha * beta * W[:, r].reshape(p, n).T
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
    return True

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    deadline = start_time + 110.0
    
    try:
        # Try Rank 70 first
        U1, V1, W1 = _get_strassen_2x2()
        U2, V2, W2 = _get_2x2x3_rank10()
        
        rank1 = U1.shape[1]
        rank2 = U2.shape[1]
        total_rank = rank1 * rank2
        
        u1_rows = np.arange(4)
        u1_i1 = u1_rows // 2
        u1_j1 = u1_rows % 2
        u2_rows = np.arange(4)
        u2_i2 = u2_rows // 2
        u2_j2 = u2_rows % 2
        u_row_map = np.zeros((4, 4), dtype=int)
        for a in range(4):
            for b in range(4):
                u_row_map[a, b] = (u1_i1[a] * 2 + u2_i2[b]) * 4 + (u1_j1[a] * 2 + u2_j2[b])

        v1_rows = np.arange(4)
        v1_j1 = v1_rows // 2
        v1_k1 = v1_rows % 2
        v2_rows = np.arange(6)
        v2_j2 = v2_rows // 3
        v2_k2 = v2_rows % 3
        v_row_map = np.zeros((4, 6), dtype=int)
        for a in range(4):
            for b in range(6):
                v_row_map[a, b] = (v1_j1[a] * 2 + v2_j2[b]) * 6 + (v1_k1[a] * 3 + v2_k2[b])

        w1_rows = np.arange(4)
        w1_k1 = w1_rows // 2
        w1_i1 = w1_rows % 2
        w2_rows = np.arange(6)
        w2_k2 = w2_rows // 2
        w2_i2 = w2_rows % 2
        w_row_map = np.zeros((4, 6), dtype=int)
        for a in range(4):
            for b in range(6):
                w_row_map[a, b] = (w1_k1[a] * 3 + w2_k2[b]) * 4 + (w1_i1[a] * 2 + w2_i2[b])

        U = np.zeros((16, total_rank), dtype=float)
        V = np.zeros((24, total_rank), dtype=float)
        W = np.zeros((24, total_rank), dtype=float)

        for c1 in range(rank1):
            col_base = c1 * rank2
            for c2 in range(rank2):
                c = col_base + c2
                u_kron = np.outer(U1[:, c1], U2[:, c2])
                U[:, c] = u_kron[u_row_map]
                v_kron = np.outer(V1[:, c1], V2[:, c2])
                V[:, c] = v_kron[v_row_map]
                w_kron = np.outer(W1[:, c1], W2[:, c2])
                W[:, c] = w_kron[w_row_map]

        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        else:
            # Fallback to Rank 77
            U2_77, V2_77, W2_77 = _get_2x2x3_hybrid()
            total_rank_77 = 7 * 11
            U = np.zeros((16, total_rank_77), dtype=float)
            V = np.zeros((24, total_rank_77), dtype=float)
            W = np.zeros((24, total_rank_77), dtype=float)
            for c1 in range(7):
                col_base = c1 * 11
                for c2 in range(11):
                    c = col_base + c2
                    u_kron = np.outer(U1[:, c1], U2_77[:, c2])
                    U[:, c] = u_kron[u_row_map]
                    v_kron = np.outer(V1[:, c1], V2_77[:, c2])
                    V[:, c] = v_kron[v_row_map]
                    w_kron = np.outer(W1[:, c1], W2_77[:, c2])
                    W[:, c] = w_kron[w_row_map]
            if _verify_decomposition(U, V, W, n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            else:
                raise ValueError("Both decompositions failed verification")

    except Exception:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _get_2x2x3_hybrid():
    """
    Returns U, V, W for 2x2x3 multiplication (rank 11).
    """
    U1, V1, W1 = _get_strassen_2x2()
    U2 = np.zeros((4, 11), dtype=float)
    V2 = np.zeros((6, 11), dtype=float)
    W2 = np.zeros((6, 11), dtype=float)
    
    for c in range(7):
        U2[:, c] = U1[:, c]
        V2[0, c] = V1[0, c]
        V2[1, c] = V1[1, c]
        V2[3, c] = V1[2, c]
        V2[4, c] = V1[3, c]
        W2[0, c] = W1[0, c]
        W2[1, c] = W1[1, c]
        W2[2, c] = W1[2, c]
        W2[3, c] = W1[3, c]
        
    U2[0, 7] = 1
    V2[2, 7] = 1
    W2[4, 7] = 1
    
    U2[1, 8] = 1
    V2[5, 8] = 1
    W2[4, 8] = 1
    
    U2[2, 9] = 1
    V2[2, 9] = 1
    W2[5, 9] = 1
    
    U2[3, 10] = 1
    V2[5, 10] = 1
    W2[5, 10] = 1
    
    return U2, V2, W2
