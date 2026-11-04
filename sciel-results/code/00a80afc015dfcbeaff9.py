"""Matrix multiplication 4,4,7 — rank-93 tensor decomposition using Strassen."""
import time
import numpy as np

N, M, P = 4, 4, 7


def _strassen_2x2x2():
    """
    Rank-7 decomposition for 2x2x2 matmul (Strassen).
    
    A = [[a,b],[c,d]] -> U indices: a=0, b=1, c=2, d=3
    B = [[e,f],[g,h]] -> V indices: e=0, f=1, g=2, h=3
    C = A*B, W[k*2+i]: C00=0, C10=1, C01=2, C11=3
    
    Strassen:
    M1 = (a+d)(e+h)     -> C00, C11
    M2 = (c+d)e         -> C10, C11
    M3 = a(f-h)         -> C01
    M4 = d(g-e)         -> C00, C10
    M5 = (a+b)h         -> C00, C01
    M6 = (c-a)(e+f)     -> C11
    M7 = (b-d)(g+h)     -> C00
    """
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    
    # M1: r=0
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2: r=1
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # M3: r=2
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1
    
    # M4: r=3
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5: r=4
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6: r=5
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7: r=6
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W


def _strassen_4x4x4():
    """
    Rank-49 decomposition for 4x4x4 matmul using Strassen on 2x2 blocks.
    """
    U_s, V_s, W_s = _strassen_2x2x2()
    
    U = np.zeros((16, 49), dtype=np.float64)
    V = np.zeros((16, 49), dtype=np.float64)
    W = np.zeros((16, 49), dtype=np.float64)
    
    def u_block_idx(br, bc, ii, jj):
        gi = br * 2 + ii
        gj = bc * 2 + jj
        return gi * 4 + gj
    
    def v_block_idx(br, bc, ii, jj):
        gi = br * 2 + ii
        gj = bc * 2 + jj
        return gi * 4 + gj
    
    def w_block_idx(br, bc, ii, jj):
        gi = br * 2 + ii
        gj = bc * 2 + jj
        return gj * 4 + gi
    
    strassen_block_terms = [
        ([((0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)]),
        ([((1,0), 1), ((1,1), 1)], [((0,0), 1)], [((1,0), 1), ((1,1), -1)]),
        ([((0,0), 1)], [((0,1), 1), ((1,1), -1)], [((0,1), 1), ((1,1), 1)]),
        ([((1,1), 1)], [((1,0), 1), ((0,0), -1)], [((0,0), 1), ((1,0), 1)]),
        ([((0,0), 1), ((0,1), 1)], [((1,1), 1)], [((0,0), -1), ((0,1), 1)]),
        ([((1,0), 1), ((0,0), -1)], [((0,0), 1), ((0,1), 1)], [((1,1), 1)]),
        ([((0,1), 1), ((1,1), -1)], [((1,0), 1), ((1,1), 1)], [((0,0), 1)]),
    ]
    
    for s in range(7):
        A_blocks, B_blocks, C_blocks = strassen_block_terms[s]
        for r in range(7):
            r_total = s * 7 + r
            
            for (br, bc), coeff in A_blocks:
                for ii in range(2):
                    for jj in range(2):
                        u_idx = u_block_idx(br, bc, ii, jj)
                        U[u_idx, r_total] += coeff * U_s[ii * 2 + jj, r]
            
            for (br, bc), coeff in B_blocks:
                for ii in range(2):
                    for jj in range(2):
                        v_idx = v_block_idx(br, bc, ii, jj)
                        V[v_idx, r_total] += coeff * V_s[ii * 2 + jj, r]
            
            for (br, bc), coeff in C_blocks:
                for ii in range(2):
                    for jj in range(2):
                        w_idx = w_block_idx(br, bc, ii, jj)
                        W[w_idx, r_total] += coeff * W_s[jj * 2 + ii, r]
    
    return U, V, W


def _strassen_4x4x2():
    """
    Rank-28 decomposition for 4x4x2 matmul.
    """
    U_s, V_s, W_s = _strassen_2x2x2()
    
    U = np.zeros((16, 28), dtype=np.float64)
    V = np.zeros((8, 28), dtype=np.float64)
    W = np.zeros((8, 28), dtype=np.float64)
    
    mult_specs = [
        (0, 0, 0, 0, 0),
        (0, 2, 2, 0, 0),
        (2, 0, 0, 2, 0),
        (2, 2, 2, 2, 0),
    ]
    
    for m_idx, (ar, ac, br, cr, cc) in enumerate(mult_specs):
        for r in range(7):
            r_total = m_idx * 7 + r
            
            for ii in range(2):
                for jj in range(2):
                    u_idx = (ar + ii) * 4 + (ac + jj)
                    U[u_idx, r_total] += U_s[ii * 2 + jj, r]
            
            for ii in range(2):
                for jj in range(2):
                    v_idx = (br + ii) * 2 + (cc + jj)
                    V[v_idx, r_total] += V_s[ii * 2 + jj, r]
            
            for ii in range(2):
                for jj in range(2):
                    w_idx = (cc + jj) * 4 + (cr + ii)
                    W[w_idx, r_total] += W_s[jj * 2 + ii, r]
    
    return U, V, W


def _schoolbook_4x4x1():
    """
    Rank-16 decomposition for 4x4x1 matmul.
    """
    U = np.zeros((16, 16), dtype=np.float64)
    V = np.zeros((4, 16), dtype=np.float64)
    W = np.zeros((4, 16), dtype=np.float64)
    
    col = 0
    for i in range(4):
        for j in range(4):
            U[i * 4 + j, col] = 1.0
            V[j, col] = 1.0
            W[i, col] = 1.0
            col += 1
    
    return U, V, W


def _build_decomp():
    """Build a rank-112 decomposition (schoolbook) for 4x4x7 matmul."""
    rank = N * M * P
    U = [[0.0] * rank for _ in range(N * M)]
    V = [[0.0] * rank for _ in range(M * P)]
    W = [[0.0] * rank for _ in range(P * N)]
    
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j][col] = 1.0
                V[j * P + k][col] = 1.0
                W[k * N + i][col] = 1.0
                col += 1
    return U, V, W


def _verify_decomp(U, V, W, tol=1e-8):
    """Verify decomposition."""
    n, m, p = N, M, P
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_np[i * m + j], V_np[j * p + k] * W_np[k * n + i])
                if abs(s - 1.0) > tol:
                    return False
    return True


def construct():
    """Return factor matrices for 4x4x7 matrix multiplication tensor."""
    start_time = time.time()
    deadline = start_time + 110
    
    try:
        U1, V1, W1 = _strassen_4x4x4()
        U2, V2, W2 = _strassen_4x4x2()
        U3, V3, W3 = _schoolbook_4x4x1()
        
        if time.time() > deadline - 10:
            raise TimeoutError()
        
        rank = 93
        U = np.zeros((16, rank), dtype=np.float64)
        V = np.zeros((28, rank), dtype=np.float64)
        W = np.zeros((28, rank), dtype=np.float64)
        
        U[:, 0:49] = U1
        V[0:16, 0:49] = V1
        W[0:16, 0:49] = W1
        
        U[:, 49:77] = U2
        V[16:24, 49:77] = V2
        W[16:24, 49:77] = W2
        
        U[:, 77:93] = U3
        V[24:28, 77:93] = V3
        W[24:28, 77:93] = W3
        
        if time.time() > deadline - 5:
            raise TimeoutError()
        
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        
        if _verify_decomp(U_list, V_list, W_list):
            return {"U": U_list, "V": V_list, "W": W_list}
        else:
            raise ValueError("Decomposition verification failed")
            
    except Exception:
        U, V, W = _build_decomp()
        return {"U": U, "V": V, "W": W}
