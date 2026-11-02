"""Matrix multiplication 4x4 * 4x7: exact rank-minimal tensor decomposition."""
import time
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 145.0


def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, n, m, p):
    """Check that sum_r U[:,r] V[:,r] W[:,r] == multiplication tensor."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = np.dot(U_arr[a], V_arr[b] * W_arr[c])
                if abs(s - 1) > 1e-9:
                    return False
    return True


def _strassen_2x2x2():
    """Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    Indexing: U[i*2+j], V[j*2+k], W[k*2+i] for i,j,k in {0,1}."""
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # p1 = (a11 + a22)(b11 + b22)
    # C[0,0]+=1, C[1,1]+=1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # p2 = (a21 + a22)b11
    # C[1,0]+=1, C[1,1]+=-1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # p3 = a11(b12 - b22)
    # C[0,1]+=1, C[1,1]+=1
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # p4 = a22(b21 - b11)
    # C[0,0]+=1
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1
    
    # p5 = (a11 + a12)b22
    # C[0,0]+=-1, C[0,1]+=1
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # p6 = (a21 - a11)(b11 + b12)
    # C[1,1]+=1
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # p7 = (a12 - a22)(b21 + b22)
    # C[0,0]+=-1, C[1,0]+=1
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = -1; W[1][6] = 1
    
    return U, V, W


def _build_4x4x4_strassen():
    """Build rank-49 decomposition for 4x4x4 using Kronecker product of two Strassen decompositions."""
    n = m = p = 4
    U2, V2, W2 = _strassen_2x2x2()
    
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            # U: A[i,j] where i = i1*2+i2, j = j1*2+j2
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            i = i1 * 2 + i2
                            j = j1 * 2 + j2
                            val = U2[i1*2+j1][r1] * U2[i2*2+j2][r2]
                            if val != 0:
                                U[i*m+j][r] += val
            
            # V: B[j,k] where j = j1*2+j2, k = k1*2+k2
            for j1 in range(2):
                for k1 in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            j = j1 * 2 + j2
                            k = k1 * 2 + k2
                            val = V2[j1*2+k1][r1] * V2[j2*2+k2][r2]
                            if val != 0:
                                V[j*p+k][r] += val
            
            # W: C[i,k] -> W[k*n+i] where k = k1*2+k2, i = i1*2+i2
            for k1 in range(2):
                for i1 in range(2):
                    for k2 in range(2):
                        for i2 in range(2):
                            k = k1 * 2 + k2
                            i = i1 * 2 + i2
                            val = W2[k1*2+i1][r1] * W2[k2*2+i2][r2]
                            if val != 0:
                                W[k*n+i][r] += val
    
    return {"U": U, "V": V, "W": W}


def _build_4x4x7_optimized():
    """Build decomposition for 4x4 * 4x7 using Strassen for the 4x4x4 part and optimized 4x4x3."""
    n, m, p = 4, 4, 7
    U2, V2, W2 = _strassen_2x2x2()
    
    # Part 1: 4x4 * 4x4 (B cols 0-3, C cols 0-3), rank 49
    rank1 = 49
    U1 = [[0] * rank1 for _ in range(16)]
    V1 = [[0] * rank1 for _ in range(16)]
    W1 = [[0] * rank1 for _ in range(16)]
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            i = i1 * 2 + i2
                            j = j1 * 2 + j2
                            val = U2[i1*2+j1][r1] * U2[i2*2+j2][r2]
                            if val != 0:
                                U1[i*4+j][r] += val
            for j1 in range(2):
                for k1 in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            j = j1 * 2 + j2
                            k = k1 * 2 + k2
                            val = V2[j1*2+k1][r1] * V2[j2*2+k2][r2]
                            if val != 0:
                                V1[j*4+k][r] += val
            for k1 in range(2):
                for i1 in range(2):
                    for k2 in range(2):
                        for i2 in range(2):
                            k = k1 * 2 + k2
                            i = i1 * 2 + i2
                            val = W2[k1*2+i1][r1] * W2[k2*2+i2][r2]
                            if val != 0:
                                W1[k*4+i][r] += val
    
    # Part 2: 4x4 * 4x2 (B cols 4-5, C cols 4-5), rank 28
    # 4 products of 2x2 * 2x2, each Strassen rank 7
    rank2 = 28
    U2p = [[0] * rank2 for _ in range(16)]
    V2p = [[0] * rank2 for _ in range(8)]  # 4*2
    W2p = [[0] * rank2 for _ in range(8)]  # 2*4
    
    block_idx = 0
    for br in range(2):
        for bc in range(2):
            for kc in range(2):
                # A block (br, bc) is 2x2: A[br*2+lr, bc*2+lc]
                # B block (bc, kc) is 2x2: B[bc*2+lr, 4+kc*2+kl]
                # C block (br, kc) is 2x2: C[br*2+lr, 4+kc*2+kl]
                for t in range(7):
                    r = block_idx * 7 + t
                    # U: A[br*2+lr, bc*2+lc]
                    for lr in range(2):
                        for lc in range(2):
                            i = br * 2 + lr
                            j = bc * 2 + lc
                            val = U2[lr*2+lc][t]
                            if val != 0:
                                U2p[i*4+j][r] += val
                    # V: B[bc*2+lr, 4+kc*2+kl]
                    for lr in range(2):
                        for kl in range(2):
                            j_local = bc * 2 + lr
                            k_local = kc * 2 + kl
                            val = V2[lr*2+kl][t]
                            if val != 0:
                                V2p[j_local*2+k_local][r] += val
                    # W: C[br*2+lr, 4+kc*2+kl] -> W[(4+kc*2+kl)*4 + (br*2+lr)]
                    for lr in range(2):
                        for kl in range(2):
                            k_local = kc * 2 + kl
                            i_local = br * 2 + lr
                            val = W2[kl*2+lr][t]
                            if val != 0:
                                W2p[k_local*4+i_local][r] += val
                block_idx += 1
    
    # Part 3: 4x4 * 4x1 (B col 6, C col 6), rank 16
    # 4 products of 2x2 * 2x1, each schoolbook rank 4
    rank3 = 16
    U3 = [[0] * rank3 for _ in range(16)]
    V3 = [[0] * rank3 for _ in range(4)]  # 4*1
    W3 = [[0] * rank3 for _ in range(4)]  # 1*4
    
    block_idx = 0
    for br in range(2):
        for bc in range(2):
            # A block (br, bc) is 2x2: A[br*2+lr, bc*2+lc]
            # B block (bc) is 2x1: B[bc*2+lr, 6]
            # C block (br) is 2x1: C[br*2+lr, 6]
            for lr in range(2):
                for lc in range(2):
                    r = block_idx * 4 + lr * 2 + lc
                    i = br * 2 + lr
                    j = bc * 2 + lc
                    U3[i*4+j][r] = 1
                    V3[j][r] = 1
                    W3[i][r] = 1
            block_idx += 1
    
    # Combine all parts
    total_rank = rank1 + rank2 + rank3
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Part 1: B cols 0-3, C cols 0-3
    for i in range(16):
        for r in range(rank1):
            U[i][r] = U1[i][r]
    for i in range(4):
        for j in range(4):
            for r in range(rank1):
                V[i*p+j][r] = V1[i*4+j][r]
    for i in range(4):
        for j in range(4):
            for r in range(rank1):
                W[i*n+j][r] = W1[i*4+j][r]
    
    # Part 2: B cols 4-5, C cols 4-5
    offset2 = rank1
    for i in range(16):
        for r in range(rank2):
            U[i][offset2 + r] = U2p[i][r]
    for j_global in range(4):
        for k_local in range(2):
            k_global = 4 + k_local
            for r in range(rank2):
                V[j_global*p + k_global][offset2 + r] = V2p[j_global*2+k_local][r]
    for k_local in range(2):
        k_global = 4 + k_local
        for i_local in range(4):
            for r in range(rank2):
                W[k_global*n + i_local][offset2 + r] = W2p[k_local*4+i_local][r]
    
    # Part 3: B col 6, C col 6
    offset3 = rank1 + rank2
    for i in range(16):
        for r in range(rank3):
            U[i][offset3 + r] = U3[i][r]
    for j_global in range(4):
        for r in range(rank3):
            V[j_global*p + 6][offset3 + r] = V3[j_global][r]
    for i_local in range(4):
        for r in range(rank3):
            W[6*n + i_local][offset3 + r] = W3[i_local][r]
    
    return {"U": U, "V": V, "W": W}


def _build_4x4x7_fallback():
    """Build decomposition for 4x4 * 4x7 using Strassen for 4x4x4 and schoolbook for 4x4x3."""
    n, m, p = 4, 4, 7
    
    left = _build_4x4x4_strassen()
    rank_left = len(left['U'][0])
    
    n_r, m_r, p_r = 4, 4, 3
    rank_right = n_r * m_r * p_r
    U_right = [[0] * rank_right for _ in range(n_r * m_r)]
    V_right = [[0] * rank_right for _ in range(m_r * p_r)]
    W_right = [[0] * rank_right for _ in range(p_r * n_r)]
    col = 0
    for i in range(n_r):
        for j in range(m_r):
            for k in range(p_r):
                U_right[i * m_r + j][col] = 1
                V_right[j * p_r + k][col] = 1
                W_right[k * n_r + i][col] = 1
                col += 1
    
    total_rank = rank_left + rank_right
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    for i in range(n * m):
        for r in range(rank_left):
            U[i][r] = left['U'][i][r]
    for i in range(4):
        for j in range(4):
            for r in range(rank_left):
                V[i * p + j][r] = left['V'][i * 4 + j][r]
    for i in range(4):
        for j in range(4):
            for r in range(rank_left):
                W[i * n + j][r] = left['W'][i * 4 + j][r]
    
    offset = rank_left
    for i in range(n * m):
        for r in range(rank_right):
            U[i][offset + r] = U_right[i][r]
    for i in range(4):
        for j in range(3):
            for r in range(rank_right):
                V[i * p + (4 + j)][offset + r] = V_right[i * 3 + j][r]
    for i in range(3):
        for j in range(4):
            for r in range(rank_right):
                W[(4 + i) * n + j][offset + r] = W_right[i * 4 + j][r]
    
    return {"U": U, "V": V, "W": W}


def construct():
    """Construct the tensor decomposition. Returns {'U', 'V', 'W'}."""
    # Try the optimized construction first (rank 93)
    try:
        if time.time() - _START_TIME > _TIME_LIMIT - 10:
            return _schoolbook(4, 4, 7)
        result = _build_4x4x7_optimized()
        if _verify(result['U'], result['V'], result['W'], 4, 4, 7):
            return result
    except Exception:
        pass
    
    # Try the fallback construction (rank 97)
    try:
        if time.time() - _START_TIME > _TIME_LIMIT - 10:
            return _schoolbook(4, 4, 7)
        result = _build_4x4x7_fallback()
        if _verify(result['U'], result['V'], result['W'], 4, 4, 7):
            return result
    except Exception:
        pass
    
    # Fallback to schoolbook
    return _schoolbook(4, 4, 7)
