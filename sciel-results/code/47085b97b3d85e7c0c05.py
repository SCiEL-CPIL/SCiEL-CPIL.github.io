"""Combined program: Strassen for 2x2 blocks + schoolbook for 1x4 strips."""
import time

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x4 * 4x8 matrix multiplication.
    
    Strategy:
    - Partition A (3x4) into A1 (2x4) and A2 (1x4)
    - Partition B (4x8) into B1 (4x4) and B2 (4x4)
    - For A1*B1: use Strassen on 2x2 blocks (4 blocks, each rank 7 = 28 total)
    - For A1*B2: use Strassen on 2x2 blocks (4 blocks, each rank 7 = 28 total)
    - For A2*B1: schoolbook 1x4*4x4, rank 16
    - For A2*B2: schoolbook 1x4*4x4, rank 16
    Total rank: 28 + 28 + 16 + 16 = 88
    """
    start_time = time.time()
    deadline = 120.0  # seconds
    
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    terms = []  # List of (u_vector, v_vector, w_vector)
    
    def add_strassen_block(i0, j0, k0):
        """
        Add Strassen decomposition for A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2]
        """
        a11_u = (i0) * m + j0
        a12_u = (i0) * m + (j0 + 1)
        a21_u = (i0 + 1) * m + j0
        a22_u = (i0 + 1) * m + (j0 + 1)
        
        b11_v = j0 * p + k0
        b12_v = j0 * p + (k0 + 1)
        b21_v = (j0 + 1) * p + k0
        b22_v = (j0 + 1) * p + (k0 + 1)
        
        c11_w = k0 * n + i0
        c12_w = (k0 + 1) * n + i0
        c21_w = k0 * n + (i0 + 1)
        c22_w = (k0 + 1) * n + (i0 + 1)
        
        # M1 = (a11 + a22) * (b11 + b22)
        u1 = [0.0] * (n * m); u1[a11_u] = 1.0; u1[a22_u] = 1.0
        v1 = [0.0] * (m * p); v1[b11_v] = 1.0; v1[b22_v] = 1.0
        w1 = [0.0] * (p * n); w1[c11_w] = 1.0; w1[c22_w] = 1.0
        terms.append((u1, v1, w1))
        
        # M2 = (a21 + a22) * b11
        u2 = [0.0] * (n * m); u2[a21_u] = 1.0; u2[a22_u] = 1.0
        v2 = [0.0] * (m * p); v2[b11_v] = 1.0
        w2 = [0.0] * (p * n); w2[c21_w] = 1.0; w2[c22_w] = -1.0
        terms.append((u2, v2, w2))
        
        # M3 = a11 * (b12 - b22)
        u3 = [0.0] * (n * m); u3[a11_u] = 1.0
        v3 = [0.0] * (m * p); v3[b12_v] = 1.0; v3[b22_v] = -1.0
        w3 = [0.0] * (p * n); w3[c12_w] = 1.0; w3[c22_w] = 1.0
        terms.append((u3, v3, w3))
        
        # M4 = a22 * (b21 - b11)
        u4 = [0.0] * (n * m); u4[a22_u] = 1.0
        v4 = [0.0] * (m * p); v4[b21_v] = 1.0; v4[b11_v] = -1.0
        w4 = [0.0] * (p * n); w4[c11_w] = 1.0; w4[c21_w] = 1.0
        terms.append((u4, v4, w4))
        
        # M5 = (a11 + a12) * b22
        u5 = [0.0] * (n * m); u5[a11_u] = 1.0; u5[a12_u] = 1.0
        v5 = [0.0] * (m * p); v5[b22_v] = 1.0
        w5 = [0.0] * (p * n); w5[c11_w] = -1.0; w5[c12_w] = 1.0
        terms.append((u5, v5, w5))
        
        # M6 = (a21 - a11) * (b11 + b12)
        u6 = [0.0] * (n * m); u6[a21_u] = 1.0; u6[a11_u] = -1.0
        v6 = [0.0] * (m * p); v6[b11_v] = 1.0; v6[b12_v] = 1.0
        w6 = [0.0] * (p * n); w6[c22_w] = 1.0
        terms.append((u6, v6, w6))
        
        # M7 = (a12 - a22) * (b21 + b22)
        u7 = [0.0] * (n * m); u7[a12_u] = 1.0; u7[a22_u] = -1.0
        v7 = [0.0] * (m * p); v7[b21_v] = 1.0; v7[b22_v] = 1.0
        w7 = [0.0] * (p * n); w7[c11_w] = 1.0
        terms.append((u7, v7, w7))
    
    def add_schoolbook_1x4_4x4(i_row, j0, k0):
        """
        Add schoolbook decomposition for A[i_row, j0:j0+4] * B[j0:j0+4, k0:k0+4]
        """
        for j in range(4):
            for k in range(4):
                u_idx = i_row * m + (j0 + j)
                v_idx = (j0 + j) * p + (k0 + k)
                w_idx = (k0 + k) * n + i_row
                u_vec = [0.0] * (n * m); u_vec[u_idx] = 1.0
                v_vec = [0.0] * (m * p); v_vec[v_idx] = 1.0
                w_vec = [0.0] * (p * n); w_vec[w_idx] = 1.0
                terms.append((u_vec, v_vec, w_vec))
    
    if time.time() - start_time > deadline - 10:
        return _schoolbook_fallback(n, m, p)
    
    # A1*B1: 2x4 * 4x4
    add_strassen_block(i0=0, j0=0, k0=0)
    add_strassen_block(i0=0, j0=2, k0=0)
    add_strassen_block(i0=0, j0=0, k0=2)
    add_strassen_block(i0=0, j0=2, k0=2)
    
    if time.time() - start_time > deadline - 10:
        return _schoolbook_fallback(n, m, p)
    
    # A1*B2: 2x4 * 4x4 (B2 is columns 4-7 of B)
    add_strassen_block(i0=0, j0=0, k0=4)
    add_strassen_block(i0=0, j0=2, k0=4)
    add_strassen_block(i0=0, j0=0, k0=6)
    add_strassen_block(i0=0, j0=2, k0=6)
    
    if time.time() - start_time > deadline - 10:
        return _schoolbook_fallback(n, m, p)
    
    # A2*B1: 1x4 * 4x4, schoolbook, rank 16
    add_schoolbook_1x4_4x4(i_row=2, j0=0, k0=0)
    
    # A2*B2: 1x4 * 4x4, schoolbook, rank 16
    add_schoolbook_1x4_4x4(i_row=2, j0=0, k0=4)
    
    if time.time() - start_time > deadline - 10:
        return _schoolbook_fallback(n, m, p)
    
    # Build U, V, W matrices from terms
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u, v, w) in enumerate(terms):
        for i in range(n * m):
            if u[i] != 0:
                U[i][r] = u[i]
        for j in range(m * p):
            if v[j] != 0:
                V[j][r] = v[j]
        for k in range(p * n):
            if w[k] != 0:
                W[k][r] = w[k]
    
    return {"U": U, "V": V, "W": W}

def _schoolbook_fallback(n, m, p):
    """Fallback to schoolbook decomposition with rank = n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
