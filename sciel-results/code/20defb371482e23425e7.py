"""Matrix multiplication 4,4,8 - Strassen rank-112 decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # n=4, m=4, p=8
    rank = 112
    U = [[0] * rank for _ in range(n * m)]  # (16, 112)
    V = [[0] * rank for _ in range(m * p)]  # (32, 112)
    W = [[0] * rank for _ in range(p * n)]  # (32, 112)
    
    def add_strassen_block(ar, ac, br, bc, cr, cc, col):
        """Add 7 Strassen multiplications for A_block(ar,ac) x B_block(br,bc) -> C_block(cr,cc)"""
        a11 = ar * 4 + ac
        a12 = ar * 4 + ac + 1
        a21 = (ar + 1) * 4 + ac
        a22 = (ar + 1) * 4 + ac + 1
        
        b11 = br * 8 + bc
        b12 = br * 8 + bc + 1
        b21 = (br + 1) * 8 + bc
        b22 = (br + 1) * 8 + bc + 1
        
        c11 = cc * 4 + cr
        c12 = (cc + 1) * 4 + cr
        c21 = cc * 4 + (cr + 1)
        c22 = (cc + 1) * 4 + (cr + 1)
        
        # m1 = (a11+a22)*(b11+b22) -> c11 += 1, c22 += 1
        U[a11][col] = 1; U[a22][col] = 1
        V[b11][col] = 1; V[b22][col] = 1
        W[c11][col] = 1; W[c22][col] = 1
        col += 1
        
        # m2 = (a21+a22)*b11 -> c21 += 1, c22 -= 1
        U[a21][col] = 1; U[a22][col] = 1
        V[b11][col] = 1
        W[c21][col] = 1; W[c22][col] = -1
        col += 1
        
        # m3 = a11*(b12-b22) -> c12 += 1, c22 += 1
        U[a11][col] = 1
        V[b12][col] = 1; V[b22][col] = -1
        W[c12][col] = 1; W[c22][col] = 1
        col += 1
        
        # m4 = a22*(b21-b11) -> c11 += 1, c21 += 1
        U[a22][col] = 1
        V[b21][col] = 1; V[b11][col] = -1
        W[c11][col] = 1; W[c21][col] = 1
        col += 1
        
        # m5 = (a11+a12)*b22 -> c11 -= 1, c12 += 1
        U[a11][col] = 1; U[a12][col] = 1
        V[b22][col] = 1
        W[c11][col] = -1; W[c12][col] = 1
        col += 1
        
        # m6 = (a21-a11)*(b11+b12) -> c22 += 1
        U[a21][col] = 1; U[a11][col] = -1
        V[b11][col] = 1; V[b12][col] = 1
        W[c22][col] = 1
        col += 1
        
        # m7 = (a12-a22)*(b21+b22) -> c11 += 1
        U[a12][col] = 1; U[a22][col] = -1
        V[b21][col] = 1; V[b22][col] = 1
        W[c11][col] = 1
        col += 1
        
        return col
    
    # 16 block multiplications: each 2x2 A block times each 2x2 B block
    blocks = [
        (0, 0, 0, 0, 0, 0),  # A11*B11 -> C11
        (0, 0, 0, 2, 0, 2),  # A11*B12 -> C12
        (0, 0, 0, 4, 0, 4),  # A11*B13 -> C13
        (0, 0, 0, 6, 0, 6),  # A11*B14 -> C14
        (0, 2, 2, 0, 0, 0),  # A12*B21 -> C11
        (0, 2, 2, 2, 0, 2),  # A12*B22 -> C12
        (0, 2, 2, 4, 0, 4),  # A12*B23 -> C13
        (0, 2, 2, 6, 0, 6),  # A12*B24 -> C14
        (2, 0, 0, 0, 2, 0),  # A21*B11 -> C21
        (2, 0, 0, 2, 2, 2),  # A21*B12 -> C22
        (2, 0, 0, 4, 2, 4),  # A21*B13 -> C23
        (2, 0, 0, 6, 2, 6),  # A21*B14 -> C24
        (2, 2, 2, 0, 2, 0),  # A22*B21 -> C21
        (2, 2, 2, 2, 2, 2),  # A22*B22 -> C22
        (2, 2, 2, 4, 2, 4),  # A22*B23 -> C23
        (2, 2, 2, 6, 2, 6),  # A22*B24 -> C24
    ]
    
    col = 0
    for ar, ac, br, bc, cr, cc in blocks:
        col = add_strassen_block(ar, ac, br, bc, cr, cc, col)
    
    # Verify correctness with numpy (with time guard)
    if time.time() - start_time < 110:
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        rng = np.random.default_rng(42)
        A_test = rng.standard_normal((n, m))
        B_test = rng.standard_normal((m, p))
        C_expected = A_test @ B_test
        
        C_decomp = np.zeros((n, p))
        for r in range(rank):
            u_vec = U_np[:, r]
            v_vec = V_np[:, r]
            w_vec = W_np[:, r]
            U_r = u_vec.reshape(n, m)
            V_r = v_vec.reshape(m, p)
            W_r = w_vec.reshape(p, n)
            C_decomp += (U_r @ V_r) * W_r.T
        
        if not np.allclose(C_decomp, C_expected, atol=1e-10):
            # Fall back to schoolbook
            rank = 128
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            c = 0
            for i in range(n):
                for j in range(m):
                    for l in range(p):
                        U[i * m + j][c] = 1
                        V[j * p + l][c] = 1
                        W[l * n + i][c] = 1
                        c += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
