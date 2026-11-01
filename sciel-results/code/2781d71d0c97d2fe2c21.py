import time
import numpy as np

def construct():
    """
    Rank-42 exact decomposition for 2x4 @ 4x6 matrix multiplication.
    
    Strategy: Split the 4-dimension into two 2-dimensions.
    C = A[:,0:2] @ B[0:2,:] + A[:,2:4] @ B[2:4,:]
    Each term is 2x2 @ 2x6, which we split into three 2x2 @ 2x2 blocks.
    Each 2x2 @ 2x2 uses Strassen's rank-7 algorithm.
    Total: 2 (left/right) * 3 (blocks) * 7 (Strassen) = 42
    
    U: (n*m, rank) = (8, 42)
    V: (m*p, rank) = (24, 42)  
    W: (p*n, rank) = (12, 42)
    
    W index convention: W[k*n+i, r] where k is column (0..5), i is row (0..1)
    So W index = k*2 + i
    """
    n, m, p = 2, 4, 6
    rank = 42
    
    U = [[0] * rank for _ in range(n * m)]   # 8 x 42
    V = [[0] * rank for _ in range(m * p)]   # 24 x 42
    W = [[0] * rank for _ in range(p * n)]   # 12 x 42
    
    # Strassen 2x2 @ 2x2 rank-7 decomposition
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b01-b11)
    # M4 = a11*(b10-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b01)
    # M7 = (a01-a11)*(b10+b11)
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    # For each Strassen product, define which A, B, C elements are involved
    # A indices: 0=a00, 1=a01, 2=a10, 3=a11
    # B indices: 0=b00, 1=b01, 2=b10, 3=b11
    # C indices: 0=c00, 1=c01, 2=c10, 3=c11 (row-major)
    
    strassen = [
        # (A_coeffs, B_coeffs, C_coeffs)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),      # M1
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),           # M2
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),           # M3
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),           # M4
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),           # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),           # M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),           # M7
    ]
    
    col = 0
    for side in range(2):  # 0=left (cols 0-1 of A, rows 0-1 of B), 1=right (cols 2-3 of A, rows 2-3 of B)
        a_col_off = side * 2
        b_row_off = side * 2
        for block in range(3):  # 3 blocks of 2 columns in B
            b_col_off = block * 2
            for s_idx in range(7):
                a_coeffs, b_coeffs, c_coeffs = strassen[s_idx]
                
                # U: A[i][j] maps to U[i*m + j]
                for a_idx, sign in a_coeffs.items():
                    a_row = a_idx // 2
                    a_col = a_idx % 2
                    u_idx = a_row * m + (a_col_off + a_col)
                    U[u_idx][col] += sign
                
                # V: B[i][j] maps to V[i*p + j]
                for b_idx, sign in b_coeffs.items():
                    b_row = b_idx // 2
                    b_col = b_idx % 2
                    v_idx = (b_row_off + b_row) * p + (b_col_off + b_col)
                    V[v_idx][col] += sign
                
                # W: C[i][k] maps to W[k*n + i]
                for c_idx, sign in c_coeffs.items():
                    c_row = c_idx // 2
                    c_col = c_idx % 2
                    w_idx = (b_col_off + c_col) * n + c_row
                    W[w_idx][col] += sign
                
                col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    # Verify correctness
    def verify():
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for t in range(rank):
                        s += U[i*m + j][t] * V[j*p + k][t] * W[k*n + i][t]
                    if s != 1:
                        return False
        return True
    
    if not verify():
        raise ValueError("Decomposition verification failed")
    
    return {"U": U, "V": V, "W": W}
