import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    # Split p into 4, 4, 1
    p1, p2, p3 = 4, 4, 1
    
    # Rank for 4x4x4 using Strassen is 49
    # Rank for 4x4x1 using Schoolbook is 16
    rank_total = 49 + 49 + 16
    
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, val_u, val_v, val_w, col):
        U[u_idx][col] = val_u
        V[v_idx][col] = val_v
        W[w_idx][col] = val_w

    # Strassen for 2x2x2
    # Indices: i in {0,1}, j in {0,1}, k in {0,1}
    # M1: (A11+A22)(B11+B22)
    # M2: (A21+A22)B11
    # M3: A11(B12-B22)
    # M4: A22(B21-B11)
    # M5: (A11+A12)B22
    # M6: (A21-A11)(B11+B12)
    # M7: (A12-A22)(B21+B22)
    
    # For 4x4x4, we have 4 blocks of 2x2.
    # Block indices: (bi, bj, bk) where bi,bj,bk in {0,1}
    # A is 4x4, so A[bi*2 + i, bj*2 + j]
    # B is 4x4, so B[bj*2 + j, bk*2 + k]
    # C is 4x4, so C[bi*2 + i, bk*2 + k]
    
    def strassen_4x4x4(offset_col, p_offset):
        col = offset_col
        for bi in range(2):
            for bj in range(2):
                for bk in range(2):
                    # M1
                    # U: A11+A22 -> indices (bi*2+0, bj*2+0) and (bi*2+1, bj*2+1)
                    # V: B11+B22 -> indices (bj*2+0, bk*2+0) and (bj*2+1, bk*2+1)
                    # W: C11 -> (bk*2+0, bi*2+0)
                    # Wait, the decomposition is linear.
                    # We need to express C as sum of U*V*W.
                    # This is getting complicated to do manually.
                    # I will use the known fact that the rank is 49 and just use the schoolbook for 4x4x4 to be safe, 
                    # but that gives 64. 
                    # Actually, I can just use the schoolbook for the whole thing if I can't easily generate Strassen.
                    # But 144 is the current score. I need to beat it.
                    # Let's use the block Strassen properly.
                    pass
        return col
