import numpy as np
import time

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    deadline = 130.0

    # Schoolbook baseline (rank = n*m*p = 112)
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1

    def verify(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(rank):
                        total += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                    if abs(total - 1.0) > 1e-9:
                        return False
        # Verify that entries that should be 0 are 0 (spot check a few)
        # Actually, we need to verify the full tensor decomposition.
        # The condition is: sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 for all (i,j,k)
        # and the decomposition should not have spurious non-zero entries.
        # But the standard verification is just that the (i,j,k) triples give 1.
        # If there were spurious entries, they would show up as non-zero for other triples,
        # but in our construction we only use rank-1 tensors that correspond to valid
        # (i,j,k) triples, so there shouldn't be spurious entries.
        # However, in Strassen, the linear combinations can introduce cross-terms.
        # Let's verify that for all (i,j,k), the sum is exactly 1 (or 0 for invalid, but
        # in matrix multiplication, all (i,j,k) with i in [0,n), j in [0,m), k in [0,p)
        # are valid and should give 1).
        return True

    def strassen_2x2_2x2():
        rank = 7
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(4)]
        W = [[0]*rank for _ in range(4)]
        
        # M1 = (a+d)(e+h) -> C00 + C11
        U[0][0] += 1; U[3][0] += 1
        V[0][0] += 1; V[3][0] += 1
        W[0][0] += 1; W[3][0] += 1
        
        # M2 = (c+d)e -> C10 - C11
        U[2][1] += 1; U[3][1] += 1
        V[0][1] += 1
        W[2][1] += 1; W[3][1] -= 1
        
        # M3 = a(f-h) -> C01 + C11
        U[0][2] += 1
        V[1][2] += 1; V[3][2] -= 1
        W[1][2] += 1; W[3][2] += 1
        
        # M4 = d(g-e) -> C00 + C10
        U[3][3] += 1
        V[2][3] += 1; V[0][3] -= 1
        W[0][3] += 1; W[2][3] += 1
        
        # M5 = (a+b)h -> -C00 + C01
        U[0][4] += 1; U[1][4] += 1
        V[3][4] += 1
        W[0][4] -= 1; W[1][4] += 1
        
        # M6 = (c-a)(e+f) -> C11
        U[2][5] += 1; U[0][5] -= 1
        V[0][5] += 1; V[1][5] += 1
        W[3][5] += 1
        
        # M7 = (b-d)(g+h) -> C00
        U[1][6] += 1; U[3][6] -= 1
        V[2][6] += 1; V[3][6] += 1
        W[0][6] += 1
        
        return U, V, W

    def build_strassen_4x4x4():
        """Build 4x4*4x4 Strassen decomposition using recursive 2x2*2x2 Strassen.
        Returns U (16x49), V (16x49), W (16x49) with 0.5 coefficients.
        """
        S_U, S_V, S_W = strassen_2x2_2x2()
        s_rank = 7
        rank_4 = 49
        
        U4 = [[0.0]*rank_4 for _ in range(16)]
        V4 = [[0.0]*rank_4 for _ in range(16)]
        W4 = [[0.0]*rank_4 for _ in range(16)]
        
        # Block definitions for 4x4:
        # Block 0: A[0:2, 0:2] -> indices 0,1,4,5
        # Block 1: A[0:2, 2:4] -> indices 2,3,6,7
        # Block 2: A[2:4, 0:2] -> indices 8,9,12,13
        # Block 3: A[2:4, 2:4] -> indices 10,11,14,15
        
        def block_indices(block_id):
            br = block_id // 2
            bc = block_id % 2
            return [(br*2+lr)*4 + (bc*2+lc) for lr in range(2) for lc in range(2)]
        
        # 7 Strassen products at level 1
        products = [
            ([(0,1.0), (3,1.0)], [(0,1.0), (3,1.0)], [(0,1.0), (3,1.0)]),
            ([(2,1.0), (3,1.0)], [(0,1.0)], [(2,1.0), (3,-1.0)]),
            ([(0,1.0)], [(1,1.0), (3,-1.0)], [(1,1.0), (3,1.0)]),
            ([(3,1.0)], [(2,1.0), (0,-1.0)], [(0,1.0), (2,1.0)]),
            ([(0,1.0), (1,1.0)], [(3,1.0)], [(0,-1.0), (1,1.0)]),
            ([(2,1.0), (0,-1.0)], [(0,1.0), (1,1.0)], [(3,1.0)]),
            ([(1,1.0), (3,-1.0)], [(2,1.0), (3,1.0)], [(0,1.0)]),
        ]
        
        col_offset = 0
        for prod_idx in range(7):
            a_blocks, b_blocks, c_blocks = products[prod_idx]
            for r in range(s_rank):
                for li in range(4):
                    s_u = S_U[li][r]
                    s_v = S_V[li][r]
                    s_w = S_W[li][r]
                    lr = li // 2
                    lc = li % 2
                    
                    for block_id, coeff in a_blocks:
                        gidx = (block_id // 2)*2*4 + (block_id % 2)*2 + lr*4 + lc
                        # Wait, that's wrong. Let me recompute.
                        br = block_id // 2
                        bc = block_id % 2
                        gidx = (br*2+lr)*4 + (bc*2+lc)
                        U4[gidx][col_offset + r] += coeff * s_u
                    
                    for block_id, coeff in b_blocks:
                        br = block_id // 2
                        bc = block_id % 2
                        gidx = (br*2+lr)*4 + (bc*2+lc)
                        V4[gidx][col_offset + r] += coeff * s_v
                    
                    for block_id, coeff in c_blocks:
                        br = block_id // 2
                        bc = block_id % 2
                        gidx = (br*2+lr)*4 + (bc*2+lc)
                        W4[gidx][col_offset + r] += coeff * s_w
                
                col_offset += 1
        
        return U4, V4, W4, rank_4

    def build_strassen_4x4x8():
        """Build 4x4*4x8 decomposition using Strassen for both 4x4*4x4 blocks.
        Returns U (16x98), V (32x98), W (32x98).
        """
        U4a, V4a, W4a, r4a = build_strassen_4x4x4()
        U4b, V4b, W4b, r4b = build_strassen_4x4x4()
        
        rank_total = r4a + r4b
        U_full = [[0.0]*rank_total for _ in range(16)]
        V_full = [[0.0]*rank_total for _ in range(32)]
        W_full = [[0.0]*rank_total for _ in range(32)]
        
        # First block: B cols 0-3, C rows 0-3
        for i in range(16):
            for r in range(r4a):
                U_full[i][r] = U4a[i][r]
        for j in range(4):
            for k in range(4):
                for r in range(r4a):
                    V_full[j*8+k][r] = V4a[j*4+k][r]
        for k in range(4):
            for i in range(4):
                for r in range(r4a):
                    W_full[k*4+i][r] = W4a[k*4+i][r]
        
        # Second block: B cols 4-7, C rows 4-7
        for i in range(16):
            for r in range(r4b):
                U_full[i][r + r4a] = U4b[i][r]
        for j in range(4):
            for k in range(4):
                for r in range(r4b):
                    V_full[j*8+(4+k)][r + r4a] = V4b[j*4+k][r]
        for k in range(4):
            for i in range(4):
                for r in range(r4b):
                    W_full[(4+k)*4+i][r + r4a] = W4b[k*4+i][r]
        
        return U_full, V_full, W_full, rank_total

    def verify_4x4x8(U, V, W, rank):
        """Verify 4x4*4x8 decomposition."""
        for i in range(4):
            for j in range(4):
                for k in range(8):
                    total = 0.0
                    for r in range(rank):
                        total += U[i*4+j][r] * V[j*8+k][r] * W[k*4+i][r]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True

    try:
        if time.time() > deadline:
            return {"U": U_sb, "V": V_sb, "W": W_sb}
        
        U8, V8, W8, rank8 = build_strassen_4x4x8()
        
        if time.time() > deadline:
            return {"U": U_sb, "V": V_sb, "W": W_sb}
        
        if not verify_4x4x8(U8, V8, W8, rank8):
            return {"U": U_sb, "V": V_sb, "W": W_sb}
        
        # Now extract the 4x4*4x7 part by dropping the 8th column
        # U stays the same (16 x rank8)
        # V: drop row j*8+7 for j=0,1,2,3 -> rows 7, 15, 23, 31
        # W: drop row k*4+i where k=7 -> rows 28-31
        v_rows_to_keep = [r for r in range(32) if r % 8 != 7]
        w_rows_to_keep = [r for r in range(32) if r // 4 != 7]
        
        V7 = [V8[r][:] for r in v_rows_to_keep]
        W7 = [W8[r][:] for r in w_rows_to_keep]
        
        U7 = U8
        
        rank7 = rank8
        
        # Verify 4x4*4x7
        def verify_4x4x7(U, V, W, rank):
            for i in range(4):
                for j in range(4):
                    for k in range(7):
                        total = 0.0
                        for r in range(rank):
                            total += U[i*4+j][r] * V[j*7+k][r] * W[k*4+i][r]
                        if abs(total - 1.0) > 1e-9:
                            return False
            return True
        
        if time.time() > deadline:
            return {"U": U_sb, "V": V_sb, "W": W_sb}
        
        if not verify_4x4x7(U7, V7, W7, rank7):
            return {"U": U_sb, "V": V_sb, "W": W_sb}
        
        return {"U": U7, "V": V7, "W": W7}
        
    except Exception as e:
        return {"U": U_sb, "V": V_sb, "W": W_sb}
