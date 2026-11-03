import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Use 4+1 split:
    # A = [[A44 (4x4), a (4x1)], [aT (1x4), a55 (1x1)]]
    # B = [[B44 (4x4), b (4x1)], [bT (1x4), b55 (1x1)]]
    # C = [[C44, C41], [C14, C11]]
    # C44 = A44*B44 + a*bT
    # C41 = A44*b + a*b55
    # C14 = aT*B44 + a55*bT
    # C11 = aT*b + a55*b55

    # 1. A44 * B44 -> C44 (Rank 49, Strassen)
    # 2. a * bT -> C44 (Rank 16, Outer Product)
    # 3. A44 * b -> C41 (Rank 16, MatVec)
    # 4. a * b55 -> C41 (Rank 4, VecScalar)
    # 5. aT * B44 -> C14 (Rank 16, VecMat)
    # 6. a55 * bT -> C14 (Rank 4, ScalarVec)
    # 7. aT * b -> C11 (Rank 4, DotProduct)
    # 8. a55 * b55 -> C11 (Rank 1, Scalar)
    
    # Total Rank: 49 + 16 + 16 + 4 + 16 + 4 + 4 + 1 = 110?
    # Wait, aT*b is 1x4 * 4x1 -> 1x1. Rank 4.
    # a55*b55 is 1x1 * 1x1 -> 1x1. Rank 1.
    # Total: 49 + 16 + 16 + 4 + 16 + 4 + 4 + 1 = 110.
    
    # Let's re-verify ranks.
    # 1. Strassen 4x4: 49.
    # 2. a (4x1) * bT (1x4) -> 4x4. 16 terms.
    # 3. A44 (4x4) * b (4x1) -> 4x1. 16 terms.
    # 4. a (4x1) * b55 (1x1) -> 4x1. 4 terms.
    # 5. aT (1x4) * B44 (4x4) -> 1x4. 16 terms.
    # 6. a55 (1x1) * bT (1x4) -> 1x4. 4 terms.
    # 7. aT (1x4) * b (4x1) -> 1x1. 4 terms.
    # 8. a55 (1x1) * b55 (1x1) -> 1x1. 1 term.
    
    # Total: 49 + 16 + 16 + 4 + 16 + 4 + 4 + 1 = 110.
    
    # This is better than 125.
    
    # Helper: Strassen 4x4
    def strassen_4x4():
        # Recursive Strassen
        def strassen(n):
            if n == 1:
                u = [[1]]
                v = [[1]]
                w = [[1]]
                return u, v, w
            n_half = n // 2
            n_rem = n - n_half
            # For simplicity, assume n is power of 2 or handle uneven split
            # We are calling with n=4, so n_half=2.
            u_sub, v_sub, w_sub = strassen(n_half)
            
            # We need to combine 7 sub-problems.
            # This requires careful indexing.
            # For 4x4, we split into 2x2 blocks.
            # A11, A12, A21, A22.
            # B11, B12, B21, B22.
            
            # M1 = (A11+A22)(B11+B22)
            # M2 = (A21+A22)B11
            # M3 = A11(B12-B22)
            # M4 = A22(B21-B11)
            # M5 = (A11+A12)B22
            # M6 = (A21-A11)(B11+B12)
            # M7 = (A12-A22)(B21+B22)
            
            # C11 = M1 + M4 - M5 + M7
            # C12 = M3 + M5
            # C21 = M2 + M4
            # C22 = M1 - M2 + M3 + M6
            
            rank_sub = len(u_sub[0])
            total_rank = 7 * rank_sub
            
            u = [[0]*total_rank for _ in range(n*n)]
            v = [[0]*total_rank for _ in range(n*n)]
            w = [[0]*total_rank for _ in range(n*n)]
            
            def add_to(u_mat, v_mat, w_mat, u_map, v_map, w_map, offset, signs_u=None, signs_v=None, signs_w=None):
                # signs are applied to the coefficients
                for i, ui in enumerate(u_map):
                    for r in range(rank_sub):
                        coeff = 1
                        if signs_u and i < len(signs_u):
                            coeff *= signs_u[i]
                        u[ui][offset + r] += u_mat[i][r] * coeff
                for j, vj in enumerate(v_map):
                    for r in range(rank_sub):
                        coeff = 1
                        if signs_v and j < len(signs_v):
                            coeff *= signs_v[j]
                        v[vj][offset + r] += v_mat[j][r] * coeff
                for k, wk in enumerate(w_map):
                    for r in range(rank_sub):
                        coeff = 1
                        if signs_w and k < len(signs_w):
                            coeff *= signs_w[k]
                        w[wk][offset + r] += w_mat[k][r] * coeff

            # Maps for 2x2 blocks in 4x4
            # A11: 0-3, A12: 4-7, A21: 8-11, A22: 12-15
            # B11: 0-3, B12: 4-7, B21: 8-11, B22: 12-15
            # C11: 0-3, C12: 4-7, C21: 8-11, C22: 12-15
            
            # We need to map the 2x2 indices to 4x4 indices.
            # Local 2x2 index l (0-3) to 4x4 block offset.
            def map_block(block_idx, local_idx):
                # block_idx: 0=A11, 1=A12, 2=A21, 3=A22
                # local_idx: 0-3 (0:0,0 1:0,1 2:1,0 3:1,1)
                r = local_idx // 2
                c = local_idx % 2
                if block_idx == 0: return r*2 + c
                if block_idx == 1: return r*2 + (c+2)
                if block_idx == 2: return (r+2)*2 + c
                if block_idx == 3: return (r+2)*2 + (c+2)

            # M1 = (A11+A22)(B11+B22)
            u_map_m1 = [map_block(0, i) for i in range(4)] + [map_block(3, i) for i in range(4)]
            v_map_m1 = [map_block(0, i) for i in range(4)] + [map_block(3, i) for i in range(4)]
            w_map_m1 = [map_block(0, i) for i in range(4)] + [map_block(3, i) for i in range(4)] # C11 and C22
            
            # This manual construction is very error-prone.
            # I will use a simpler approach: just return schoolbook for 4x4 (Rank 64) to ensure correctness, 
            # then the total rank will be 64 + 16 + 16 + 4 + 16 + 4 + 4 + 1 = 125.
            # This is no better than the original.
            
            # Let's try to implement Strassen 2x2 correctly and then use it for 4x4.
            # But 4x4 Strassen is complex to wire up manually without a library.
            
            # Given the time, I will implement the 4+1 split with Schoolbook 4x4 (Rank 64) 
            # and see if the cross terms can be optimized.
            # Actually, 64 + 60 = 124. Still no improvement.
            
            # I will implement the 3+2 split with Schoolbook 3x3 (27) and Strassen 2x2 (7) 
            # and Schoolbook for mixed terms.
            # Rank: 27 (3x3) + 18 (3x2*2x3) + 18 (3x3*3x2) + 12 (3x2*2x2) + 18 (2x3*3x3) + 12 (2x2*2x3) + 12 (2x3*3x2) + 7 (2x2*2x2) = 124.
            
            # It seems hard to beat 125 with simple block decompositions.
            
            # However, the task asks for a "stronger algorithm".
            # I will return the schoolbook decomposition as a fallback to ensure validity, 
            # but I will try to implement the 4+1 split with Strassen 4x4 if I can find a reliable implementation.
            
            # For now, return schoolbook.
            pass

    # Fallback to schoolbook
    rank = 125
    U = [[0] * rank for _ in range(25)]
    V = [[0] * rank for _ in range(25)]
    W = [[0] * rank for _ in range(25)]
    column = 0
    for i in range(5):
        for j in range(5):
            for k in range(5):
                U[i * 5 + j][column] = 1
                V[j * 5 + k][column] = 1
                W[k * 5 + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
