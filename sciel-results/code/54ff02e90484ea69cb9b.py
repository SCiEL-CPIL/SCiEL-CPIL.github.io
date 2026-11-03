import time
_start_time = time.time()
_TIMEOUT = 145.0  # safety margin before 150s kill

def construct():
    n, m, p = 2, 4, 8
    
    # Embed 2x4 * 4x8 into 4x4 * 4x4 multiplication.
    # A_embed is 4x4 with A in top-left 2x4 block, zeros elsewhere.
    # B_embed is 4x4. We need C = A * B where A is 2x4, B is 4x8.
    # 
    # Strategy: Split B (4x8) into B_left (4x4) and B_right (4x4).
    # C = A * B = A * [B_left | B_right] = [A*B_left | A*B_right]
    # 
    # For each 4x4 sub-problem, embed A (2x4) into 4x4 as:
    # A_embed = [[A, 0], [0, 0]] (top-left 2x4 is A, rest zeros)
    # 
    # Then A_embed * B_sub = [[A*B_sub, 0], [0, 0]]
    # We only need the top-left 2x4 of the result.
    # 
    # Use Strassen's rank-32 algorithm for 4x4*4x4.
    # This requires decomposing 4x4 into 2x2 blocks and applying Strassen.
    # 
    # Actually, let me think more carefully. The standard approach:
    # 4x4 * 4x4 can be done with 32 multiplications of 2x2 matrices
    # (Strassen on 4x4 gives rank 32 for the tensor).
    # 
    # But we need a specific decomposition. Let me construct it directly.
    # 
    # For 4x4 * 4x4 -> 4x4, partition into 2x2 blocks:
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    # 
    # Strassen on 4x4 (each block is 2x2):
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B01-B11)
    # M3 = A11(B10-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    # 
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    # 
    # This gives 7 rank-1 tensors for 2x2*2x2, but for 4x4 we need
    # to apply this recursively or use a direct rank-32 decomposition.
    # 
    # Actually, the rank of 4x4*4x4 multiplication tensor is 32
    # (this is the Strassen-based bound). Let me construct it.
    # 
    # For 4x4 * 4x4, we can view it as 4 blocks of 2x2.
    # Each 2x2*2x2 sub-multiplication has rank 7 (Strassen).
    # But there's overlap, so total rank is less than 4*7=28? No,
    # actually for 4x4*4x4 the Strassen rank is 32.
    # 
    # Let me use a different approach: directly construct the rank-32
    # decomposition for 4x4*4x4.
    # 
    # Actually, I think the cleanest approach is:
    # 1. Use the known rank-32 decomposition for 4x4*4x4 (Strassen-based)
    # 2. For each of the two 4x4 sub-problems (B_left and B_right),
    #    embed A into 4x4 and apply the decomposition
    # 3. Extract only the top 2 rows of the 4x4 result
    # 
    # But wait - if I embed A into 4x4 with zeros in bottom 2 rows,
    # then A_embed * B = [[A*B, 0], [0, 0]]
    # So I only need the top-left 2x4 of the 4x4 result.
    # 
    # The key insight: since the bottom 2 rows of A_embed are zero,
    # many of the Strassen terms will have zero contributions to
    # the bottom rows of the result. But the rank of the overall
    # decomposition is still 32 per 4x4*4x4 sub-problem.
    # 
    # Total rank = 2 * 32 = 64? That's worse than 56!
    # 
    # Hmm, let me reconsider. Maybe I should use a smarter embedding.
    # 
    # Alternative: Since A is only 2x4, I can use a rank-7 Strassen
    # decomposition for each 2x4 * 4x4 sub-problem.
    # 2x4 * 4x4: partition A into [A_left, A_right] (each 2x2)
    # and B into [[B00, B01], [B10, B11]] (each 2x2)
    # 
    # C = A_left * B_left_half + A_right * B_right_half
    # where B_left_half = [B00; B10] (4x2) and B_right_half = [B01; B11] (4x2)
    # 
    # This is getting complicated. Let me try a different approach.
    # 
    # Direct construction: For 2x4 * 4x8, I can partition into:
    # A = [A0 | A1] where A0, A1 are 2x2
    # B = [[B00, B01, B02, B03], [B10, B11, B12, B13], [B20, B21, B22, B23], [B30, B31, B32, B33]]
    # 
    # C = A0 * B[:, 0:2] + A1 * B[:, 2:4] + A0 * B[:, 4:6] + A1 * B[:, 6:8]
    # 
    # Each 2x2 * 4x2 sub-problem can be done with rank 7 (Strassen-like).
    # Total: 4 * 7 = 28? But that seems too optimistic.
    # 
    # Actually, 2x2 * 4x2: partition B into two 2x2 blocks vertically.
    # C = A * B_top + A * B_bottom
    # This is not a standard tensor decomposition.
    # 
    # Let me go back to the original approach and try to reduce the rank
    # by finding linear dependencies among the 56 Strassen terms.
    # 
    # Actually, let me try the embedding approach more carefully.
    # 
    # Key insight: If I embed A (2x4) into 4x4 as [[A, 0], [0, 0]],
    # then in the Strassen decomposition of 4x4*4x4, the terms that
    # involve the bottom 2 rows of A_embed will be zero. So I can
    # potentially reduce the rank.
    # 
    # Let me construct the rank-32 decomposition for 4x4*4x4 and
    # see which terms survive when A has zeros in the bottom 2 rows.
    
    # For now, let me just return the current 56-rank solution
    # and add a time guard.
    
    total_rank = 56
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    strassen_terms = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]

    group = 0
    for jj in range(p // 2):
        for half in range(2):
            base_r = group * 7
            group += 1
            
            a_map = {
                0: 0 * m + (2 * half + 0),
                1: 0 * m + (2 * half + 1),
                2: 1 * m + (2 * half + 0),
                3: 1 * m + (2 * half + 1)
            }
            
            b_map = {
                0: (2 * half + 0) * p + (2 * jj + 0),
                1: (2 * half + 0) * p + (2 * jj + 1),
                2: (2 * half + 1) * p + (2 * jj + 0),
                3: (2 * half + 1) * p + (2 * jj + 1)
            }
            
            c_map = {
                0: (2 * jj + 0) * n + 0,
                1: (2 * jj + 1) * n + 0,
                2: (2 * jj + 0) * n + 1,
                3: (2 * jj + 1) * n + 1
            }

            for r in range(7):
                rr = base_r + r
                u_list, v_list, w_list = strassen_terms[r]
                
                for idx, val in u_list:
                    U[a_map[idx]][rr] += val
                    
                for idx, val in v_list:
                    V[b_map[idx]][rr] += val
                    
                for idx, val in w_list:
                    W[c_map[idx]][rr] += val

    return {"U": U, "V": V, "W": W}
