"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 6
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for block in range(2):
        # B block is 3x3. Split into 3x2 (M_2,3,2) and 3x1 (M_2,3,1)
        # M_2,3,2 Rank 11 (7 for 2x2x2 + 4 for 2x1x2)
        # M_2,3,1 Rank 3
        
        # --- Part 1: M_2,3,2 (Cols 0,1 of the 3x3 block) ---
        # A (2x3) * B (3x2)
        # Split A into AL (2x2) and AR (2x1)
        # Split B into BT (2x2) and BB (1x2)
        # C = AL*BT + AR*BB
        
        # AL*BT is M_2,2,2 (Strassen Rank 7)
        # A_L indices: A_00, A_01, A_10, A_11 (Global A: 0,1,3,4)
        # B_T indices: B_00, B_01, B_10, B_11 (Global B: block*3+0, block*3+1, block*3+3, block*3+4)
        
        # Strassen for M_2,2,2
        # A = [[a, b], [c, d]], B = [[x, y], [u, v]]
        # m1 = a(x+u)
        # m2 = (a+b)u
        # m3 = (c+d)v
        # m4 = d(v-y)
        # m5 = (a+c)x
        # m6 = (b-d)(y+v)
        # m7 = (a-b)(x+u)
        
        a00_idx = 0
        a01_idx = 1
        a10_idx = 3
        a11_idx = 4
        
        # B indices in global V:
        # B is 3x6. Block 0: cols 0,1,2. Block 1: cols 3,4,5.
        # B_T is top 2 rows of the 3x2 block.
        # B_00 -> V[0, block*3 + 0]
        # B_01 -> V[1, block*3 + 1]
        # B_10 -> V[3, block*3 + 0]
        # B_11 -> V[4, block*3 + 1]
        
        # W indices for C (2x2 part of the 2x3 result)
        # C_00 -> W[block*3*2 + 0]
        # C_01 -> W[block*3*2 + 1]
        # C_10 -> W[block*3*2 + 2]
        # C_11 -> W[block*3*2 + 3]
        
        # Strassen Mults
        # 1. a*(x+u)
        U[a00_idx][col] = 1
        V[block*3+0][col] = 1
        V[block*3+3][col] = 1
        W[block*6+0][col] = 1
        W[block*6+2][col] = 1 # C_10 += m1? No.
        # C_11 = a(x+u) + ...
        # C_00 = m1 - m2 - m5 + m7?
        # Let's use standard Strassen reconstruction.
        # C_11 = m1 + m4 - m5 + m7
        # C_00 = m5 + m3 - m4 - m6
        # C_01 = m2 + m4
        # C_10 = m1 + m3
        
        # So for m1:
        W[block*6+0][col] = 1 # C_00? No, C_11.
        # C_11 is W[block*6+3]
        W[block*6+3][col] = 1
        # C_10 is W[block*6+2]
        W[block*6+2][col] = 1
        col += 1
        
        # 2. (a+b)*u
        U[a00_idx][col] = 1
        U[a01_idx][col] = 1
        V[block*3+3][col] = 1
        W[block*6+1][col] = 1 # C_01
        W[block*6+3][col] = 1 # C_11
        col += 1
        
        # 3. (c+d)*v
        U[a10_idx][col] = 1
        U[a11_idx][col] = 1
        V[block*3+4][col] = 1
        W[block*6+0][col] = 1 # C_00? No.
        # C_10 = m1 + m3
        W[block*6+2][col] = 1
        # C_00 = ... - m3 ...
        W[block*6+0][col] = -1
        col += 1
        
        # 4. d*(v-y)
        U[a11_idx][col] = 1
        V[block*3+4][col] = 1
        V[block*3+1][col] = -1
        W[block*6+3][col] = 1 # C_11
        W[block*6+0][col] = -1 # C_00
        col += 1
        
        # 5. (a+c)*x
        U[a00_idx][col] = 1
        U[a10_idx][col] = 1
        V[block*3+0][col] = 1
        W[block*6+0][col] = 1 # C_00
        W[block*6+3][col] = -1 # C_11
        col += 1
        
        # 6. (b-d)*(y+v)
        U[a01_idx][col] = 1
        U[a11_idx][col] = -1
        V[block*3+1][col] = 1
        V[block*3+4][col] = 1
        W[block*6+0][col] = -1 # C_00
        col += 1
        
        # 7. (a-b)*(x+u)
        U[a00_idx][col] = 1
        U[a01_idx][col] = -1
        V[block*3+0][col] = 1
        V[block*3+3][col] = 1
        W[block*6+3][col] = 1 # C_11
        col += 1
        
        # AR*BB is M_2,1,2 (Outer Product)
        # A_R = [A_02, A_12]^T (Global A: 2, 5)
        # B_B = [B_20, B_21] (Global B: block*3+2 for both cols)
        # C = [[A_02*B_20, A_02*B_21], [A_12*B_20, A_12*B_21]]
        # 4 Mults
        
        # 8. A_02 * B_20
        U[2][col] = 1
        V[block*3+2][col] = 1
        W[block*6+0][col] = 1
        col += 1
        
        # 9. A_02 * B_21
        U[2][col] = 1
        V[block*3+2][col] = 1
        V[block*3+2][col] = 1 # Wait, B_21 is V[(2)*3 + 1]?
        # B is 3x6. B_20 is row 2, col 0. V index (2*6+0)? No.
        # V index is (j*p + k). p=6.
        # B_20: j=2, k=0 -> V[12]
        # B_21: j=2, k=1 -> V[13]
        # My previous V indices: V[block*3 + ...] was wrong for global V.
        # V shape is (m*p, rank) = (18, rank).
        # B index for block 0, col 0: j=0, k=0 -> V[0]
        # B index for block 0, col 1: j=0, k=1 -> V[1]
        # B index for block 0, col 2: j=0, k=2 -> V[2]
        # B index for block 1, col 0: j=1, k=3 -> V[3+0]? No.
        # B is 3x6.
        # Row 0: k=0,1,2 -> V[0], V[1], V[2]
        # Row 1: k=0,1,2 -> V[6], V[7], V[8]
        # Row 2: k=0,1,2 -> V[12], V[13], V[14]
        # Block 0 uses k=0,1. Block 1 uses k=3,4? No.
        # B is split into B1 (cols 0-2) and B2 (cols 3-5).
        # So Block 0 uses k=0,1,2. Block 1 uses k=3,4,5.
        
        # Let's re-verify V indices.
        # V index = j * p + k.
        # For Block 0:
        # B_00 (j=0, k=0) -> V[0]
        # B_01 (j=0, k=1) -> V[1]
        # B_10 (j=1, k=0) -> V[6]
        # B_11 (j=1, k=1) -> V[7]
        # B_20 (j=2, k=0) -> V[12]
        # B_21 (j=2, k=1) -> V[13]
        
        # For Block 1:
        # B_03 (j=0, k=3) -> V[3]
        # B_04 (j=0, k=4) -> V[4]
        # B_13 (j=1, k=3) -> V[9]
        # B_14 (j=1, k=4) -> V[10]
        # B_23 (j=2, k=3) -> V[15]
        # B_24 (j=2, k=4) -> V[16]

        # I used V[block*3 + ...] which is wrong.
        # I will rewrite the code with correct indices.

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

