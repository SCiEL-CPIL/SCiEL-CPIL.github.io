"""Matrix multiplication 3x5 x 5x7 via Strassen padding to 4x8."""
import time

DIMENSIONS = [3, 5, 7]

def construct():
    deadline = time.time() + 140.0
    n, m, p = DIMENSIONS  # 3, 5, 7
    # Pad to N=4, M=8, P=4
    N, M, P = 4, 8, 4
    nB = N // 2  # 2
    mB = M // 2  # 4
    pB = P // 2  # 2
    # Strassen base rank for 2x2x2
    s_rank = 7
    # Each M is (nB x mB) x (mB x pB) = 2x4 x 4x2, rank = 4 * s_rank = 28
    m_rank = mB * s_rank  # 28
    total_rank = 7 * m_rank  # 196? No wait.
    
    # Actually: 7 M's, each M has rank m_rank = mB * s_rank = 4*7 = 28
    # Total rank = 7 * 28 = 196. That's worse than schoolbook 105!
    
    # Let me reconsider. The schoolbook rank is n*m*p = 3*5*7 = 105.
    # With padding to 4x8x4, schoolbook would be 4*8*4 = 128.
    # Strassen on 4x8x4: 7 blocks, each 2x4x4, schoolbook rank per block = 2*4*4=32
    # Strassen rank per block = 4*7 = 28 (since mB=4, use 4 independent 2x2x2 Strassen)
    # Total = 7 * 28 = 196. Worse.
    
    # The issue: Strassen helps when all three dimensions are equal (2x2x2).
    # For 2x4x4, we decompose the 4 into 2+2, giving 4 sub-2x2x2 problems, each rank 7.
    # So rank per 2x4x4 block = 4*7 = 28 vs schoolbook 32. Savings per block.
    # 7 blocks * 28 = 196 vs 7*32 = 224 for Strassen-schoolbook, or 128 pure schoolbook.
    
    # So 196 > 105. Strassen padding doesn't help here!
    
    # Alternative: Don't pad. Use the actual 3x5x7 structure.
    # Split 3 into 2+1, 5 into 4+1 or 3+2, 7 into 4+3 or 2+5.
    # 
    # Let me think differently. The minimum rank for 3x5x7 matrix multiplication.
    # Lower bounds: max(3*5, 5*7, 3*7) = max(15,35,21) = 35.
    # Upper bounds: schoolbook = 105.
    # 
    # Can we do better than 105?
    # Split: A (3x5) = [A_top(2x5); A_bot(1x5)], B (5x7) = [B_left(5x4) | B_right(5x3)]
    # C = A*B is 3x7.
    # C_top_left(2x4) = A_top * B_left
    # C_top_right(2x3) = A_top * B_right  
    # C_bot_left(1x4) = A_bot * B_left
    # C_bot_right(1x3) = A_bot * B_right
    #
    # Each sub-product:
    # 2x4 x 4x4: rank 32 schoolbook, or use Strassen? 2x4x4: split 4=2+2, get 4 sub 2x2x2, each rank 7, total 28.
    # 2x3 x 3x3: rank 18 schoolbook.
    # 1x4 x 4x4: rank 16 schoolbook.
    # 1x3 x 3x3: rank 9 schoolbook.
    #
    # Total with Strassen on 2x4x4: 28+18+16+9 = 71 < 105!
    
    # Let me verify: 2x4x4 with Strassen.
    # A is 2x4, B is 4x4. Split B into 4 columns: B = [B0(4x2) | B1(4x2)].
    # Actually split both: A = [A0(2x2) | A1(2x2)], B = [B00(2x2); B01(2x2)] * [B10(2x2); B11(2x2)]
    # Wait, B is 4x4, so B = [[B00, B01], [B10, B11]] each 2x2.
    # C = A*B is 2x4 = [C0(2x2) | C1(2x2)].
    # C0 = A0*B00 + A1*B10
    # C1 = A0*B01 + A1*B11
    # Using Strassen on each 2x2x2:
    # C0 needs: M_a = A0*B00, M_b = A1*B10. That's 2 multiplications, each rank 7 = 14.
    # C1 needs: M_c = A0*B01, M_d = A1*B11. That's 2 multiplications, each rank 7 = 14.
    # Total for 2x4x4: 28. Yes!
    
    # Now 2x3 x 3x3: A(2x3)*B(3x3) = C(2x3). Schoolbook rank = 2*3*3 = 18.
    # Can we do better? Split: A = [A0(2x2)|A1(2x1)], B = [B0(2x3); B1(1x3)]
    # C = A0*B0 + A1*B1. A0*B0 is 2x2x3 (rank 12), A1*B1 is 2x1x3 (rank 6). Total 18. No gain.
    
    # 1x4 x 4x4: A(1x4)*B(4x4)=C(1x4). Rank = 1*4*4 = 16.
    # Split: A=[A0(1x2)|A1(1x2)], B=[[B00(2x2),B01(2x2)],[B10(2x2),B11(2x2)]]
    # C = [A0*B00 + A1*B10 | A0*B01 + A1*B11]
    # 4 multiplications of 1x2x2, each rank 4. Total 16. No Strassen gain for 1x2x2.
    
    # 1x3 x 3x3: rank 9. No gain.
    
    # Total: 28 + 18 + 16 + 9 = 71.
    
    # But wait, can we share Strassen products across the blocks?
    # C_top_left = A_top * B_left where A_top is 2x5, B_left is 5x4.
    # C_top_right = A_top * B_right where B_right is 5x3.
    # These share A_top. If we split A_top into [A0(2x2)|A1(2x3)], 
    # B_left = [B_L0(2x4); B_L1(3x4)], B_right = [B_R0(2x3); B_R1(3x3)]
    # C_top_left = A0*B_L0 + A1*B_L1
    # C_top_right = A0*B_R0 + A1*B_R1
    # A0*B_L0: 2x2 x 2x4 = 2x4, rank 2*2*4=16 schoolbook, or split 4=2+2: 2*(2x2x2) = 2*7=14
    # A1*B_L1: 2x3 x 3x4 = 2x4, rank 24 schoolbook
    # A0*B_R0: 2x2 x 2x3 = 2x3, rank 12
    # A1*B_R1: 2x3 x 3x3 = 2x3, rank 18
    # Total: 14+24+12+18 = 68 for top part.
    
    # This is getting complicated. Let me just implement the straightforward 71-rank solution.
    
    # Actually, let me reconsider the whole approach. Let me use a cleaner decomposition.
    
    # Strategy: Split 3 = 2+1, 5 = 4+1, 7 = 4+3
    # A (3x5) = [A_TL(2x4) | A_TR(2x1); A_BL(1x4) | A_BR(1x1)]
    # B (5x7) = [B_TL(4x4) | B_TR(4x3); B_BL(1x4) | B_BR(1x3)]
    # C (3x7) = [C_TL(2x4) | C_TR(2x3); C_BL(1x4) | C_BR(1x3)]
    # C_TL = A_TL*B_TL + A_TR*B_BL
    # C_TR = A_TL*B_TR + A_TR*B_BR
    # C_BL = A_BL*B_TL + A_BR*B_BL
    # C_BR = A_BL*B_TR + A_BR*B_BR
    
    # Ranks:
    # A_TL*B_TL: 2x4 x 4x4 = 2x4. Use Strassen: split 4=2+2.
    #   A_TL = [A0(2x2)|A1(2x2)], B_TL = [[B00(2x2),B01(2x2)],[B10(2x2),B11(2x2)]]
    #   C_TL_part1 = [A0*B00+A1*B10 | A0*B01+A1*B11]
    #   4 Strassen 2x2x2 products: rank 4*7 = 28
    # A_TR*B_BL: 2x1 x 1x4 = 2x4. Rank = 2*1*4 = 8.
    # A_TL*B_TR: 2x4 x 4x3 = 2x3. Split 4=2+2: A_TL=[A0|A1], B_TR=[[B00(2x3);B01(2x3)]]
    #   = A0*B00 + A1*B01. Each 2x2x3, rank 12. Total 24.
    # A_TR*B_BR: 2x1 x 1x3 = 2x3. Rank 6.
    # A_BL*B_TL: 1x4 x 4x4 = 1x4. Split: A_BL=[A0(1x2)|A1(1x2)], B_TL as before.
    #   = [A0*B00+A1*B10 | A0*B01+A1*B11]. 4 products of 1x2x2, each rank 4. Total 16.
    # A_BR*B_BL: 1x1 x 1x4 = 1x4. Rank 4.
    # A_BL*B_TR: 1x4 x 4x3 = 1x3. Split: = A0*B00 + A1*B01 where B00,B01 are 2x3. Each 1x2x3, rank 6. Total 12.
    # A_BR*B_BR: 1x1 x 1x3 = 1x3. Rank 3.
    
    # Total: 28 + 8 + 24 + 6 + 16 + 4 + 12 + 3 = 101. Only slightly better than 105.
    
    # Hmm. Let me try 3=2+1, 5=3+2, 7=4+3.
    # This is getting messy. Let me try a different strategy.
    
    # What about just using the known best upper bound? For 3x5x7, 
    # the best known is probably still close to schoolbook unless we find clever sharing.
    
    # Let me try: don't split, just use Strassen where applicable on the whole thing.
    # Actually, the simplest improvement: split 5 = 4+1.
    # A (3x5) = [A_L(3x4) | A_R(3x1)]
    # B (5x7) = [B_T(4x7); B_B(1x7)]
    # C = A_L*B_T + A_R*B_B
    # A_L*B_T: 3x4 x 4x7 = 3x7. Rank 84 schoolbook.
    # A_R*B_B: 3x1 x 1x7 = 3x7. Rank 21.
    # Total: 105. No improvement.
    
    # Split 7 = 4+3:
    # A (3x5) = [A_T(3x4); A_B(3x1)]? No, A is 3x5, split columns: [A_L(3x4)|A_R(3x1)]
    # B (5x7) = [B_L(5x4) | B_R(5x3)]
    # C = [A_L*B_L + A_R*B_L | A_L*B_R + A_R*B_R]
    # A_L*B_L: 3x4x4, rank 48. A_R*B_L: 3x1x4, rank 12. A_L*B_R: 3x4x3, rank 36. A_R*B_R: 3x1x3, rank 9.
    # Total: 48+12+36+9 = 105. Same.
    
    # The key insight: for rectangular matrices, Strassen doesn't directly help unless dimensions match.
    # 
    # Let me try the 2+1, 4+1, 4+3 split more carefully with Strassen on 2x4x4.
    # 
    # Actually, I realize I should just implement the 71-rank solution I computed earlier:
    # Split 3=2+1, 5=4+1, 7=4+3
    # C_TL(2x4) = A_TL(2x4)*B_TL(4x4) + A_TR(2x1)*B_BL(1x4)
    # C_TR(2x3) = A_TL(2x4)*B_TR(4x3) + A_TR(2x1)*B_BR(1x3)
    # C_BL(1x4) = A_BL(1x4)*B_TL(4x4) + A_BR(1x1)*B_BL(1x4)
    # C_BR(1x3) = A_BL(1x4)*B_TR(4x3) + A_BR(1x1)*B_BR(1x3)
    
    # Ranks with Strassen on 2x4x4:
    # A_TL*B_TL: 28 (Strassen)
    # A_TR*B_BL: 8
    # A_TL*B_TR: 24 (split 4=2+2, two 2x2x3 products)
    # A_TR*B_BR: 6
    # A_BL*B_TL: 16
    # A_BR*B_BL: 4
    # A_BL*B_TR: 12
    # A_BR*B_BR: 3
    # Total: 28+8+24+6+16+4+12+3 = 101
    
    # Still 101. Let me try to share.
    
    # Actually, let me just go with the straightforward approach and see if I can get below 105.
    # The simplest guaranteed improvement: use Strassen on the 2x4x4 part.
    
    # Let me implement the 101-rank solution. Actually, let me try yet another split.
    
    # 3=1+2, 5=2+3, 7=3+4
    # Too many combinations. Let me just implement a clean solution.
    
    # Final decision: implement the 3=2+1, 5=4+1, 7=4+3 split with Strassen on 2x4x4.
    # Rank = 101.
    
    # Wait, I can do better. Let me reconsider A_TL*B_TR (2x4 x 4x3).
    # Instead of splitting 4=2+2 giving two 2x2x3 (rank 12 each = 24),
    # what if I use a different approach? 2x4x3 minimum rank is at least max(2*4,4*3,2*3)=max(8,12,6)=12.
    # Schoolbook is 24. Can I do 12? Probably not easily.
    
    # Let me just implement rank 101 and verify it works.
    
    # Actually, I just realized: maybe I should try a completely different approach.
    # What's the actual minimum rank for 3x5x7?
    # Lower bound: max(15, 35, 21) = 35.
    # The best known upper bound for small rectangular multiplications...
    
    # For now, let me implement a verified solution with rank < 105.
    # I'll use the 101-rank approach.
    
    # Let me build this step by step.
    
    # Strassen 2x2x2 base:
    # M0 = A00*B00
    # M1 = (A00+A10)*B10
    # M2 = A10*B01
    # M3 = A01*B11
    # M4 = (A01+A11)*B00
    # M5 = (A00-A11)*B11
    # M6 = (A10-A01)*B01
    # C00 = M0+M3+M5+M6
    # C01 = M4+M3
    # C10 = M1+M2
    # C11 = M2-M5+M1
    
    # For 2x4x4 (A is 2x4, B is 4x4, C is 2x4):
    # A = [A0(2x2) | A1(2x2)]
    # B = [[B00(2x2), B01(2x2)], [B10(2x2), B11(2x2)]]
    # C = [C0(2x2) | C1(2x2)]
    # C0 = A0*B00 + A1*B10
    # C1 = A0*B01 + A1*B11
    # Use 4 separate 2x2x2 Strassen:
    # For C0: M_a = A0*B00, M_b = A1*B10. But these are independent, not Strassen.
    # Actually C0 = A0*B00 + A1*B10 is just two matrix products, each 2x2x2.
    # Each 2x2x2 has rank 7 (Strassen) or 8 (schoolbook).
    # So C0 needs rank 7+7 = 14 if we use Strassen for each.
    # Similarly C1 needs 14. Total 28.
    
    # OK let me just implement this. I'll create helper functions.
    
    def strassen_2x2x2():
        """Return (U, V, W) for 2x2x2 Strassen, each of shape (4, 7)."""
        U = [[0]*7 for _ in range(4)]  # 2x2 = 4 entries
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        # Indices: A[i,j] -> U[i*2+j], B[j,k] -> V[j*2+k], C[i,k] -> W[k*2+i]
        # M0 = A00*B00: A00=U[0], B00=V[0], contributes to C00=W[0]
        U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
        # M1 = (A00+A10)*B10: A00=U[0], A10=U[2], B10=V[2], contributes to C10=W[2]
        U[0][1] = 1; U[2][1] = 1; V[2][1] = 1; W[2][1] = 1
        # M2 = A10*B01: A10=U[2], B01=V[1], contributes to C11=W[3]
        U[2][2] = 1; V[1][2] = 1; W[3][2] = 1
        # M3 = A01*B11: A01=U[1], B11=V[3], contributes to C01=W[1] and C11=W[3]
        U[1][3] = 1; V[3][3] = 1; W[1][3] = 1; W[3][3] = 1
        # M4 = (A01+A11)*B00: A01=U[1], A11=U[3], B00=V[0], contributes to C01=W[1]
        U[1][4] = 1; U[3][4] = 1; V[0][4] = 1; W[1][4] = 1
        # M5 = (A00-A11)*B11: A00=U[0], A11=U[3], B11=V[3], contributes to C00=W[0] and C11=W[3]
        U[0][5] = 1; U[3][5] = -1; V[3][5] = 1; W[0][5] = 1; W[3][5] = -1
        # M6 = (A10-A01)*B01: A10=U[2], A01=U[1], B01=V[1], contributes to C00=W[0] and C11=W[3]
        U[2][6] = 1; U[1][6] = -1; V[1][6] = 1; W[0][6] = 1; W[3][6] = 1
        return U, V, W
    
    # Now build the full 3x5x7 decomposition.
    # Split: 3=2+1, 5=4+1, 7=4+3
    # A (3x5): rows 0-1 are "top" (2 rows), row 2 is "bottom" (1 row)
    #          cols 0-3 are "left" (4 cols), col 4 is "right" (1 col)
    # B (5x7): rows 0-3 are "top" (4 rows), row 4 is "bottom" (1 row)
    #          cols 0-3 are "left" (4 cols), cols 4-6 are "right" (3 cols)
    # C (3x7): rows 0-1 "top", row 2 "bottom"; cols 0-3 "left", cols 4-6 "right"
    
    # C_TL(2x4) = A_TL(2x4)*B_TL(4x4) + A_TR(2x1)*B_BL(1x4)
    # C_TR(2x3) = A_TL(2x4)*B_TR(4x3) + A_TR(2x1)*B_BR(1x3)
    # C_BL(1x4) = A_BL(1x4)*B_TL(4x4) + A_BR(1x1)*B_BL(1x4)
    # C_BR(1x3) = A_BL(1x4)*B_TR(4x3) + A_BR(1x1)*B_BR(1x3)
    
    # I'll build each part separately and concatenate.
    
    total_rank = 0
    U_list = []
    V_list = []
    W_list = []
    
    def add_part(U_part, V_part, W_part, rank_part):
        nonlocal total_rank
        for r in range(rank_part):
            U_list.append([U_part[i][r] for i in range(15)])  # 3*5=15
            V_list.append([V_part[i][r] for i in range(35)])  # 5*7=35
            W_list.append([W_part[i][r] for i in range(21)])  # 7*3=21
        total_rank += rank_part
    
    # This approach is getting complex. Let me just build the full U, V, W directly.
    
    # I'll use a dictionary-based approach: for each rank column, specify which (i,j), (j,k), (k,i) entries are nonzero.
    
    # Let me define the structure more carefully.
    
    # A indices: A[i,j] where i in 0..2, j in 0..4. U index: i*5+j.
    # B indices: B[j,k] where j in 0..4, k in 0..6. V index: j*7+k.
    # C indices: C[i,k] where i in 0..2, k in 0..6. W index: k*3+i (per convention W[k*n+i]).
    
    # Part 1: A_TL*B_TL where A_TL is A[0:2, 0:4], B_TL is B[0:4, 0:4]
    # This is 2x4 x 4x4 = 2x4.
    # Split A_TL = [A0(2x2) | A1(2x2)] where A0 = A[0:2,0:2], A1 = A[0:2,2:4]
    # Split B_TL = [[B00(2x2), B01(2x2)], [B10(2x2), B11(2x2)]]
    #   B00 = B[0:2,0:2], B01 = B[0:2,2:4], B10 = B[2:4,0:2], B11 = B[2:4,2:4]
    # C_TL = [C0(2x2) | C1(2x2)] where C0 = C[0:2,0:2], C1 = C[0:2,2:4]
    # C0 = A0*B00 + A1*B10
    # C1 = A0*B01 + A1*B11
    
    # For C0 = A0*B00 + A1*B10:
    # A0*B00: 2x2x2, use Strassen (rank 7)
    # A1*B10: 2x2x2, use Strassen (rank 7)
    # Total 14 for C0.
    
    # For C1 = A0*B01 + A1*B11:
    # A0*B01: 2x2x2, Strassen (rank 7)
    # A1*B11: 2x2x2, Strassen (rank 7)
    # Total 14 for C1.
    
    # Part 2: A_TR*B_BL where A_TR is A[0:2, 4:5] (2x1), B_BL is B[4:5, 0:4] (1x4)
    # C contribution: 2x4, rank 2*1*4 = 8.
    
    # Part 3: A_TL*B_TR where A_TL is A[0:2, 0:4] (2x4), B_TR is B[0:4, 4:7] (4x3)
    # 2x4 x 4x3 = 2x3. Split 4=2+2:
    # A_TL = [A0(2x2)|A1(2x2)], B_TR = [B0(2x3); B1(2x3)]
    # C = A0*B0 + A1*B1. Each 2x2x3, rank 12. Total 24.
    
    # Part 4: A_TR*B_BR where A_TR is A[0:2,4:5] (2x1), B_BR is B[4:5,4:7] (1x3)
    # 2x1 x 1x3 = 2x3, rank 6.
    
    # Part 5: A_BL*B_TL where A_BL is A[2:3,0:4] (1x4), B_TL is B[0:4,0:4] (4x4)
    # 1x4 x 4x4 = 1x4. Split 4=2+2:
    # A_BL = [A0(1x2)|A1(1x2)], B_TL = [[B00(2x2),B01(2x2)],[B10(2x2),B11(2x2)]]
    # C = [A0*B00+A1*B10 | A0*B01+A1*B11]. Four 1x2x2 products, each rank 4. Total 16.
    
    # Part 6: A_BR*B_BL where A_BR is A[2:3,4:5] (1x1), B_BL is B[4:5,0:4] (1x4)
    # 1x1 x 1x4 = 1x4, rank 4.
    
    # Part 7: A_BL*B_TR where A_BL is A[2:3,0:4] (1x4), B_TR is B[0:4,4:7] (4x3)
    # 1x4 x 4x3 = 1x3. Split 4=2+2:
    # A_BL = [A0(1x2)|A1(1x2)], B_TR = [B0(2x3); B1(2x3)]
    # C = A0*B0 + A1*B1. Each 1x2x3, rank 6. Total 12.
    
    # Part 8: A_BR*B_BR where A_BR is A[2:3,4:5] (1x1), B_BR is B[4:5,4:7] (1x3)
    # 1x1 x 1x3 = 1x3, rank 3.
    
    # Total: 14+14+8+24+6+16+4+12+3 = 101.
    
    # Let me implement this. I'll create a helper to add a 2x2x2 Strassen block.
    
    U = [[0]*101 for _ in range(15)]
    V = [[0]*101 for _ in range(35)]
    W = [[0]*101 for _ in range(21)]
    
    col = 0
    
    def add_strassen_2x2x2(a_i0, a_j0, b_j0, b_k0, c_i0, c_k0):
        """Add a Strassen 2x2x2 block.
        A is at rows a_i0..a_i0+1, cols a_j0..a_j0+1
        B is at rows b_j0..b_j0+1, cols b_k0..b_k0+1
        C is at rows c_i0..c_i0+1, cols c_k0..c_k0+1
        """
        nonlocal col
        # M0 = A00*B00
        U[a_i0*5 + a_j0][col] += 1
        V[b_j0*7 + b_k0][col] += 1
        W[(c_k0)*3 + (c_i0)][col] += 1  # C00
        col += 1
        # M1 = (A00+A10)*B10
        U[a_i0*5 + a_j0][col] += 1
        U[(a_i0+1)*5 + a_j0][col] += 1
        V[(b_j0+1)*7 + b_k0][col] += 1
        W[(c_k0)*3 + (c_i0+1)][col] += 1  # C10
        col += 1
        # M2 = A10*B01
        U[(a_i0+1)*5 + a_j0][col] += 1
        V[b_j0*7 + (b_k0+1)][col] += 1
        W[(c_k0+1)*3 + (c_i0+1)][col] += 1  # C11
        col += 1
        # M3 = A01*B11
        U[a_i0*5 + (a_j0+1)][col] += 1
        V[(b_j0+1)*7 + (b_k0+1)][col] += 1
        W[(c_k0+1)*3 + (c_i0)][col] += 1  # C01
        W[(c_k0+1)*3 + (c_i0+1)][col] += 1  # C11
        col += 1
        # M4 = (A01+A11)*B00
        U[a_i0*5 + (a_j0+1)][col] += 1
        U[(a_i0+1)*5 + (a_j0+1)][col] += 1
        V[b_j0*7 + b_k0][col] += 1
        W[(c_k0+1)*3 + (c_i0)][col] += 1  # C01
        col += 1
        # M5 = (A00-A11)*B11
        U[a_i0*5 + a_j0][col] += 1
        U[(a_i0+1)*5 + (a_j0+1)][col] += -1
        V[(b_j0+1)*7 + (b_k0+1)][col] += 1
        W[(c_k0)*3 + (c_i0)][col] += 1  # C00
        W[(c_k0+1)*3 + (c_i0+1)][col] += -1  # C11
        col += 1
        # M6 = (A10-A01)*B01
        U[(a_i0+1)*5 + a_j0][col] += 1
        U[a_i0*5 + (a_j0+1)][col] += -1
        V[b_j0*7 + (b_k0+1)][col] += 1
        W[(c_k0)*3 + (c_i0)][col] += 1  # C00
        W[(c_k0+1)*3 + (c_i0+1)][col] += 1  # C11
        col += 1
    
    def add_schoolbook_2x2x2(a_i0, a_j0, b_j0, b_k0, c_i0, c_k0):
        """Add a schoolbook 2x2x2 block (rank 8)."""
        nonlocal col
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    U[(a_i0+di)*5 + (a_j0+dj)][col] += 1
                    V[(b_j0+dj)*7 + (b_k0+dk)][col] += 1
                    W[(c_k0+dk)*3 + (c_i0+di)][col] += 1
                    col += 1
    
    def add_rect_schoolbook(a_i0, a_j0, a_rows, a_cols, b_j0, b_k0, b_rows, b_cols, c_i0, c_k0):
        """Add a schoolbook rectangular block."""
        nonlocal col
        for di in range(a_rows):
            for dj in range(a_cols):
                for dk in range(b_cols):
                    U[(a_i0+di)*5 + (a_j0+dj)][col] += 1
                    V[(b_j0+dj)*7 + (b_k0+dk)][col] += 1
                    W[(c_k0+dk)*3 + (c_i0+di)][col] += 1
                    col += 1
    
    # Part 1a: C0 = A0*B00 (2x2x2 Strassen)
    # A0 = A[0:2, 0:2], B00 = B[0:2, 0:2], C0 = C[0:2, 0:2]
    add_strassen_2x2x2(0, 0, 0, 0, 0, 0)
    
    # Part 1b: C0 += A1*B10 (2x2x2 Strassen)
    # A1 = A[0:2, 2:4], B10 = B[2:4, 0:2], C0 = C[0:2, 0:2]
    add_strassen_2x2x2(0, 2, 2, 0, 0, 0)
    
    # Part 1c: C1 = A0*B01 (2x2x2 Strassen)
    # A0 = A[0:2, 0:2], B01 = B[0:2, 2:4], C1 = C[0:2, 2:4]
    add_strassen_2x2x2(0, 0, 0, 2, 0, 2)
    
    # Part 1d: C1 += A1*B11 (2x2x2 Strassen)
    # A1 = A[0:2, 2:4], B11 = B[2:4, 2:4], C1 = C[0:2, 2:4]
    add_strassen_2x2x2(0, 2, 2, 2, 0, 2)
    
    # Part 2: A_TR*B_BL: A[0:2, 4:5] * B[4:5, 0:4] -> C[0:2, 0:4]
    add_rect_schoolbook(0, 4, 2, 1, 4, 0, 1, 4, 0, 0)
    
    # Part 3: A_TL*B_TR: A[0:2, 0:4] * B[0:4, 4:7] -> C[0:2, 4:7]
    # Split: A0=A[0:2,0:2], A1=A[0:2,2:4], B0=B[0:2,4:7], B1=B[2:4,4:7]
    # C = A0*B0 + A1*B1, each 2x2x3
    add_rect_schoolbook(0, 0, 2, 2, 0, 4, 2, 3, 0, 4)
    add_rect_schoolbook(0, 2, 2, 2, 2, 4, 2, 3, 0, 4)
    
    # Part 4: A_TR*B_BR: A[0:2, 4:5] * B[4:5, 4:7] -> C[0:2, 4:7]
    add_rect_schoolbook(0, 4, 2, 1, 4, 4, 1, 3, 0, 4)
    
    # Part 5: A_BL*B_TL: A[2:3, 0:4] * B[0:4, 0:4] -> C[2:3, 0:4]
    # Split: A0=A[2:3,0:2], A1=A[2:3,2:4], B00=B[0:2,0:2], B01=B[0:2,2:4], B10=B[2:4,0:2], B11=B[2:4,2:4]
    # C = [A0*B00+A1*B10 | A0*B01+A1*B11]
    add_rect_schoolbook(2, 0, 1, 2, 0, 0, 2, 2, 2, 0)
    add_rect_schoolbook(2, 2, 1, 2, 2, 0, 2, 2, 2, 0)
    add_rect_schoolbook(2, 0, 1, 2, 0, 2, 2, 2, 2, 2)
    add_rect_schoolbook(2, 2, 1, 2, 2, 2, 2, 2, 2, 2)
    
    # Part 6: A_BR*B_BL: A[2:3, 4:5] * B[4:5, 0:4] -> C[2:3, 0:4]
    add_rect_schoolbook(2, 4, 1, 1, 4, 0, 1, 4, 2, 0)
    
    # Part 7: A_BL*B_TR: A[2:3, 0:4] * B[0:4, 4:7] -> C[2:3, 4:7]
    # Split: A0=A[2:3,0:2], A1=A[2:3,2:4], B0=B[0:2,4:7], B1=B[2:4,4:7]
    add_rect_schoolbook(2, 0, 1, 2, 0, 4, 2, 3, 2, 4)
    add_rect_schoolbook(2, 2, 1, 2, 2, 4, 2, 3, 2, 4)
    
    # Part 8: A_BR*B_BR: A[2:3, 4:5] * B[4:5, 4:7] -> C[2:3, 4:7]
    add_rect_schoolbook(2, 4, 1, 1, 4, 4, 1, 3, 2, 4)
    
    # col should be 101
    assert col == 101, f"Expected 101, got {col}"
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
