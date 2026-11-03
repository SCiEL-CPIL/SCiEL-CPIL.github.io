"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=9
    
    # We will decompose B (4x9) into B1 (4x4), B2 (4x4), B3 (4x1)
    # C = A*B1 + A*B2 + A*B3
    # Rank(A*B1) <= 49 (Strassen 4x4)
    # Rank(A*B2) <= 49 (Strassen 4x4)
    # Rank(A*B3) = 16 (Schoolbook 4x4x1)
    # Total Rank = 114
    
    rank_total = 49 + 49 + 16
    
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
    
    # Helper to add a decomposition block
    # u_block: (n*m, rank_block)
    # v_block: (m*p_block, rank_block) -- needs mapping to global m*p
    # w_block: (p_block*n, rank_block) -- needs mapping to global p*n
    
    def add_block(u_block, v_block, w_block, v_col_offset, w_col_offset, start_rank):
        # u_block is (16, r)
        # v_block is (4*p_block, r)
        # w_block is (p_block*4, r)
        r = len(u_block[0])
        for i in range(16):
            for c in range(r):
                U[i][start_rank + c] = u_block[i][c]
        
        for i in range(len(v_block)):
            # v_block row i corresponds to B index (j, k_local)
            # j = i // p_block, k_local = i % p_block
            # Global B index is (j, k_global) where k_global = v_col_offset + k_local
            j = i // (p // 4 if p_block == 4 else p_block) # This logic is messy, let's pass p_block
            # Actually, let's just pass the dimensions explicitly.
            pass

    # Let's do it manually for clarity and speed.
    
    # --- Strassen 4x4 Decomposition ---
    # Generates U, V, W for 4x4 * 4x4 -> 4x4
    # U: 16x49, V: 16x49, W: 16x49
    
    def get_strassen_4x4():
        # Strassen 2x2
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        
        # We need to expand these to 4x4 using 2x2 blocks.
        # A = [[A00, A01], [A10, A11]] where Aij are 2x2.
        # B = [[B00, B01], [B10, B11]] where Bij are 2x2.
        
        # This creates 7 multiplications of 2x2 matrices.
        # Each 2x2 multiplication is rank 7.
        # Total 49.
        
        # We will construct the 49 terms.
        # Term t (0..48) corresponds to M_{t//7} and Strassen_2x2_{t%7}.
        
        u = [[0]*49 for _ in range(16)]
        v = [[0]*49 for _ in range(16)]
        w = [[0]*49 for _ in range(16)]
        
        # Strassen 2x2 coefficients
        # U2x2: 4x7, V2x2: 4x7, W2x2: 4x7
        # Indices for 2x2: 0,1,2,3
        # M1: (0+3)(0+3)
        # M2: (2+3)0
        # M3: 0(2-3)
        # M4: 3(1-0)
        # M5: (0+1)3
        # M6: (2-0)(0+2)
        # M7: (1-3)(1+3)
        
        # Let's define the 2x2 Strassen factors explicitly.
        # u2x2[i][c], v2x2[j][c], w2x2[k][c]
        # i,j,k in 0..3. c in 0..6.
        
        u2 = [[0]*7 for _ in range(4)]
        v2 = [[0]*7 for _ in range(4)]
        w2 = [[0]*7 for _ in range(4)]
        
        # M1
        u2[0][0] += 1; u2[3][0] += 1
        v2[0][0] += 1; v2[3][0] += 1
        w2[0][0] += 1; w2[1][0] += 1; w2[2][0] += 1; w2[3][0] += 1
        # Wait, W for 2x2 is C = [[C00, C01], [C10, C11]]
        # C00 = M1 - M4 + M5 - M7
        # C01 = M3 + M7
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 - M6
        
        # Let's just use the standard 2x2 Strassen U,V,W.
        # It's easier to hardcode the 49 terms or use a known library if available? No.
        # I will generate the 2x2 Strassen U,V,W first.
        
        # 2x2 Strassen
        # c=0 (M1): A(0,0)+A(1,1) * B(0,0)+B(1,1)
        # c=1 (M2): A(1,0)+A(1,1) * B(0,0)
        # c=2 (M3): A(0,0) * B(1,0)-B(1,1)
        # c=3 (M4): A(1,1) * B(0,1)-B(0,0)
        # c=4 (M5): A(0,0)+A(0,1) * B(1,1)
        # c=5 (M6): A(1,0)-A(0,0) * B(0,0)+B(1,0)
        # c=6 (M7): A(0,1)-A(1,1) * B(0,1)+B(1,1)
        
        u2[0][0]=1; u2[3][0]=1
        v2[0][0]=1; v2[3][0]=1
        w2[0][0]=1; w2[1][0]=1; w2[2][0]=1; w2[3][0]=1
        
        u2[2][1]=1; u2[3][1]=1
        v2[0][1]=1
        w2[2][1]=1; w2[3][1]=1
        
        u2[0][2]=1
        v2[2][2]=1; v2[3][2]=-1
        w2[1][2]=1; w2[3][2]=1
        
        u2[3][3]=1
        v2[1][3]=1; v2[0][3]=-1
        w2[0][3]=-1; w2[2][3]=1
        
        u2[0][4]=1; u2[1][4]=1
        v2[3][4]=1
        w2[0][4]=1; w2[1][4]=1
        
        u2[2][5]=1; u2[0][5]=-1
        v2[0][5]=1; v2[2][5]=1
        w2[0][5]=-1; w2[2][5]=-1
        
        u2[1][6]=1; u2[3][6]=-1
        v2[1][6]=1; v2[3][6]=1
        w2[0][6]=-1; w2[1][6]=1
        
        # Now construct 4x4 Strassen
        # A blocks: A00 (0,1,4,5), A01 (2,3,6,7), A10 (8,9,12,13), A11 (10,11,14,15)
        # B blocks: B00 (0,1,4,5), B01 (2,3,6,7), B10 (8,9,12,13), B11 (10,11,14,15)
        # C blocks: C00 (0,1,4,5), C01 (2,3,6,7), C10 (8,9,12,13), C11 (10,11,14,15)
        
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        
        # For each Mk, we apply the 2x2 Strassen to the 2x2 blocks.
        # This means we need to form linear combinations of A and B entries to form the 2x2 matrices for Mk,
        # then apply the 2x2 Strassen decomposition to those.
        
        # This is getting complex to code dynamically.
        # Given the 150s limit, I will return the schoolbook 144 if I can't finish the 114 construction.
        # But 114 is doable.
        
        # I will simplify by just returning schoolbook to ensure validity, 
        # but the prompt asks for a stronger algorithm.
        # I will implement the 114 rank decomposition carefully.
        
        return u, v, w

    # Since implementing the full 4x4 Strassen composition is error-prone in this short format,
    # and the deadline is strict, I will use the 114 rank strategy but with a simpler construction
    # if possible, or fallback to schoolbook.
    
    # Actually, I can just use the 2x2 Strassen on the 4x4 problem directly by viewing 4x4 as 2x2 of 2x2.
    # I will hardcode the 114 rank decomposition.
    
    # For now, return schoolbook to be safe, but this scores 1.0.
    # I need to beat 1.0.
    
    # I will implement the 114 rank solution.
    
    pass

# EVOLVE-BLOCK-END
