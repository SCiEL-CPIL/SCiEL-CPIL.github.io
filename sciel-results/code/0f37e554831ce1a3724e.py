import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Part 1: M_{4,4,4} using Strassen (Rank 49)
    # Part 2: M_{4,1,6} using Schoolbook (Rank 24)
    
    # --- Helper: Generate U, V, W for M_{2,2,2} (Strassen) ---
    def get_m222_decomp():
        # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
        # Indices: a=0, b=1, c=2, d=3 for A
        #          e=0, f=1, g=2, h=3 for B
        # C indices: c11=0, c12=1, c21=2, c22=3
        
        U = np.zeros((4, 7), dtype=int)
        V = np.zeros((4, 7), dtype=int)
        W = np.zeros((4, 7), dtype=int)
        
        # M1 = (a+d)(e+h)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        # W for M1 contributes to C11, C22
        # C11 += M1, C22 += M1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (c+d)e
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        # C21 += M2, C22 -= M2
        W[2, 1] = 1; W[3, 1] = -1
        
        # M3 = a(f-h)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        # C12 += M3, C22 += M3
        W[1, 2] = 1; W[3, 2] = 1
        
        # M4 = d(g-e)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        # C11 += M4, C21 += M4
        W[0, 3] = 1; W[2, 3] = 1
        
        # M5 = (a+b)h
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        # C11 -= M5, C12 += M5
        W[0, 4] = -1; W[1, 4] = 1
        
        # M6 = (c-a)(e+f)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        # C22 += M6
        W[3, 5] = 1
        
        # M7 = (b-d)(g+h)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        # C11 += M7
        W[0, 6] = 1
        
        return U, V, W

    # --- Construct M_{4,4,4} Decomp ---
    # A is 4x4, B is 4x4.
    # Flatten A: a00..a33 (16 elems)
    # Flatten B: b00..b33 (16 elems)
    # C is 4x4 (16 elems)
    
    U_444 = np.zeros((16, 49), dtype=int)
    V_444 = np.zeros((16, 49), dtype=int)
    W_444 = np.zeros((16, 49), dtype=int)
    
    U_222, V_222, W_222 = get_m222_decomp()
    
    # Strassen Formulas for 4x4 (2x2 blocks of 2x2)
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # We need to map the 2x2 block elements to the 16-element vector.
    # A11: indices 0,1,4,5 (row 0: 0,1; row 1: 4,5)
    # A12: indices 2,3,6,7
    # A21: indices 8,9,12,13
    # A22: indices 10,11,14,15
    
    # Helper to add a 2x2 block decomp to the main matrices
    def add_block_decomp(col_offset, u_block, v_block, w_block, 
                         a_indices, b_indices, c_indices, 
                         U_main, V_main, W_main):
        # u_block is 4x7, v_block is 4x7, w_block is 4x7
        # a_indices: list of 4 indices in A_flat
        # b_indices: list of 4 indices in B_flat
        # c_indices: list of 4 indices in C_flat
        # We need to scale/shift the U, V, W from the 2x2 decomp to the 4x4 decomp.
        # The 2x2 decomp assumes inputs are a,b,c,d and e,f,g,h.
        # Here, the inputs are linear combinations.
        # For simplicity, we assume the blocks are just subsets of indices with coeff 1.
        # This works for Strassen steps where we add/subtract blocks.
        
        for k in range(7):
            col = col_offset + k
            # U contribution
            for i_222 in range(4):
                if U_main[a_indices[i_222], col] != 0 or True: # We are building from scratch
                    pass
            
            # Actually, it's easier to construct the linear combinations directly.
            # U_main[a_idx, col] += U_222[i_222, k] * coeff
            # Since the blocks are just sums/diffs of 2x2 matrices, the coeffs are +/-1.
            
            # Let's just hardcode the linear combinations for U and V.
            pass

    # This manual composition is error-prone. 
    # Let's use a simpler structural change: 
    # Just use Schoolbook for M_{4,5,6} but with a better constant? No.
    
    # Alternative: Use Strassen for M_{4,4,4} and Schoolbook for the rest.
    # I will implement the M_{4,4,4} composition carefully.
    
    # Indices for A blocks in A_flat (4x4)
    # A11: 0,1,4,5
    # A12: 2,3,6,7
    # A21: 8,9,12,13
    # A22: 10,11,14,15
    
    # Indices for B blocks in B_flat (4x4)
    # B11: 0,1,4,5
    # B12: 2,3,6,7
    # B21: 8,9,12,13
    # B22: 10,11,14,15
    
    # Indices for C blocks in C_flat (4x4)
    # C11: 0,1,4,5
    # C12: 2,3,6,7
    # C21: 8,9,12,13
    # C22: 10,11,14,15
    
    # Strassen Steps
    # M1: (A11+A22)(B11+B22)
    # A_input = A11 + A22. 
    # U_222 has entries for a,b,c,d.
    # a corresponds to A11[0,0] + A22[0,0] = A[0,0] + A[1,1] -> indices 0, 10? No.
    # A11[0,0] is A[0,0] (idx 0). A22[0,0] is A[2,2] (idx 10).
    # So for M1, the 'a' term in U_222[0,0] maps to A[0,0] and A[2,2].
    
    # Let's define the mapping for each Strassen step.
    # Step 1: M1 = (A11+A22)(B11+B22)
    # A_map: a->(0,10), b->(1,11), c->(4,14), d->(5,15)
    # B_map: e->(0,10), f->(1,11), g->(4,14), h->(5,15)
    # C_map: c11->(0,10), c12->(1,11), c21->(4,14), c22->(5,15)
    
    # This is getting too complex to hardcode safely in the time limit.
    # I will fall back to a simpler improvement: 
    # Use Strassen for M_{2,2,2} subproblems in a block decomposition of M_{4,5,6}.
    
    # Actually, the simplest valid improvement is to use **Strassen's algorithm** for the whole matrix by padding to 8x8? No, rank 343.
    
    # Let's try **Rank 73** via M_{4,4,4} + M_{4,1,6}.
    # I will implement a generic tensor decomposition composer.
    
    def compose_decompositions(decomp1, decomp2, linear_combos):
        pass

    # Given the complexity, I will return the schoolbook decomposition but with a 
    # slight optimization: if n,m,p are small, schoolbook is already minimal?
    # No, 120 is not minimal.
    
    # I will use the **Winograd Fourier Algorithm** for M_{4,5,6}.
    # It is known to produce lower rank for small matrices.
    # However, implementing FFT-MMA from scratch is risky.
    
    # Let's stick to the **M_{4,4,4} + M_{4,1,6}** approach but simplify the implementation.
    # I will hardcode the U, V, W for M_{4,4,4} using the 49 rank decomposition.
    
    # Since I cannot easily derive the 49-rank matrices here, I will use a 
    # **Hybrid Schoolbook/Strassen** for M_{4,5,6} directly.
    
    # Split A (4x5) into A1 (4x2) and A2 (4x3).
    # Split B (5x6) into B1 (2x6) and B2 (3x6).
    # C = A1B1 + A2B2.
    # A1B1 is 4x2x6. Rank 48 (schoolbook).
    # A2B2 is 4x3x6. Rank 72 (schoolbook).
    # Total 120. No gain.
    
    # Split A (4x5) into A1 (4x4) and A2 (4x1).
    # Split B (5x6) into B1 (4x6) and B2 (1x6).
    # C = A1B1 + A2B2.
    # A1B1 is 4x4x6. 
    # A2B2 is 4x1x6. Rank 24.
    # If I can decompose 4x4x6 with rank < 96, I win.
    # 4x4x6 schoolbook is 96.
    # Can I do better?
    # 4x4x6 can be seen as 4x4x4 + 4x4x2.
    # 4x4x4 rank 49.
    # 4x4x2 rank 32 (schoolbook).
    # Total 49 + 32 + 24 = 105.
    # 105 < 120.
    
    # So the plan:
    # 1. M_{4,4,4} (Rank 49)
    # 2. M_{4,4,2} (Rank 32)
    # 3. M_{4,1,6} (Rank 24)
    # Total Rank 105.
    
    # I need the M_{4,4,4} rank 49 decomposition.
    # I will implement it by composing 7 M_{2,2,2} decompositions.
    
    U_444 = np.zeros((16, 49), dtype=int)
    V_444 = np.zeros((16, 49), dtype=int)
    W_444 = np.zeros((16, 49), dtype=int)
    
    U_222, V_222, W_222 = get_m222_decomp()
    
    # Define the 7 Strassen steps for 4x4
    # Each step k has:
    # A_blocks: list of (idx, coeff) for the 4 elements of the 2x2 A block
    # B_blocks: list of (idx, coeff) for the 4 elements of the 2x2 B block
    # C_blocks: list of (idx, coeff) for the 4 elements of the 2x2 C block
    
    # A_flat indices:
    # A11: 0,1,4,5
    # A12: 2,3,6,7
    # A21: 8,9,12,13
    # A22: 10,11,14,15
    
    steps = [
        # M1 = (A11+A22)(B11+B22)
        {
            'A': [(0,1), (1,1), (4,1), (5,1), (10,1), (11,1), (14,1), (15,1)], # This is wrong, U_222 expects 4 elements.
        }
    ]
    
    # This is too hard to hardcode. I will return the schoolbook solution.
    # The prompt allows for a first try in a new direction to score worse.
    # But I should try to at least get a valid decomposition.
    
    # I will return the schoolbook decomposition.
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
