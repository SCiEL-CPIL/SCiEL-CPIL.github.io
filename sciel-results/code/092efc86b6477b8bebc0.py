"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    # Dimensions
    n, m, p = DIMENSIONS
    
    # We will construct a decomposition of rank 57.
    # Split m=5 into 4+1.
    # A = [A4 | A1] (4x4, 4x1)
    # B = [B4; B1] (4x6, 1x6)
    # AB = A4 B4 + A1 B1
    
    # Split p=6 into 4+2.
    # B4 = [B44 | B42] (4x4, 4x2)
    # A4 B4 = A4 B44 + A4 B42
    
    # 1. A4 B44 (4x4x4) -> Rank 49 (Strassen)
    # 2. A4 B42 (4x4x2) -> Rank 2 (Two 4x4x1, each rank 1)
    # 3. A1 B1 (4x1x6) -> Rank 6 (Six 4x1x1, each rank 1)
    
    rank = 49 + 2 + 6
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_offset = 0
    
    # --- Part 1: A4 B44 (4x4x4) using Strassen ---
    # Indices for A4: i in 0..3, j in 0..3. Index in A: i*5 + j.
    # Indices for B44: j in 0..3, k in 0..3. Index in B: j*6 + k.
    # Indices for C44: i in 0..3, k in 0..3. Index in C: k*4 + i.
    
    # Strassen for 4x4 using 2x2 blocks.
    # We implement the 2x2 Strassen decomposition and apply it to the 4x4 case by iterating over 2x2 blocks?
    # No, standard Strassen on 4x4 splits into four 2x2 blocks.
    # The rank is 7 * 7 = 49.
    
    # 2x2 Strassen decomposition:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need to map this to the 4x4 case.
    # A4 is 4x4. Split into A11, A12, A21, A22 (each 2x2).
    # B44 is 4x4. Split into B11, B12, B21, B22 (each 2x2).
    # C44 is 4x4. Split into C11, C12, C21, C22 (each 2x2).
    
    # For each of the 7 Strassen multiplications (e.g., M1), we multiply two 2x2 matrices.
    # Each 2x2 multiplication has rank 7.
    # So we need 7 * 7 = 49 terms.
    
    # Helper to add a 2x2 Strassen decomposition for specific block indices
    def add_2x2_strassen(a_base_i, a_base_j, b_base_j, b_base_k, c_base_k, c_base_i, offset):
        """
        Adds rank-7 decomposition for multiplying 2x2 matrices.
        a_base_i, a_base_j: top-left index of A block in global A (4x5)
        b_base_j, b_base_k: top-left index of B block in global B (5x6)
        c_base_k, c_base_i: top-left index of C block in global C (6x4)
        """
        # 2x2 Strassen factors
        # U indices: (i, j) in 2x2 block. Global A index: (a_base_i + i)*5 + (a_base_j + j)
        # V indices: (j, k) in 2x2 block. Global B index: (b_base_j + j)*6 + (b_base_k + k)
        # W indices: (k, i) in 2x2 block. Global C index: (c_base_k + k)*4 + (c_base_i + i)
        
        # Strassen 2x2 definitions
        # M1: (A11+A22)(B11+B22)
        # M2: (A21+A22)B11
        # M3: A11(B21-B22)
        # M4: A22(B12-B11)
        # M5: (A11+A12)B22
        # M6: (A21-A11)(B11+B12)
        # M7: (A12-A22)(B21+B22)
        
        # We iterate over the 7 Strassen steps
        # For each step, we have a "virtual" 2x2 multiplication.
        # We need to decompose each virtual 2x2 multiplication with rank 7.
        
        # This is complex to nest. 
        # Alternative: Just use schoolbook for the 2x2 parts? Rank 8.
        # 7 * 8 = 56. 56 + 2 + 6 = 64.
        # 64 is still < 120.
        # But 49 is better.
        
        # Let's implement the nested Strassen explicitly.
        # For each of the 7 Strassen terms, we define linear combinations of A and B blocks.
        # Then we multiply the resulting 2x2 matrices.
        
        # This requires generating the U, V, W factors for the 2x2 multiplication.
        
        # 2x2 Multiplication Decomposition (Rank 7)
        # T_ij_k = sum_r U_ir V_jr W_kr
        # For 2x2, indices i,j,k in {0,1}.
        # U (2x7), V (2x7), W (2x7)
        
        # Standard 2x2 Strassen factors:
        # r=0: M1. U11=1, U22=1. V11=1, V22=1. W11=1, W12=0, W21=0, W22=1? No.
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Let's define the 7 rank-1 terms for 2x2 multiplication.
        # Term 0 (M1): A11+A22, B11+B22 -> C11, C22
        # Term 1 (M2): A21+A22, B11 -> C21
        # Term 2 (M3): A11, B21-B22 -> C12
        # Term 3 (M4): A22, B12-B11 -> C11, C21
        # Term 4 (M5): A11+A12, B22 -> C11, C12
        # Term 5 (M6): A21-A11, B11+B12 -> C22
        # Term 6 (M7): A12-A22, B21+B22 -> C11, C22
        
        # We will add these 7 terms to the global U, V, W.
        
        # Map local 2x2 indices (li, lj) to global indices
        def get_a_idx(li, lj):
            return (a_base_i + li) * 5 + (a_base_j + lj)
        
        def get_b_idx(lj, lk):
            return (b_base_j + lj) * 6 + (b_base_k + lk)
        
        def get_c_idx(lk, li):
            return (c_base_k + lk) * 4 + (c_base_i + li)
            
        # Term 0: M1
        # A: 11, 22. B: 11, 22. C: 11, 22.
        U[get_a_idx(0,0)][offset+0] += 1
        U[get_a_idx(1,1)][offset+0] += 1
        V[get_b_idx(0,0)][offset+0] += 1
        V[get_b_idx(1,1)][offset+0] += 1
        W[get_c_idx(0,0)][offset+0] += 1
        W[get_c_idx(1,1)][offset+0] += 1
        
        # Term 1: M2
        # A: 21, 22. B: 11. C: 21.
        U[get_a_idx(1,0)][offset+1] += 1
        U[get_a_idx(1,1)][offset+1] += 1
        V[get_b_idx(0,0)][offset+1] += 1
        W[get_c_idx(0,1)][offset+1] += 1
        
        # Term 2: M3
        # A: 11. B: 21, -22. C: 12.
        U[get_a_idx(0,0)][offset+2] += 1
        V[get_b_idx(1,0)][offset+2] += 1
        V[get_b_idx(1,1)][offset+2] += -1
        W[get_c_idx(1,0)][offset+2] += 1
        
        # Term 3: M4
        # A: 22. B: 12, -11. C: 11, 21.
        U[get_a_idx(1,1)][offset+3] += 1
        V[get_b_idx(0,1)][offset+3] += 1
        V[get_b_idx(0,0)][offset+3] += -1
        W[get_c_idx(0,0)][offset+3] += 1
        W[get_c_idx(0,1)][offset+3] += 1
        
        # Term 4: M5
        # A: 11, 12. B: 22. C: 11, 12.
        U[get_a_idx(0,0)][offset+4] += 1
        U[get_a_idx(0,1)][offset+4] += 1
        V[get_b_idx(1,1)][offset+4] += 1
        W[get_c_idx(0,0)][offset+4] += 1
        W[get_c_idx(1,0)][offset+4] += 1
        
        # Term 5: M6
        # A: 21, -11. B: 11, 12. C: 22.
        U[get_a_idx(1,0)][offset+5] += 1
        U[get_a_idx(0,0)][offset+5] += -1
        V[get_b_idx(0,0)][offset+5] += 1
        V[get_b_idx(0,1)][offset+5] += 1
        W[get_c_idx(1,1)][offset+5] += 1
        
        # Term 6: M7
        # A: 12, -22. B: 21, 22. C: 11, 22.
        U[get_a_idx(0,1)][offset+6] += 1
        U[get_a_idx(1,1)][offset+6] += -1
        V[get_b_idx(1,0)][offset+6] += 1
        V[get_b_idx(1,1)][offset+6] += 1
        W[get_c_idx(0,0)][offset+6] += 1
        W[get_c_idx(1,1)][offset+6] += 1

    # Apply Strassen to the 4x4 block (0,0)
    # A4 is at (0,0) in A (4x4 part).
    # B44 is at (0,0) in B (4x4 part).
    # C44 is at (0,0) in C (4x4 part).
    
    # We need to apply the 2x2 Strassen logic to the 4x4 case.
    # The 4x4 Strassen uses 7 multiplications of 2x2 matrices.
    # Each 2x2 multiplication is decomposed into 7 rank-1 terms.
    # Total 49 terms.
    
    # We call add_2x2_strassen for each of the 7 "virtual" 2x2 multiplications?
    # No, add_2x2_strassen adds 7 terms for ONE 2x2 multiplication.
    # We need to call it 7 times, with different linear combinations of the 4x4 blocks?
    # No, the 4x4 Strassen defines 7 products of 2x2 matrices.
    # Let M1_44 = (A11+A22)(B11+B22).
    # This M1_44 is a 2x2 matrix product.
    # We need to decompose M1_44 with rank 7.
    # So we call add_2x2_strassen for M1_44.
    
    # However, add_2x2_strassen assumes the inputs are simple blocks.
    # Here, the inputs to M1_44 are sums of blocks.
    # This requires expanding the linear combinations.
    
    # This is getting too complex for a quick implementation.
    # Let's use Schoolbook for the 4x4x4 part? Rank 64.
    # Total Rank: 64 + 2 + 6 = 72.
    # 72 is still a good improvement over 120.
    
    # Let's stick to Schoolbook for 4x4x4 to ensure correctness and simplicity.
    # Rank 64 for 4x4x4.
    
    for i in range(4):
        for j in range(4):
            for k in range(4):
                # A index: i*5 + j
                # B index: j*6 + k
                # C index: k*4 + i
                U[i*5 + j][col_offset] = 1
                V[j*6 + k][col_offset] = 1
                W[k*4 + i][col_offset] = 1
                col_offset += 1
                
    # --- Part 2: A4 B42 (4x4x2) ---
    # Rank 2.
    # B42 is columns 4,5 of B (indices 4, 5).
    # A4 is rows 0-3, cols 0-3 of A.
    # C is columns 4,5 of C (indices 4,5).
    
    # Term 1: k=4
    # C_i4 = sum_j A_ij B_j4
    # Rank 1: U=1, V=1, W=1.
    for i in range(4):
        for j in range(4):
            U[i*5 + j][col_offset] = 1
            V[j*6 + 4][col_offset] = 1
            W[4*4 + i][col_offset] = 1
    col_offset += 1
    
    # Term 2: k=5
    for i in range(4):
        for j in range(4):
            U[i*5 + j][col_offset] = 1
            V[j*6 + 5][col_offset] = 1
            W[5*4 + i][col_offset] = 1
    col_offset += 1
    
    # --- Part 3: A1 B1 (4x1x6) ---
    # Rank 6.
    # A1 is column 4 of A (index 4).
    # B1 is row 4 of B (indices 24-29).
    # C is all columns 0-5.
    
    for k in range(6):
        for i in range(4):
            # A index: i*5 + 4
            # B index: 4*6 + k
            # C index: k*4 + i
            U[i*5 + 4][col_offset] = 1
            V[4*6 + k][col_offset] = 1
            W[k*4 + i][col_offset] = 1
        col_offset += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
