"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Helper to generate Strassen decomposition for 2x2x2
    # Returns U, V, W of shape (4, 7)
    def get_strassen_2x2x2():
        # Indices: i, j in 0..1, k in 0..1
        # U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1 = (A11+A22)(B11+B22)
        # A11 idx 0, A22 idx 3. B11 idx 0, B22 idx 3. C11 idx 0, C22 idx 3? No, W is output.
        # W[k*2+i]. C11 is k=0, i=0 -> idx 0. C12 is k=0, i=1 -> idx 1.
        # C21 is k=1, i=0 -> idx 2. C22 is k=1, i=1 -> idx 3.
        
        # M1
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # M2 = (A21+A22)B11
        # A21 idx 2, A22 idx 3. B11 idx 0.
        # C21 (idx 2) += M2, C22 (idx 3) += M2
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = 1
        
        # M3 = A11(B12-B22)
        # A11 idx 0. B12 idx 1, B22 idx 3.
        # C12 (idx 1) += M3, C22 (idx 3) += M3
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = 1
        
        # M4 = A22(B21-B11)
        # A22 idx 3. B21 idx 2, B11 idx 0.
        # C11 (idx 0) += M4, C21 (idx 2) += M4
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1
        
        # M5 = (A11+A12)B22
        # A11 idx 0, A12 idx 1. B22 idx 3.
        # C11 (idx 0) -= M5, C12 (idx 1) += M5
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1
        
        # M6 = (A21-A11)(B11+B12)
        # A21 idx 2, A11 idx 0. B11 idx 0, B12 idx 1.
        # C22 (idx 3) += M6
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # M7 = (A12-A22)(B21+B22)
        # A12 idx 1, A22 idx 3. B21 idx 2, B22 idx 3.
        # C11 (idx 0) += M7
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1
        
        return U, V, W

    # Generate 4x4x4 decomposition using 7 Strassen 2x2x2 blocks
    # Rank 49
    U4 = [[0]*49 for _ in range(16)]
    V4 = [[0]*49 for _ in range(16)]
    W4 = [[0]*49 for _ in range(16)]
    
    S_U, S_V, S_W = get_strassen_2x2x2()
    
    # The 7 blocks of 4x4 matrix multiplication
    # Each block is a 2x2 matrix multiplication
    # Block (bi, bj) of C is sum over (di, dj) of A(bi,di) * B(dj,bj)
    # Strassen combines these.
    # We map the 7 Strassen multiplications to the 4x4 blocks.
    # Let the 4x4 matrix be divided into 2x2 blocks of 2x2 matrices.
    # A = [[A00, A01], [A10, A11]] where Aij are 2x2.
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    
    # Strassen M1..M7 operate on these 2x2 blocks.
    # Each Mx is a 2x2 matrix product.
    # We use the 2x2x2 Strassen decomposition for each Mx.
    
    # Define which 2x2 sub-blocks of A and B are used in each Mx
    # and with what signs.
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Output C blocks:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We iterate over the 7 Strassen multiplications (0..6)
    # For each, we iterate over the 7 Strassen multiplications of the 2x2x2 core (0..6)
    # Total 49 terms.
    
    # Mapping of 4x4 indices to 2x2 block indices
    # A[i,j] where i,j in 0..3.
    # Block row bi = i//2, block col bj = j//2.
    # Sub-row di = i%2, sub-col dj = j%2.
    
    # For a given Strassen multiplication Mx (x in 0..6), it computes a product of two 2x2 matrices.
    # Let the input 2x2 matrices be P and Q.
    # P is formed from A blocks, Q from B blocks.
    # The result is a 2x2 matrix R.
    # R is added to specific C blocks with specific coefficients.
    
    # We define the structure of each Mx in terms of A-blocks and B-blocks.
    # A-block indices (br, bc) where br, bc in 0..1.
    # B-block indices (br, bc) where br, bc in 0..1.
    
    # M1: P = A00+A11, Q = B00+B11.
    # P is 2x2. Its elements are sums of A sub-blocks.
    # This is getting complex to map directly to indices.
    # Easier: Use the known rank 49 decomposition explicitly.
    
    # I will use a hardcoded rank 49 decomposition for 4x4x4 if I can find one, 
    # or generate it via the recursive Strassen construction carefully.
    
    # Let's try the recursive construction properly.
    # The tensor is T_4.
    # T_4 = sum_{x=0}^6 T_{Mx} where T_{Mx} is the tensor for the x-th Strassen multiplication.
    # Each T_{Mx} is a 4x4x4 tensor that is sparse, effectively 2x2x2 in the block structure.
    # The block structure means that for a fixed block coordinate, the sub-tensor is 2x2x2.
    
    # Let's define the 2x2x2 Strassen tensor indices.
    # U2, V2, W2 are 4x7.
    
    # For Mx, we have:
    # U_Mx[i,j, r] = sum_{(br,bc, s) in A_terms[x]} coeff * U2[i_sub, j_sub, s] ?? No.
    # U_Mx depends on A.
    # A is 4x4. Mx uses specific 2x2 sub-matrices of A.
    # Let A_Mx be the 2x2 matrix formed by A.
    # Let B_Mx be the 2x2 matrix formed by B.
    # The multiplication is A_Mx * B_Mx.
    # The result is a 2x2 matrix C_Mx.
    # C_Mx is placed into the 4x4 C matrix at specific block locations with coefficients.
    
    # We can build U4, V4, W4 by iterating over the 7 Mx's and the 7 internal Strassen terms.
    
    # For a specific Mx (x in 0..6):
    # 1. Identify the 2x2 sub-matrices of A and B used.
    # 2. For each of the 7 internal Strassen terms (s in 0..6):
    #    - The term contributes to U, V, W.
    #    - U entries correspond to A elements.
    #    - V entries correspond to B elements.
    #    - W entries correspond to C elements.
    
    # Let's define the A and B block usage for each Mx.
    # A is indexed by (br, bc) in 0..1.
    # B is indexed by (br, bc) in 0..1.
    
    # M1: A terms: (0,0), (1,1). B terms: (0,0), (1,1).
    #      C output: (0,0) with +1, (1,1) with +1? No, C00 and C11 get contributions.
    #      Actually, C00 += M1, C11 += M1.
    #      Wait, C00 is block (0,0), C11 is block (1,1).
    #      So W entries for block (0,0) and (1,1) are affected.
    
    # This is too error-prone to do on the fly. I will use the schoolbook rank 64 for 4x4x4 
    # if I can't be sure about the Strassen mapping, but 49 is much better.
    # However, I can just use the fact that the rank 49 decomposition exists and is standard.
    # I will hardcode the rank 49 decomposition for 4x4x4 from a known source or generate it 
    # using a reliable recursive function.
    
    # Let's try to generate it recursively.
    def generate_strassen_4x4x4():
        # Returns U, V, W of shape (16, 49)
        U = [[0]*49 for _ in range(16)]
        V = [[0]*49 for _ in range(16)]
        W = [[0]*49 for _ in range(16)]
        
        S_U, S_V, S_W = get_strassen_2x2x2()
        
        # Define the 7 Strassen multiplications for 4x4
        # Each is defined by:
        # - A_blocks: list of (block_r, block_c, sign)
        # - B_blocks: list of (block_r, block_c, sign)
        # - C_blocks: list of (block_r, block_c, sign)
        
        # M1 = (A00+A11)(B00+B11) -> C00, C11
        # M2 = (A10+A11)B00 -> C10, C11
        # M3 = A00(B01-B11) -> C01, C11
        # M4 = A11(B10-B00) -> C00, C10
        # M5 = (A00+A01)B11 -> C00 (-1), C01 (+1)
        # M6 = (A10-A00)(B00+B01) -> C11
        # M7 = (A01-A11)(B10+B11) -> C00
        
        # Note: The C output of a 2x2 multiplication is a 2x2 matrix.
        # The block (br, bc) of C receives the entire 2x2 result?
        # No, the 4x4 C is composed of 2x2 blocks.
        # The result of Mx is a 2x2 matrix.
        # It is added to specific 2x2 blocks of C.
        
        # Let's define the mapping for each Mx.
        # x: 0..6
        # A_blocks[x]: list of (br, bc, sign)
        # B_blocks[x]: list of (br, bc, sign)
        # C_blocks[x]: list of (br, bc, sign)
        
        A_blocks = [
            [(0,0,1), (1,1,1)],      # M1
            [(1,0,1), (1,1,1)],      # M2
            [(0,0,1)],               # M3
            [(1,1,1)],               # M4
            [(0,0,1), (0,1,1)],      # M5
            [(1,0,1), (0,0,-1)],     # M6
            [(0,1,1), (1,1,-1)]      # M7
        ]
        
        B_blocks = [
            [(0,0,1), (1,1,1)],      # M1
            [(0,0,1)],               # M2
            [(0,1,1), (1,1,-1)],     # M3
            [(1,0,1), (0,0,-1)],     # M4
            [(1,1,1)],               # M5
            [(0,0,1), (0,1,1)],      # M6
            [(1,0,1), (1,1,1)]       # M7
        ]
        
        C_blocks = [
            [(0,0,1), (1,1,1)],      # M1
            [(1,0,1), (1,1,1)],      # M2
            [(0,1,1), (1,1,1)],      # M3
            [(0,0,1), (1,0,1)],      # M4
            [(0,0,-1), (0,1,1)],     # M5
            [(1,1,1)],               # M6
            [(0,0,1)]                # M7
        ]
        
        for x in range(7):
            for s in range(7):
                r = x * 7 + s
                
                # Process U (A)
                for (br, bc, sign) in A_blocks[x]:
                    # The 2x2 block (br, bc) of A corresponds to rows 2*br..2*br+1, cols 2*bc..2*bc+1
                    # The internal Strassen term s uses specific sub-indices of the 2x2 block.
                    # S_U[i_sub, j_sub, s] is non-zero for i_sub, j_sub in 0..1
                    for i_sub in range(2):
                        for j_sub in range(2):
                            if S_U[i_sub*2+j_sub][s] != 0:
                                i = 2*br + i_sub
                                j = 2*bc + j_sub
                                U[i*4+j][r] += sign * S_U[i_sub*2+j_sub][s]
                
                # Process V (B)
                for (br, bc, sign) in B_blocks[x]:
                    for i_sub in range(2):
                        for j_sub in range(2):
                            if S_V[i_sub*2+j_sub][s] != 0:
                                i = 2*br + i_sub
                                j = 2*bc + j_sub
                                V[i*4+j][r] += sign * S_V[i_sub*2+j_sub][s]
                
                # Process W (C)
                for (br, bc, sign) in C_blocks[x]:
                    for i_sub in range(2):
                        for j_sub in range(2):
                            if S_W[i_sub*2+j_sub][s] != 0:
                                i = 2*br + i_sub
                                j = 2*bc + j_sub
                                W[j*4+i][r] += sign * S_W[i_sub*2+j_sub][s]
        
        return U, V, W

    U4, V4, W4 = generate_strassen_4x4x4()
    
    # Now construct 4x4x8 decomposition
    # Rank 98
    U = [[0]*98 for _ in range(16)]
    V = [[0]*98 for _ in range(32)]
    W = [[0]*98 for _ in range(32)]
    
    # Block 1: k in 0..3
    for i in range(4):
        for j in range(4):
            for r in range(49):
                U[i*4+j][r] = U4[i*4+j][r]
        
    for j in range(4):
        for k in range(4):
            for r in range(49):
                V[j*4+k][r] = V4[j*4+k][r]
        
    for k in range(4):
        for i in range(4):
            for r in range(49):
                W[k*4+i][r] = W4[k*4+i][r]
                
    # Block 2: k in 4..7
    for i in range(4):
        for j in range(4):
            for r in range(49):
                U[i*4+j][r+49] = U4[i*4+j][r]
                
    for j in range(4):
        for k in range(4):
            for r in range(49):
                V[j*4+(k+4)][r+49] = V4[j*4+k][r]
                
    for k in range(4):
        for i in range(4):
            for r in range(49):
                W[(k+4)*4+i][r+49] = W4[k*4+i][r]
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
