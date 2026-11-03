import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=7
    
    # Helper: Get Strassen 2x2x2 U, V, W
    def get_strassen_2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Col 0: M1 = (a+d)(e+h)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[1, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # Col 1: M2 = (c+d)e
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[0, 1] = -1; W[2, 1] = 1
        
        # Col 2: M3 = a(f-h)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        # Col 3: M4 = d(g-e)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = -1; W[1, 3] = 1
        
        # Col 4: M5 = (a+b)h
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = 1; W[2, 4] = 1
        
        # Col 5: M6 = (c-a)(e+f)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # Col 6: M7 = (b-d)(g+h)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = -1
        
        return U, V, W

    U_2, V_2, W_2 = get_strassen_2x2()
    
    # Component 1: 4x4x4 Strassen (Rank 49)
    # B cols 0-3
    U1 = np.zeros((16, 49))
    V1 = np.zeros((16, 49)) # m*p = 4*4 = 16
    W1 = np.zeros((16, 49)) # p*n = 4*4 = 16
    
    # Strassen blocks for 4x4x4
    # A blocks: 00, 01, 10, 11 (indices in U: 0-3, 4-7, 8-11, 12-15)
    # B blocks: 00, 01, 10, 11 (indices in V: 0-3, 4-7, 8-11, 12-15)
    # C blocks: 00, 01, 10, 11 (indices in W: 0-3, 4-7, 8-11, 12-15)
    
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    
    # We will iterate over the 7 products
    # For each product, we use the 2x2 Strassen decomposition
    # We need to map the local 2x2 indices to the global 4x4 indices
    
    # Local A indices: 0,1,2,3 correspond to rows/cols of the 2x2 block
    # Global A indices:
    # Block 00: 0,1,4,5
    # Block 01: 2,3,6,7
    # Block 10: 8,9,12,13
    # Block 11: 10,11,14,15
    
    # Local B indices: 0,1,2,3
    # Global B indices:
    # Block 00: 0,1,4,5
    # Block 01: 2,3,6,7
    # Block 10: 8,9,12,13
    # Block 11: 10,11,14,15
    
    # Local C indices: 0,1,2,3
    # Global C indices (W):
    # Block 00: 0,1,4,5
    # Block 01: 2,3,6,7
    # Block 10: 8,9,12,13
    # Block 11: 10,11,14,15
    
    # However, W is indexed by k*n+i.
    # For 4x4x4, k is 0..3.
    # W index = k*4 + i.
    # Block 00 (k=0): i=0,1 -> W indices 0,1. i=2,3 -> W indices 4,5? No.
    # C is 4x4. C[i, k].
    # W index = k*4 + i.
    # Block 00 of C (rows 0-1, cols 0-1):
    # i=0, k=0 -> 0
    # i=1, k=0 -> 1
    # i=0, k=1 -> 4
    # i=1, k=1 -> 5
    # So Block 00 of C maps to W indices 0,1,4,5.
    # Block 01 of C (rows 0-1, cols 2-3):
    # i=0, k=2 -> 8
    # i=1, k=2 -> 9
    # i=0, k=3 -> 12
    # i=1, k=3 -> 13
    # So Block 01 maps to 8,9,12,13.
    # Block 10 of C (rows 2-3, cols 0-1):
    # i=2, k=0 -> 2
    # i=3, k=0 -> 3
    # i=2, k=1 -> 6
    # i=3, k=1 -> 7
    # So Block 10 maps to 2,3,6,7.
    # Block 11 of C (rows 2-3, cols 2-3):
    # i=2, k=2 -> 10
    # i=3, k=2 -> 11
    # i=2, k=3 -> 14
    # i=3, k=3 -> 15
    # So Block 11 maps to 10,11,14,15.

    # Let's define the mappings
    # A Block 00: [0, 1, 4, 5]
    # A Block 01: [2, 3, 6, 7]
    # A Block 10: [8, 9, 12, 13]
    # A Block 11: [10, 11, 14, 15]
    
    # B Block 00: [0, 1, 4, 5]
    # B Block 01: [2, 3, 6, 7]
    # B Block 10: [8, 9, 12, 13]
    # B Block 11: [10, 11, 14, 15]
    
    # W Block 00: [0, 1, 4, 5]
    # W Block 01: [8, 9, 12, 13]
    # W Block 10: [2, 3, 6, 7]
    # W Block 11: [10, 11, 14, 15]
    
    # Define the 7 products
    # Each product is defined by (A_block1, A_block2, sign1, sign2, B_block1, B_block2, sign3, sign4)
    # Actually, it's easier to just sum the contributions.
    
    # We will create a list of (A_idx, B_idx, W_idx, coeff) for each of the 7 Strassen steps
    # Then for each step, we expand into 7 sub-steps using U_2, V_2, W_2.
    
    # Strassen Steps:
    # 1. (A00+A11) * (B00+B11)
    # 2. (A10+A11) * B00
    # 3. A00 * (B01-B11)
    # 4. A11 * (B10-B00)
    # 5. (A00+A01) * B11
    # 6. (A10-A00) * (B00+B01)
    # 7. (A01-A11) * (B10+B11)
    
    # C = 
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M4
    # C10 = M2 + M5
    # C11 = M1 - M2 + M3 + M6
    
    # We will build U1, V1, W1 by iterating through the 7 main products and the 7 sub-products.
    
    main_products = [
        # (A_blocks, B_blocks)
        # Each A_block is a list of (block_id, coeff)
        # A_block_id: 0=00, 1=01, 2=10, 3=11
        # B_block_id: 0=00, 1=01, 2=10, 3=11
        # W_block_id: 0=00, 1=01, 2=10, 3=11
        # The output of this product contributes to C blocks with coefficients.
        
        # This is getting complex to manage manually in code without a helper.
        # I will write a helper to add a 2x2x2 product to the global matrices.
        
    ]
    
    # Helper to add a 2x2x2 product
    # a_blocks: list of (block_id, coeff) for A
    # b_blocks: list of (block_id, coeff) for B
    # w_blocks: list of (block_id, coeff) for C
    # offset: column offset in U1, V1, W1
    
    def add_2x2_product(U_g, V_g, W_g, a_blocks, b_blocks, w_blocks, offset):
        # a_blocks, b_blocks, w_blocks are lists of (block_id, coeff)
        # We need to expand this into the 7 Strassen sub-products.
        # But wait, the Strassen decomposition is for the multiplication of two 2x2 matrices.
        # Here, the "2x2 matrices" are themselves linear combinations of blocks.
        # This is not how Strassen works. Strassen works on the blocks directly.
        
        # Correct approach:
        # The 4x4x4 Strassen decomposition consists of 7 products of 2x2 blocks.
        # Product 1: P1 = (A00+A11) * (B00+B11)
        # P1 is a 2x2 matrix.
        # We compute P1 using the 2x2 Strassen decomposition.
        # So for P1, we add 7 columns to U1, V1, W1.
        # In these 7 columns, the A entries come from A00 and A11.
        # The B entries come from B00 and B11.
        # The W entries go to C00 and C11 (with coeffs from Strassen W).
        
        # So, for each of the 7 main products (P1..P7):
        # 1. Identify the source A blocks and B blocks.
        # 2. For each of the 7 sub-products of the 2x2 Strassen:
        #    - Add contributions to U from the A blocks.
        #    - Add contributions to V from the B blocks.
        #    - Add contributions to W from the C blocks (which depend on which C block P_k contributes to).
        
        # Let's do this.
        
        pass

