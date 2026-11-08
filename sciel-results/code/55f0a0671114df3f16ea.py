import numpy as np
import time

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    # Rank 98 decomposition using Strassen for 4x4x4 twice
    n, m, p = DIMENSIONS
    
    # Helper: Generate Strassen factors for 2x2x2
    # Returns U (4x7), V (4x7), W (4x7)
    def strassen_2x2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A11+A22)(B11+B22)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (A21+A22)B11
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1
        
        # M3 = A11(B12-B22)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1
        
        # M4 = A22(B21-B11)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[2, 3] = 1; W[3, 3] = 1
        
        # M5 = (A11+A12)B22
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # M6 = (A21-A11)(B11+B12)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7 = (A12-A22)(B21+B22)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1; W[1, 6] = 1
        
        return U, V, W

    # Helper: Generate factors for 4x4x4 using tensor product of Strassen
    # The 4x4 matrix is viewed as 2x2 blocks of 2x2 matrices.
    # The rank is 7*7 = 49.
    def strassen_4x4x4():
        U2, V2, W2 = strassen_2x2x2()
        rank = 49
        U4 = np.zeros((16, rank))
        V4 = np.zeros((16, rank))
        W4 = np.zeros((16, rank))
        
        # Map block indices to global indices
        # A is 4x4. Block (I, J) where I, J in {0,1}.
        # Global index for A[I*2+i, J*2+j] is (I*2+i)*4 + (J*2+j).
        # In U4, the row index corresponds to A's global index.
        # In V4, the row index corresponds to B's global index (for 4x4, m=4, p=4).
        # In W4, the row index corresponds to C's global index (for 4x4, n=4, p=4).
        # W convention: W[k*n+i]. For 4x4, n=4. Index k*4+i.
        
        for i_block in range(2):
            for j_block in range(2):
                for k_block in range(2):
                    # This loop structure is not quite right for the tensor product.
                    # The tensor product of two 2x2x2 Strassen decompositions
                    # means we have 7 terms for the "outer" block multiplication
                    # and 7 terms for the "inner" block multiplication.
                    # Total 49 terms.
                    pass
                    
        # Correct construction:
        # The 4x4x4 tensor is the tensor product of the 2x2x2 block tensor and the 2x2x2 element tensor.
        # Rank is 7*7=49.
        # Let the outer Strassen indices be (a, b, c) in {0..6}
        # Let the inner Strassen indices be (d, e, f) in {0..6}
        # The term index is t = a*7 + d.
        
        for a in range(7):
            for d in range(7):
                t = a * 7 + d
                # Outer block indices I, J, K determined by a
                # Inner element indices i, j, k determined by d
                
                # We need to know which blocks are involved in term 'a'
                # And which elements within those blocks are involved in term 'd'
                
                # This requires mapping the Strassen terms to block indices.
                # Let's precompute the block indices for each Strassen term.
                pass
                
        # Let's define the block indices for the 7 Strassen terms.
        # Term 0: M1. Blocks: (0,0)+(1,1) for A, (0,0)+(1,1) for B, (0,0)+(1,1) for C
        # This is complex to hardcode manually in the loop.
        # Alternative: Use the explicit U, V, W matrices for 4x4x4.
        
        # Let's construct U4, V4, W4 by iterating over the 49 terms.
        # For each term t, we have a linear form in A, a linear form in B, and a linear form in C.
        
        # Let's use a different approach: build the 4x4x4 factors from the 2x2x2 factors.
        # The 4x4 matrix A can be written as [[A00, A01], [A10, A11]] where Ai are 2x2.
        # The Strassen decomposition for blocks gives 7 products of block linear forms.
        # Each block product is a 2x2x2 multiplication, which has rank 7.
        # So we have 7 * 7 = 49 scalar products.
        
        # Let's define the block linear forms for A, B, C.
        # For each of the 7 outer terms, we have coefficients for the 4 blocks of A, B, C.
        
        # Outer Term 0: M1 = (A00+A11)(B00+B11) -> C00, C11
        # Outer Term 1: M2 = (A10+A11)B00 -> C10
        # Outer Term 2: M3 = A00(B01-B11) -> C01
        # Outer Term 3: M4 = A11(B10-B00) -> C10, C11
        # Outer Term 4: M5 = (A00+A01)B11 -> C00, C01
        # Outer Term 5: M6 = (A10-A00)(B00+B01) -> C11
        # Outer Term 6: M7 = (A01-A11)(B10+B11) -> C00, C01
        
        # This is getting complicated to map to the exact scalar indices.
        # Let's stick to the known result: Rank 98 is achievable.
        # I will construct the 49-term decomposition for 4x4x4 by iterating
        # through the 7 outer terms and 7 inner terms.
        
        # For simplicity and correctness, I will use the explicit formula for the 49 terms.
        # This is a standard result.
        
        # Let's define the indices for the 49 terms.
        # U4 has 16 rows, 49 cols.
        # V4 has 16 rows, 49 cols.
        # W4 has 16 rows, 49 cols.
        
        # I will generate these by iterating over the 7 outer terms and 7 inner terms.
        # For each outer term a, we have a set of blocks involved in A, B, C.
        # For each inner term d, we have a set of elements involved in the 2x2 blocks.
        
        # Let's define the block involvement for the 7 outer terms.
        # outer_A[a] = list of (block_idx, coeff)
        # outer_B[a] = list of (block_idx, coeff)
        # outer_C[a] = list of (block_idx, coeff)
        
        outer_A = [
            [(0, 1), (3, 1)],      # M1: A00+A11
            [(2, 1), (3, 1)],      # M2: A10+A11
            [(0, 1)],              # M3: A00
            [(3, 1)],              # M4: A11
            [(0, 1), (1, 1)],      # M5: A00+A01
            [(2, 1), (0, -1)],     # M6: A10-A00
            [(1, 1), (3, -1)],     # M7: A01-A11
        ]
        outer_B = [
            [(0, 1), (3, 1)],      # M1: B00+B11
            [(0, 1)],              # M2: B00
            [(1, 1), (3, -1)],     # M3: B01-B11
            [(2, 1), (0, -1)],     # M4: B10-B00
            [(3, 1)],              # M5: B11
            [(0, 1), (1, 1)],      # M6: B00+B01
            [(2, 1), (3, 1)],      # M7: B10+B11
        ]
        outer_C = [
            [(0, 1), (3, 1)],      # M1: C00+C11
            [(2, 1)],              # M2: C10
            [(1, 1)],              # M3: C01
            [(2, 1), (3, 1)],      # M4: C10+C11
            [(0, -1), (1, 1)],     # M5: -C00+C01
            [(3, 1)],              # M6: C11
            [(0, 1), (1, 1)],      # M7: C00+C01
        ]
        
        # Inner terms are the 7 Strassen terms for 2x2x2.
        # inner_U[d], inner_V[d], inner_W[d] are 4x1 vectors (for the 2x2 block).
        # We need to map the 2x2 block indices to the 4x4 global indices.
        # Block (I, J) of A corresponds to rows I*2..I*2+1, cols J*2..J*2+1.
        # Global row index for A is i*4+j.
        # For a block (I, J) and inner element (i2, j2) in 2x2:
        # Global row = (I*2+i2)*4 + (J*2+j2).
        
        U2, V2, W2 = strassen_2x2x2()
        
        for a in range(7):
            for d in range(7):
                t = a * 7 + d
                # Accumulate contributions to U4, V4, W4
                # For U4:
                for (block_idx, coeff) in outer_A[a]:
                    I = block_idx // 2
                    J = block_idx % 2
                    for i2 in range(2):
                        for j2 in range(2):
                            u2 = U2[i2*2+j2, d]
                            if u2 != 0:
                                global_row = (I*2+i2)*4 + (J*2+j2)
                                U4[global_row, t] += coeff * u2
                                
                # For V4:
                for (block_idx, coeff) in outer_B[a]:
                    J = block_idx // 2
                    K = block_idx % 2
                    for j2 in range(2):
                        for k2 in range(2):
                            v2 = V2[j2*2+k2, d]
                            if v2 != 0:
                                global_row = (J*2+j2)*4 + (K*2+k2)
                                V4[global_row, t] += coeff * v2
                                
                # For W4:
                for (block_idx, coeff) in outer_C[a]:
                    I = block_idx // 2
                    K = block_idx % 2
                    for i2 in range(2):
                        for k2 in range(2):
                            w2 = W2[i2*2+k2, d]
                            if w2 != 0:
                                global_row = (K*2+k2)*4 + (I*2+i2) # W[k*n+i] -> k*4+i
                                W4[global_row, t] += coeff * w2
                                
        return U4, V4, W4

    # Now construct the 4x4x8 decomposition
    # U (16x98), V (32x98), W (32x98)
    U4, V4, W4 = strassen_4x4x4()
    
    U = np.zeros((16, 98))
    V = np.zeros((32, 98))
    W = np.zeros((32, 98))
    
    # First 49 terms: A * B_L
    # B_L is cols 0-3 of B.
    # V4 has rows 0-15 corresponding to B_L (4x4).
    # In the full V (32x98), rows 0-15 correspond to B_L, rows 16-31 correspond to B_R.
    # So for the first 49 terms, we use V4 in rows 0-15.
    U[:, 0:49] = U4
    V[0:16, 0:49] = V4
    W[0:16, 0:49] = W4 # W rows 0-15 correspond to C_L (cols 0-3)
    
    # Next 49 terms: A * B_R
    # B_R is cols 4-7 of B.
    # In the full V, rows 16-31 correspond to B_R.
    # So for the next 49 terms, we use V4 in rows 16-31.
    U[:, 49:98] = U4
    V[16:32, 49:98] = V4
    W[16:32, 49:98] = W4 # W rows 16-31 correspond to C_R (cols 4-7)
    
    # Convert to lists of lists for the interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
