"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal (or near-minimal) exact tensor decomposition for
    multiplying a 4x4 matrix by a 4x9 matrix.
    
    The decomposition is constructed by splitting the 4x9 matrix into three
    4x3 blocks. Each 4x3 multiplication is decomposed into a 2x3 and a 2x3 part.
    The 2x3 multiplication is then decomposed into a 2x2 Strassen part (rank 7)
    and a 2x1 schoolbook part (rank 2), totaling rank 9 per 2x3 block.
    
    Total rank: 3 blocks * 2 (2x3 parts) * 9 (rank per 2x3) = 54.
    Wait, let's re-verify the rank.
    4x4 * 4x3.
    Split 4x4 A into 2x2 blocks: A00, A01, A10, A11.
    Split 4x3 B into 2x3 blocks: B0 (top 2x3), B1 (bottom 2x3).
    
    This structure is getting complex. Let's use a simpler, known lower rank.
    The schoolbook rank is 4*4*9 = 144.
    A simple Strassen-based approach for 4x4x4 has rank 49.
    For 4x4x9, we can use the 4x4x4 decomposition for the first 3 columns (rank 49*3 = 147? No, that's not additive like that if we share factors).
    
    Actually, let's look at the 2x2x2 Strassen decomposition (rank 7).
    The 4x4x4 multiplication can be done with rank 49 (Strassen).
    The 4x4x9 multiplication can be seen as 4x4x(9).
    We can split the 9 columns into 4+4+1.
    Rank(4x4x4) = 49.
    Rank(4x4x1) = 16 (schoolbook, since it's just 16 multiplications).
    Total rank = 49 + 49 + 16 = 114.
    
    Let's implement the Strassen 4x4x4 decomposition.
    It is built recursively from 2x2x2 Strassen.
    """
    n, m, p = DIMENSIONS
    
    # Helper to get Strassen 2x2x2 matrices
    def strassen_2x2x2():
        # Returns U, V, W of shape (4, 7)
        U = np.array([
            [1, 0, 1, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 1]
        ])
        V = np.array([
            [1, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0],
            [1, 0, 0, 0, 0, 1, 0]
        ])
        W = np.array([
            [1, 0, 0, 1, -1, 0, 1],
            [0, 1, 0, 0, 1, 0, 0],
            [0, 0, 1, 1, 0, 0, 0],
            [1, 0, -1, 0, 0, 1, 0]
        ])
        return U, V, W
    
    def strassen_4x4x4():
        # Returns U, V, W of shape (16, 49)
        # Recursive construction
        U2, V2, W2 = strassen_2x2x2()
        
        # The 4x4x4 tensor is composed of 16 2x2x2 blocks.
        # A = [[A00, A01], [A10, A11]]
        # B = [[B00, B01], [B10, B11]]
        # C = A*B
        # C00 = A00*B00 + A01*B10
        # C01 = A00*B01 + A01*B11
        # C10 = A10*B00 + A11*B10
        # C11 = A10*B01 + A11*B11
        
        # We can use the Kronecker product of the 2x2x2 decomposition.
        # But Strassen is not simply Kronecker. We need to map the 2x2 blocks.
        # Actually, the rank of 4x4x4 is 49. This is achieved by applying Strassen to the 2x2 blocks.
        
        # Let's construct it by placing the 2x2x2 decomposition in the right places.
        # This is complex. Let's use a known explicit 4x4x4 rank 49 decomposition.
        # Or, we can just use schoolbook for now if Strassen is too hard to code correctly quickly.
        # Schoolbook rank for 4x4x9 is 144.
        # Is there a simpler improvement?
        
        # Let's try to implement the recursive Strassen properly.
        # The 4x4x4 multiplication tensor T_{4,4,4} can be decomposed using T_{2,2,2}.
        # T_{4,4,4} = T_{2,2,2} \otimes T_{2,2,2} + ...
        # Actually, the standard way is:
        # A_{4x4} = [[A00, A01], [A10, A11]]
        # B_{4x4} = [[B00, B01], [B10, B11]]
        # 
        # We compute 7 products for each 2x2 block? No.
        # We compute products of the form (Aij + Akk) * (Bll + Bmm).
        
        # Let's use the explicit matrices for 4x4x4 rank 49.
        # This is well-known.
        # U, V, W are 16x49 matrices.
        # I will generate them programmatically if I can, otherwise I'll stick to schoolbook or a simpler bound.
        
        # Given the time constraint and complexity, let's try a different approach.
        # We can use the fact that rank(4,4,9) <= rank(4,4,4) + rank(4,4,4) + rank(4,4,1)
        # rank(4,4,1) = 16 (schoolbook).
        # rank(4,4,4) = 49 (Strassen).
        # Total = 49 + 49 + 16 = 114.
        
        # Let's implement Strassen 4x4x4.
        # The 2x2x2 Strassen U, V, W are:
        # U:
        # 1 0 1 0 1 0 0
        # 0 1 0 0 0 1 0
        # 0 0 0 1 0 0 1
        # 1 0 0 0 0 0 1
        # V:
        # 1 1 0 0 0 0 0
        # 0 0 1 0 0 0 0
        # 0 0 0 1 1 0 0
        # 1 0 0 0 0 1 0
        # W:
        # 1 0 0 1 -1 0 1
        # 0 1 0 0 1 0 0
        # 0 0 1 1 0 0 0
        # 1 0 -1 0 0 1 0
        
        # To get 4x4x4, we use the block structure.
        # A = [[A00, A01], [A10, A11]]
        # B = [[B00, B01], [B10, B11]]
        
        # The 16 entries of A are indexed by (i, j) where i, j in {0, 1, 2, 3}.
        # i = 2*I + I2, j = 2*J + J2.
        
        # We will construct U, V, W of size 16x49.
        # The 49 columns correspond to the 7 Strassen products for each of the 7 "meta-Strassen" products?
        # No, the rank is 49 = 7 * 7.
        
        # Let's denote the Strassen 2x2x2 factors as U2, V2, W2 (4x7).
        # The 4x4x4 decomposition can be seen as:
        # For each of the 7 meta-products (from the outer 2x2x2 Strassen),
        # we perform a 2x2x2 Strassen decomposition on the blocks.
        
        # Let the outer Strassen products be M1..M7.
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B10 - B11)
        # M4 = A11 * (B01 - B00)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        
        # Each Aij is a 2x2 matrix. Each Bij is a 2x2 matrix.
        # So M1 is a 2x2 matrix product of sums.
        # M1 = (A00 + A11) * (B00 + B11)
        # Let X = A00 + A11, Y = B00 + B11.
        # X * Y can be computed with Strassen rank 7.
        # So M1 requires 7 scalar multiplications.
        # Similarly for M2..M7.
        # Total 7 * 7 = 49 multiplications.
        
        # We need to map these to the global U, V, W.
        # U is 16x49.
        # V is 16x49.
        # W is 16x49.
        
        # Let's index the columns of U, V, W by (k, l) where k in 0..6, l in 0..6.
        # k indexes the outer Strassen product, l indexes the inner Strassen product.
        
        # For each k in 0..6:
        #   The outer Strassen formula defines linear combinations of A blocks.
        #   Let U_outer_k be a 4x4 matrix (for A) such that A_outer = sum_k U_outer_k * A_k?
        #   No, Strassen defines:
        #   M1 = (A00 + A11) * (B00 + B11)
        #   The A part for M1 is A00 + A11.
        #   In terms of 4x4 A, A00 is A[0:2, 0:2], A11 is A[2:4, 2:4].
        #   So the A factor for M1 is a 4x4 matrix with A[0:2, 0:2] + A[2:4, 2:4] in the relevant blocks?
        #   Actually, A_outer_k is a 4x4 matrix that selects the blocks.
        
        # Let's define the outer Strassen U, V, W for 2x2x2 where the "elements" are 2x2 matrices.
        # U_outer is 4x7 (where 4 are the block indices 00, 01, 10, 11).
        # V_outer is 4x7.
        # W_outer is 4x7.
        
        # U_outer = U2 (from strassen_2x2x2)
        # V_outer = V2
        # W_outer = W2
        
        # Now, for each k in 0..6, we have a product of two 2x2 matrices (which are sums of blocks of A and B).
        # This product is computed using the inner Strassen decomposition.
        
        # Let's build U, V, W.
        U = np.zeros((16, 49))
        V = np.zeros((16, 49))
        W = np.zeros((16, 49))
        
        U2, V2, W2 = strassen_2x2x2()
        
        for k in range(7):
            for l in range(7):
                col = k * 7 + l
                
                # The A factor for this scalar multiplication is:
                # (sum_{i in 0..3} U_outer[i, k] * A_i) * (sum_{j in 0..3} U_inner[j, l] * (2x2 block))
                # Wait, the outer Strassen combines 4x4 blocks (which are 2x2 matrices).
                # The inner Strassen combines 2x2 elements.
                
                # Let's be precise.
                # A is 4x4. A[i, j].
                # Block A_{IJ} is 2x2, where I = i//2, J = j//2.
                # A_{00} = A[0:2, 0:2], A_{01} = A[0:2, 2:4], etc.
                
                # The outer Strassen product k uses linear combinations of A_{00}, A_{01}, A_{10}, A_{11}.
                # Let A_k = sum_{I, J} U_outer[I*2+J, k] * A_{IJ} (conceptually).
                # Actually, U_outer has rows 0,1,2,3 corresponding to blocks 00, 01, 10, 11.
                # So for a fixed k, the coefficient of block A_{IJ} is U_outer[I*2+J, k].
                
                # Similarly for B_k = sum_{I, J} V_outer[I*2+J, k] * B_{IJ}.
                
                # Now, A_k and B_k are 2x2 matrices (since A_{IJ} are 2x2).
                # We compute A_k * B_k using the inner Strassen.
                # A_k * B_k = sum_{l} (sum_{I', J'} U_inner[I'*2+J', l] * (A_k)_{I'J'}) * (sum_{I'', J''} V_inner[I''*2+J'', l] * (B_k)_{I''J''}) * W_inner[I'*2+J', l] ...
                
                # This is getting confusing. Let's use the property of Kronecker products.
                # The 4x4x4 tensor is the Kronecker product of two 2x2x2 tensors?
                # No, T_{4,4,4} is not T_{2,2,2} \otimes T_{2,2,2}.
                # T_{2,2,2} \otimes T_{2,2,2} has rank 7*7=49, which is correct.
                # But is it the correct tensor?
                # T_{4,4,4} = sum_{i, j, k} A_{i, j} B_{j, k} E_{i, k}.
                # If we index i = 2I + I', j = 2J + J', k = 2K + K'.
                # Then A_{i, j} = A_{I, J, I', J'}, B_{j, k} = B_{J, K, J', K'}.
                # The product is A_{I, J, I', J'} B_{J, K, J', K'}.
                # This is exactly the Kronecker product of the two 2x2x2 tensors.
                
                # So, U_{4x4x4} = U_{2x2x2} \otimes U_{2x2x2}
                # V_{4x4x4} = V_{2x2x2} \otimes V_{2x2x2}
                # W_{4x4x4} = W_{2x2x2} \otimes W_{2x2x2}
                
                # Let's verify this.
                # U2 is 4x7. U2 \otimes U2 is 16x49.
                # The rows of U2 \otimes U2 are indexed by (i1, i2) where i1 in 0..3, i2 in 0..3.
                # This corresponds to i = 2*i1 + i2?
                # Kronecker product (A \otimes B)[i, j] = A[i//m, j//n] * B[i%m, j%n].
                # So (U2 \otimes U2)[2*I + I', 2*J + J'] = U2[I, J] * U2[I', J'].
                # This matches the indexing i = 2I + I'.
                
                # So we can simply use the Kronecker product!
                
        U4 = np.kron(U2, U2)
        V4 = np.kron(V2, V2)
        W4 = np.kron(W2, W2)
        
        # Now we have U4, V4, W4 for 4x4x4 (rank 49).
        # We need 4x4x9.
        # We can use U4, V4, W4 for the first 4 columns (rank 49).
        # We can use U4, V4, W4 for the next 4 columns (rank 49).
        # We can use schoolbook for the last 1 column (rank 16).
        
        # Total rank = 49 + 49 + 16 = 114.
        
        # Let's construct the full U, V, W for 4x4x9.
        # U is 16x114.
        # V is 36x114. (4*9 = 36)
        # W is 36x114. (4*9 = 36)
        
        U = np.zeros((16, 114))
        V = np.zeros((36, 114))
        W = np.zeros((36, 114))
        
        # Block 1: Columns 0-3 of B.
        # V4 is 16x49. We need to embed it into V (36x49).
        # V4 corresponds to B[0:4, 0:4].
        # In V (36x49), the rows 0-15 correspond to B[0:4, 0:4].
        V[0:16, 0:49] = V4
        U[:, 0:49] = U4
        # W4 is 16x49. It corresponds to C[0:4, 0:4].
        # In W (36x49), the rows 0-15 correspond to C[0:4, 0:4].
        W[0:16, 0:49] = W4
        
        # Block 2: Columns 4-7 of B.
        # V4 corresponds to B[0:4, 4:8].
        # In V (36x49), the rows 16-31 correspond to B[0:4, 4:8].
        V[16:32, 49:98] = V4
        U[:, 49:98] = U4
        # W4 corresponds to C[0:4, 4:8].
        # In W (36x49), the rows 16-31 correspond to C[0:4, 4:8].
        W[16:32, 49:98] = W4
        
        # Block 3: Column 8 of B.
        # Schoolbook for 4x4x1.
        # U_schoolbook is 16x16.
        # V_schoolbook is 4x16. (Since it's 4x1, m=4, p=1)
        # W_schoolbook is 4x16.
        
        U_sb = np.zeros((16, 16))
        V_sb = np.zeros((4, 16))
        W_sb = np.zeros((4, 16))
        
        col = 0
        for i in range(4):
            for j in range(4):
                U_sb[i * 4 + j, col] = 1
                V_sb[j, col] = 1
                W_sb[i, col] = 1
                col += 1
        
        # Embed into U, V, W.
        U[:, 98:114] = U_sb
        # V_sb is 4x16. It corresponds to B[0:4, 8].
        # In V (36x16), the rows 32-35 correspond to B[0:4, 8].
        V[32:36, 98:114] = V_sb
        # W_sb is 4x16. It corresponds to C[0:4, 8].
        # In W (36x16), the rows 32-35 correspond to C[0:4, 8].
        W[32:36, 98:114] = W_sb
        
        # Convert to lists of integers/half-integers
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        
        # The problem requires integer/half-integer entries.
        # Our entries are -1, 0, 1. So they are integers.
        
        return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
