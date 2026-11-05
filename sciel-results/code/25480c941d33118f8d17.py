import time
import sys

DIMENSIONS = [4, 4, 8]

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    n, m, p = DIMENSIONS
    
    # Base rank-7 decomposition for 2x2 matrices
    # u, v are 4-vectors for A, B blocks
    # w is 4-vector for C block
    base_decomp = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),  # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),  # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),  # M3
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),  # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),  # M5
        ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]),  # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),  # M7
    ]
    
    # Strassen coefficients for 4x4 (2x2 blocks of 2x2)
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Each M_k is a product of two 2x2 matrices.
    # We need to express the linear forms for U_k and V_k in terms of the 4 blocks.
    # U_k is a 2x2 matrix, V_k is a 2x2 matrix.
    # The elements of U_k are linear combinations of A11, A12, A21, A22.
    # The elements of V_k are linear combinations of B11, B12, B21, B22.
    
    # Let's define the linear forms for U_k and V_k as 4x4 matrices (4 blocks, 4 elements each)
    # Actually, it's easier to think of U_k as a 2x2 matrix where each entry is a linear form on the 4 blocks.
    # But since the blocks are 2x2, the linear form is on the 16 elements of A.
    
    # Let's construct the 49 terms directly.
    # For each k in 1..7:
    #   Determine the linear forms for U_k (4x4 matrix of coefficients for A's 16 elements? No, U_k is 2x2 blocks)
    #   Actually, U_k is a 2x2 matrix of linear forms on the 4 blocks.
    #   Let's represent U_k as a 4x4 matrix where each row/col corresponds to a block? No.
    #   U_k is a 2x2 matrix. Its entries are linear combinations of A11, A12, A21, A22.
    #   Let U_k[i,j] = sum_{b in 1..4} c_{i,j,b} * A_b.
    #   Then the vector for A in the rank-1 term is formed by placing c_{i,j,b} in the position of element (i,j) of block A_b.
    
    # Let's define the coefficients for U_k and V_k.
    # A_b are 2x2 matrices. Let's index elements of A as A[r,c] where r,c in 0..3.
    # Block A11 is A[0:2, 0:2], A12 is A[0:2, 2:4], A21 is A[2:4, 0:2], A22 is A[2:4, 2:4].
    
    # M1: U1 = A11+A22, V1 = B11+B22
    # U1[0,0] = A11, U1[0,1] = 0, U1[1,0] = 0, U1[1,1] = A22 ? No.
    # U1 is a 2x2 matrix of blocks.
    # U1 = [[A11, 0], [0, A22]] + [[0, 0], [0, 0]]? No.
    # U1 = A11 + A22 means the block matrix [[A11, 0], [0, A22]]? No.
    # In Strassen, U1 and V1 are 2x2 matrices.
    # U1 = [[A11, 0], [0, A22]] is not correct. U1 is a matrix whose entries are linear combinations of the blocks.
    # Actually, U1 is a 2x2 matrix where each entry is a 2x2 matrix.
    # U1[0,0] = A11, U1[0,1] = 0, U1[1,0] = 0, U1[1,1] = A22.
    # So U1 is the block diagonal matrix of A11 and A22.
    # Similarly V1 is block diagonal of B11 and B22.
    
    # Let's list the block matrices for U_k and V_k.
    # U1 = [[A11, 0], [0, A22]]
    # V1 = [[B11, 0], [0, B22]]
    
    # U2 = [[0, 0], [A21+A22, 0]]? No.
    # M2 = (A21+A22)B11.
    # U2 = [[0, 0], [A21+A22, 0]]? No, U2 is 2x2.
    # U2[0,0]=0, U2[0,1]=0, U2[1,0]=A21+A22, U2[1,1]=0.
    # V2 = [[B11, 0], [0, 0]].
    
    # U3 = [[A11, 0], [0, 0]].
    # V3 = [[0, B12], [0, -B22]].
    
    # U4 = [[0, 0], [0, A22]].
    # V4 = [[0, -B11], [B21, 0]].
    
    # U5 = [[A11+A12, 0], [0, 0]].
    # V5 = [[0, 0], [0, B22]].
    
    # U6 = [[0, 0], [A21-A11, 0]].
    # V6 = [[B11, B12], [0, 0]].
    
    # U7 = [[0, A12-A22], [0, 0]].
    # V7 = [[0, 0], [B21, B22]].
    
    # For each k, we have U_k and V_k as 2x2 block matrices.
    # We decompose the product U_k V_k using the base rank-7 decomposition.
    # The base decomposition computes C = U V where U, V are 2x2 matrices of scalars.
    # Here U, V are 2x2 matrices of 2x2 matrices.
    # So each "scalar" multiplication in the base decomposition is actually a 2x2 matrix multiplication.
    # We replace each scalar multiplication in the base decomposition with the rank-4 schoolbook decomposition for 2x2 matrices?
    # No, we replace it with the rank-7 Strassen decomposition for 2x2 matrices.
    # This is recursive.
    
    # Base decomposition for 2x2 (scalars):
    # M1 = (u11+u22)(v11+v22)
    # ...
    # Here u11, u22 etc are 2x2 matrices.
    # So (u11+u22) is a 2x2 matrix.
    # The product (u11+u22)(v11+v22) is a 2x2 matrix product.
    # We decompose this product using the base rank-7 decomposition again?
    # No, the base rank-7 decomposition is for 2x2 matrices of scalars.
    # If the "scalars" are 2x2 matrices, we need to decompose the product of two 2x2 matrices.
    # The rank of 2x2 matrix multiplication is 7 (Strassen).
    # So we use the same base_decomp structure, but the "scalars" are 2x2 matrices.
    # This means we apply the base_decomp to the 2x2 matrices.
    # This results in 7 terms, each of which is a product of linear combinations of 2x2 matrices.
    # But a product of linear combinations of 2x2 matrices is not a single rank-1 term.
    # It is a sum of rank-1 terms.
    # Specifically, if we have (U_a U_b) where U_a, U_b are 2x2 matrices, we need to decompose U_a U_b.
    # This is exactly what the base_decomp does.
    # So we apply base_decomp to the 2x2 matrices.
    # This generates 7 terms for each of the 7 M_k.
    # Total 49 terms.
    
    # Implementation:
    # 1. Define the 7 U_k, V_k block matrices (coefficients for A, B blocks).
    # 2. For each k, apply the base_decomp to the 2x2 matrices U_k, V_k.
    # 3. This generates 7 terms for each k.
    # 4. Each term from the base_decomp is (u_vec, v_vec, w_vec) where u_vec, v_vec are 4-vectors.
    # 5. These 4-vectors correspond to linear combinations of the 4 elements of the 2x2 matrix U_k.
    # 6. But U_k is a 2x2 matrix of 2x2 blocks.
    # 7. So each element of U_k is a 2x2 matrix.
    # 8. The linear combination of the 4 elements of U_k is a 2x2 matrix.
    # 9. So the "u_vec" in the base_decomp term actually represents a 2x2 matrix of coefficients.
    # 10. We need to expand this into the full 16-element vector for A.
    
    # Let's do this carefully.
    
    # Base decomp term: (u4, v4, w4)
    # u4 is a 4-vector [u0, u1, u2, u3] corresponding to elements of U (2x2).
    # U is a 2x2 matrix. Its elements are U[0,0], U[0,1], U[1,0], U[1,1].
    # In the context of 4x4, U[0,0] is a 2x2 block (A11, A12, A21, A22 combination).
    # So u0 is a coefficient for U[0,0].
    # U[0,0] itself is a linear combination of A blocks.
    # So the full coefficient for A is u0 * (coeffs for U[0,0]) + ...
    
    # Let's define the block coefficients for U_k and V_k.
    # U_k is a 2x2 matrix of 2x2 blocks.
    # Let U_k_blocks be a 4x4 matrix where U_k_blocks[i,j] is the coefficient vector for the block at (i,j).
    # Actually, it's easier to store U_k as a 4x4 matrix of coefficients for the 16 elements of A.
    
    # Let's construct the 4x4 matrices U_k_full and V_k_full (16x1 vectors).
    
    def block_to_vec(block_coeffs):
        # block_coeffs is a 2x2 matrix of 4-vectors (coeffs for 2x2 block)
        # Returns a 16-vector
        vec = [0]*16
        for i in range(2):
            for j in range(2):
                # Block (i,j) occupies rows 2*i to 2*i+1, cols 2*j to 2*j+1
                for r in range(2):
                    for c in range(2):
                        idx = (2*i + r)*4 + (2*j + c)
                        vec[idx] += block_coeffs[i][j][r*2+c]
        return vec

    # Define U_k, V_k as 2x2 matrices of 4-vectors (coeffs for the 4 blocks)
    # A blocks: 0:A11, 1:A12, 2:A21, 3:A22
    # Each block coeff is a 4-vector [c00, c01, c10, c11] for the 2x2 block elements.
    
    # U1 = [[A11, 0], [0, A22]]
    U1 = [[ [1,0,0,0], [0,0,0,0] ], [ [0,0,0,0], [0,0,1,0] ]] # Wait, A22 is block 3.
    # Block 0: A11, 1: A12, 2: A21, 3: A22.
    # U1[0,0] = A11 => coeff [1,0,0,0] for block 0? No, the block itself is 2x2.
    # The coefficient for the block is a 2x2 matrix of scalars.
    # Here U1[0,0] is exactly A11. So the coefficient matrix is Identity.
    # U1[1,1] is exactly A22. So the coefficient matrix is Identity.
    
    # Let's represent U_k as a 2x2 matrix where each entry is a 4x4 matrix? No.
    # Each entry is a 2x2 matrix of coefficients for the 2x2 block.
    # So U_k[i][j] is a 2x2 matrix.
    
    U1 = [[ [[1,0],[0,1]], [[0,0],[0,0]] ],
           [ [[0,0],[0,0]], [[1,0],[0,1]] ]]
           
    V1 = [[ [[1,0],[0,1]], [[0,0],[0,0]] ],
           [ [[0,0],[0,0]], [[1,0],[0,1]] ]]
           
    U2 = [[ [[0,0],[0,0]], [[0,0],[0,0]] ],
           [ [[1,0],[0,1]]+[[1,0],[0,1]], [[0,0],[0,0]] ]] # A21+A22
    # A21 is block 2, A22 is block 3.
    # U2[1,0] = A21+A22.
    # Coeff for block 2 is Identity, for block 3 is Identity.
    # But my representation is per block position in the 2x2 matrix of blocks.
    # U2[1,0] corresponds to the block at row 1, col 0 of the 4x4 matrix.
    # This block is A21.
    # So U2[1,0] should be Identity (for A21) + 0 (for others)?
    # No, U2[1,0] is a linear combination of the 4 blocks.
    # So U2[1,0] = 1*A21 + 1*A22.
    # This means the coefficient for block 2 (A21) is Identity, and for block 3 (A22) is Identity.
    # But in my 2x2 matrix of blocks, position (1,0) is A21.
    # So I need to sum the contributions.
    
    # Let's use a 4x4 matrix of 2x2 coefficient matrices.
    # U_k_full[i][j] is the 2x2 coefficient matrix for the element at (i,j) of the 4x4 matrix.
    
    # This is getting complicated. Let's use the block_to_vec approach with a 4x4 matrix of 4-vectors.
    # U_k_blocks[i][j] is a 4-vector representing the coefficient for the block at (i,j).
    # Wait, the block at (i,j) is one of A11, A12, A21, A22.
    # So U_k_blocks is a 2x2 matrix of 4-vectors.
    # Each 4-vector has entries for the 4 blocks.
    
    # U1:
    # U1[0,0] (block A11) = A11. Coeffs: [1,0,0,0]
    # U1[0,1] (block A12) = 0. Coeffs: [0,0,0,0]
    # U1[1,0] (block A21) = 0. Coeffs: [0,0,0,0]
    # U1[1,1] (block A22) = A22. Coeffs: [0,0,0,1]
    
    U1_blocks = [[ [1,0,0,0], [0,0,0,0] ],
                  [ [0,0,0,0], [0,0,0,1] ]]
                  
    V1_blocks = [[ [1,0,0,0], [0,0,0,0] ],
                  [ [0,0,0,0], [0,0,0,1] ]]
                  
    U2_blocks = [[ [0,0,0,0], [0,0,0,0] ],
                  [ [0,0,1,1], [0,0,0,0] ]] # A21+A22 at (1,0)
                  
    V2_blocks = [[ [1,0,0,0], [0,0,0,0] ],
                  [ [0,0,0,0], [0,0,0,0] ]] # B11 at (0,0)
                  
    U3_blocks = [[ [1,0,0,0], [0,0,0,0] ],
                  [ [0,0,0,0], [0,0,0,0] ]] # A11 at (0,0)
                  
    V3_blocks = [[ [0,0,0,0], [0,1,0,0] ],
                  [ [0,0,0,0], [0,0,0,-1] ]] # B12 at (0,1), -B22 at (1,1)
                  
    U4_blocks = [[ [0,0,0,0], [0,0,0,0] ],
                  [ [0,0,0,0], [0,0,0,1] ]] # A22 at (1,1)
                  
    V4_blocks = [[ [0,0,-1,0], [0,0,0,0] ],
                  [ [0,0,1,0], [0,0,0,0] ]] # -B11 at (0,0), B21 at (1,0)
                  
    U5_blocks = [[ [1,1,0,0], [0,0,0,0] ],
                  [ [0,0,0,0], [0,0,0,0] ]] # A11+A12 at (0,0)
                  
    V5_blocks = [[ [0,0,0,0], [0,0,0,0] ],
                  [ [0,0,0,0], [0,0,0,1] ]] # B22 at (1,1)
                  
    U6_blocks = [[ [0,0,0,0], [0,0,0,0] ],
                  [ [0,-1,1,0], [0,0,0,0] ]] # A21-A11 at (1,0)
                  
    V6_blocks = [[ [1,0,0,0], [0,1,0,0] ],
                  [ [0,0,0,0], [0,0,0,0] ]] # B11 at (0,0), B12 at (0,1)
                  
    U7_blocks = [[ [0,0,0,0], [0,1,0,-1] ],
                  [ [0,0,0,0], [0,0,0,0] ]] # A12-A22 at (0,1)
                  
    V7_blocks = [[ [0,0,0,0], [0,0,0,0] ],
                  [ [0,0,1,0], [0,0,0,1] ]] # B21 at (1,0), B22 at (1,1)
    
    U_blocks_list = [U1_blocks, U2_blocks, U3_blocks, U4_blocks, U5_blocks, U6_blocks, U7_blocks]
    V_blocks_list = [V1_blocks, V2_blocks, V3_blocks, V4_blocks, V5_blocks, V6_blocks, V7_blocks]
    
    # Now, for each k, apply base_decomp to the 2x2 matrices U_k, V_k.
    # The base_decomp operates on 4 elements.
    # Here the 4 elements are the 4 blocks.
    # So for each term (u4, v4, w4) in base_decomp:
    #   u4 is a 4-vector [u0, u1, u2, u3]
    #   u0 is the coefficient for U[0,0] (block A11)
    #   u1 is the coefficient for U[0,1] (block A12)
    #   u2 is the coefficient for U[1,0] (block A21)
    #   u3 is the coefficient for U[1,1] (block A22)
    #   The full coefficient for A is sum_{i,j} u_{i,j} * U_k_blocks[i][j]
    #   Where U_k_blocks[i][j] is a 4-vector of coefficients for the 4 blocks.
    #   Wait, U_k_blocks[i][j] is a 4-vector representing the linear combination of the 4 blocks.
    #   So the full 16-vector for A is:
    #   For each block b (0..3):
    #      coeff_b = sum_{i,j} u_{i,j} * U_k_blocks[i][j][b]
    #   Then place coeff_b in the 4 positions of block b.
    
    U = [[0]*98 for _ in range(16)]
    V = [[0]*98 for _ in range(32)]
    W = [[0]*98 for _ in range(32)]
    
    col = 0
    for k in range(7):
        U_k_blocks = U_blocks_list[k]
        V_k_blocks = V_blocks_list[k]
        
        for u4, v4, w4 in base_decomp:
            # Compute full U vector (16)
            U_vec = [0]*16
            for b in range(4):
                coeff = 0
                for i in range(2):
                    for j in range(2):
                        coeff += u4[i*2+j] * U_k_blocks[i][j][b]
                # Place coeff in the 4 positions of block b
                # Block 0: 0:2, 0:2 -> indices 0,1,4,5
                # Block 1: 0:2, 2:4 -> indices 2,3,6,7
                # Block 2: 2:4, 0:2 -> indices 8,9,12,13
                # Block 3: 2:4, 2:4 -> indices 10,11,14,15
                if b == 0:
                    U_vec[0] += coeff; U_vec[1] += coeff; U_vec[4] += coeff; U_vec[5] += coeff
                elif b == 1:
                    U_vec[2] += coeff; U_vec[3] += coeff; U_vec[6] += coeff; U_vec[7] += coeff
                elif b == 2:
                    U_vec[8] += coeff; U_vec[9] += coeff; U_vec[12] += coeff; U_vec[13] += coeff
                elif b == 3:
                    U_vec[10] += coeff; U_vec[11] += coeff; U_vec[14] += coeff; U_vec[15] += coeff
            
            # Compute full V vector (16 for B1)
            V_vec = [0]*16
            for b in range(4):
                coeff = 0
                for i in range(2):
                    for j in range(2):
                        coeff += v4[i*2+j] * V_k_blocks[i][j][b]
                if b == 0:
                    V_vec[0] += coeff; V_vec[1] += coeff; V_vec[4] += coeff; V_vec[5] += coeff
                elif b == 1:
                    V_vec[2] += coeff; V_vec[3] += coeff; V_vec[6] += coeff; V_vec[7] += coeff
                elif b == 2:
                    V_vec[8] += coeff; V_vec[9] += coeff; V_vec[12] += coeff; V_vec[13] += coeff
                elif b == 3:
                    V_vec[10] += coeff; V_vec[11] += coeff; V_vec[14] += coeff; V_vec[15] += coeff
            
            # Compute full W vector (16 for C1)
            # w4 is a 4-vector for the 4 blocks of C.
            # C blocks are C11, C12, C21, C22.
            # w4[0] for C11, w4[1] for C12, w4[2] for C21, w4[3] for C22.
            W_vec = [0]*16
            for b in range(4):
                coeff = w4[b]
                if b == 0:
                    W_vec[0] += coeff; W_vec[1] += coeff; W_vec[4] += coeff; W_vec[5] += coeff
                elif b == 1:
                    W_vec[2] += coeff; W_vec[3] += coeff; W_vec[6] += coeff; W_vec[7] += coeff
                elif b == 2:
                    W_vec[8] += coeff; W_vec[9] += coeff; W_vec[12] += coeff; W_vec[13] += coeff
                elif b == 3:
                    W_vec[10] += coeff; W_vec[11] += coeff; W_vec[14] += coeff; W_vec[15] += coeff
            
            # Assign to U, V, W
            for r in range(16):
                U[r][col] = U_vec[r]
            for r in range(16):
                V[r][col] = V_vec[r]
                V[r+16][col] = 0 # B2 part is 0
            for r in range(16):
                W[r][col] = W_vec[r]
                W[r+16][col] = 0 # C2 part is 0
            col += 1
            
            if time.time() > deadline:
                return {"U": U, "V": V, "W": W}
        
        # Now do the same for B2 and C2
        # U is the same
        # V is shifted to B2 (indices 16-31)
        # W is shifted to C2 (indices 16-31)
        
        for u4, v4, w4 in base_decomp:
            U_vec = [0]*16
            for b in range(4):
                coeff = 0
                for i in range(2):
                    for j in range(2):
                        coeff += u4[i*2+j] * U_k_blocks[i][j][b]
                if b == 0:
                    U_vec[0] += coeff; U_vec[1] += coeff; U_vec[4] += coeff; U_vec[5] += coeff
                elif b == 1:
                    U_vec[2] += coeff; U_vec[3] += coeff; U_vec[6] += coeff; U_vec[7] += coeff
                elif b == 2:
                    U_vec[8] += coeff; U_vec[9] += coeff; U_vec[12] += coeff; U_vec[13] += coeff
                elif b == 3:
                    U_vec[10] += coeff; U_vec[11] += coeff; U_vec[14] += coeff; U_vec[15] += coeff
            
            V_vec = [0]*16
            for b in range(4):
                coeff = 0
                for i in range(2):
                    for j in range(2):
                        coeff += v4[i*2+j] * V_k_blocks[i][j][b]
                if b == 0:
                    V_vec[0] += coeff; V_vec[1] += coeff; V_vec[4] += coeff; V_vec[5] += coeff
                elif b == 1:
                    V_vec[2] += coeff; V_vec[3] += coeff; V_vec[6] += coeff; V_vec[7] += coeff
                elif b == 2:
                    V_vec[8] += coeff; V_vec[9] += coeff; V_vec[12] += coeff; V_vec[13] += coeff
                elif b == 3:
                    V_vec[10] += coeff; V_vec[11] += coeff; V_vec[14] += coeff; V_vec[15] += coeff
            
            W_vec = [0]*16
            for b in range(4):
                coeff = w4[b]
                if b == 0:
                    W_vec[0] += coeff; W_vec[1] += coeff; W_vec[4] += coeff; W_vec[5] += coeff
                elif b == 1:
                    W_vec[2] += coeff; W_vec[3] += coeff; W_vec[6] += coeff; W_vec[7] += coeff
                elif b == 2:
                    W_vec[8] += coeff; W_vec[9] += coeff; W_vec[12] += coeff; W_vec[13] += coeff
                elif b == 3:
                    W_vec[10] += coeff; W_vec[11] += coeff; W_vec[14] += coeff; W_vec[15] += coeff
            
            for r in range(16):
                U[r][col] = U_vec[r]
            for r in range(16):
                V[r][col] = 0 # B1 part is 0
                V[r+16][col] = V_vec[r]
            for r in range(16):
                W[r][col] = 0 # C1 part is 0
                W[r+16][col] = W_vec[r]
            col += 1
            
            if time.time() > deadline:
                return {"U": U, "V": V, "W": W}
        
        if time.time() > deadline:
            return {"U": U, "V": V, "W": W}
            
    return {"U": U, "V": V, "W": W}
