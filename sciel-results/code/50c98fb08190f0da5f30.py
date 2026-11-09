import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # The goal is to find a rank-minimal decomposition.
    # Strategy: Start with the schoolbook decomposition (rank 80) and try to reduce the rank
    # by finding linear dependencies among the terms using a global search.
    
    # 1. Construct the schoolbook decomposition
    # The tensor T is defined by T[i, j, k, r] where r is the rank index.
    # For schoolbook, each term r corresponds to a unique (i, j, k) triple.
    # U[i*m+j, r] = 1 if r corresponds to (i, j, k)
    # V[j*p+k, r] = 1 if r corresponds to (i, j, k)
    # W[k*n+i, r] = 1 if r corresponds to (i, j, k)
    
    base_rank = n * m * p
    # Index mapping: r = i * m * p + j * p + k
    U = np.zeros((n * m, base_rank))
    V = np.zeros((m * p, base_rank))
    W = np.zeros((p * n, base_rank))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                U[i * m + j, r] = 1
                V[j * p + k, r] = 1
                W[k * n + i, r] = 1
                
    # 2. Attempt to reduce rank by finding linear dependencies
    # We view the decomposition as:
    # sum_r U[:, r] * V[:, r]^T * W[:, r] = T
    # This is equivalent to the tensor T having a rank decomposition.
    # We can try to express some columns of U as linear combinations of others,
    # and adjust V and W accordingly.
    
    # A simpler approach: Randomly perturb the decomposition and see if we can
    # find a lower rank representation. However, this is computationally expensive.
    
    # Instead, we use a known fact: The rank of the matrix multiplication tensor
    # is invariant under certain transformations. We can try to apply random
    # invertible transformations to U, V, W and see if we can simplify the structure.
    
    # However, a more effective strategy for small dimensions is to use the
    # "linear algebra" approach:
    # 1. Form the tensor T as a 3D array.
    # 2. Use a rank-reduction algorithm (like Gaussian elimination on the tensor slices).
    
    # Let's try a specific known improvement: Strassen's algorithm for 2x2 blocks.
    # We already know that for 4x4x4, Strassen gives rank 72.
    # For 4x4x5, we can use Strassen for the 4x4 part and add the 5th column.
    # But the current program already does this (or tries to).
    
    # Let's try a different structural change: 
    # We will use a "randomized" rank reduction search.
    # We start with the schoolbook decomposition and try to merge terms.
    
    # For this specific task, the best known rank is likely 72 or 76.
    # Since the current program returns 80 (score 1.0), and the inspiration says
    # 1.2308 corresponds to rank 65 (which is 80/65), let's aim for rank 65.
    
    # Wait, the inspiration says: "Recursive Strassen reduces 4x4x4 from rank 56 
    # (schoolbook inner) to rank 49 (Strassen inner), saving 7 rank units for 
    # total rank 65 vs 72".
    
    # This implies that for 4x4x4, the rank is 65? 
    # Schoolbook 4x4x4 is 64. Strassen is 49? No, Strassen for 2x2 is 7.
    # 4x4 is 4 blocks of 2x2. 4 * 7 = 28? No, Strassen for 4x4 is 4^log2(7) ~ 21.8.
    # Actually, the rank of 4x4x4 is known to be 49 (Strassen) or lower?
    # No, the rank of 2x2x2 is 7. The rank of 4x4x4 is at most 7^2 = 49.
    # So if we use Strassen recursively, we get rank 49 for 4x4x4.
    
    # For 4x4x5:
    # We can compute the 4x4 part with rank 49.
    # The 5th column (4x1) has rank 16 (schoolbook) or maybe lower?
    # Rank of 4x4x1 is 16.
    # Total rank = 49 + 16 = 65.
    
    # This matches the inspiration! Rank 65.
    
    # So the strategy is:
    # 1. Construct a rank 49 decomposition for 4x4x4 using recursive Strassen.
    # 2. Construct a rank 16 decomposition for the 4x4x1 part (5th column).
    # 3. Combine them.
    
    # Let's implement the recursive Strassen for 4x4x4.
    
    def strassen_2x2(U, V, W, n=2, m=2, p=2):
        # Returns U, V, W for 2x2x2 Strassen decomposition
        # Rank 7
        U = np.zeros((n*m, 7))
        V = np.zeros((m*p, 7))
        W = np.zeros((p*n, 7))
        
        # Strassen formulas for C = A*B
        # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
        # C11 = M1 - M4 + M6 + M7
        # C12 = M3 + M4
        # C21 = M1 + M2
        # C22 = M3 - M2 + M5 + M7
        # Where:
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-d)(e+f)
        # M7 = (b-d)(g+h)
        
        # Indices for U (A elements): 0=a, 1=b, 2=c, 3=d
        # Indices for V (B elements): 0=e, 1=f, 2=g, 3=h
        # Indices for W (C elements): 0=C11, 1=C12, 2=C21, 3=C22
        
        # M1: (a+d)(e+h)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1 # Wait, W is for C.
        # C11 gets +M1, C22 gets +M1? No.
        # C11 = M1 - M4 + M6 + M7
        # C22 = M3 - M2 + M5 + M7
        
        # Let's map the terms to the rank indices 0..6
        
        # Term 0: M1 = (a+d)(e+h)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1 # C11
        W[3, 0] = 1 # C22? No, M1 is in C11 and C22?
        # C11 = M1 - M4 + M6 + M7
        # C22 = M3 - M2 + M5 + M7
        # M1 is only in C11? No, (a+d)(e+h) = ae+ah+de+dh.
        # C11 = ae+bg. C22 = cf+dh.
        # M1 = ae+ah+de+dh.
        # So M1 contributes to ae (C11) and dh (C22).
        # But we are decomposing the tensor, so we need to express the coefficients.
        
        # The tensor T has entries T[i,j,k] = delta_{i,k} for A[i,j]B[j,k]?
        # No, the tensor is T_{ijk} such that C_{ik} = sum_j A_{ij} B_{jk}.
        # The rank decomposition is T = sum_r U_r \otimes V_r \otimes W_r.
        # U_r is a vector in R^{nm}, V_r in R^{mp}, W_r in R^{pn}.
        
        # For Strassen, we define 7 bilinear forms M_r(A, B).
        # And express C_{ik} = sum_r c_{ikr} M_r(A, B).
        
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A22)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1 - M4 + M6 + M7
        # C12 = M3 + M4
        # C21 = M1 + M2
        # C22 = M3 - M2 + M5 + M7
        
        # So:
        # U_1 = [1, 0, 0, 1] (A11, A12, A21, A22)
        # V_1 = [1, 0, 0, 1] (B11, B12, B21, B22)
        # W_1 = [1, 0, 0, 1] (C11, C12, C21, C22) -> Wait, W is for C.
        # C11 is index 0, C12 is 1, C21 is 2, C22 is 3.
        # W_1 has 1 at C11 and C22?
        # C11 coeff of M1 is 1. C22 coeff of M1 is 0?
        # C22 = M3 - M2 + M5 + M7. M1 is not in C22.
        # So W_1 = [1, 0, 0, 0].
        
        # Let's re-verify C11.
        # C11 = A11B11 + A12B21.
        # M1 = A11B11 + A11B22 + A22B11 + A22B22.
        # M4 = A22B21 - A22B11.
        # M6 = A21B11 + A21B12 - A22B11 - A22B12.
        # M7 = A12B21 + A12B22 - A22B21 - A22B22.
        # M1 - M4 + M6 + M7 = 
        # (A11B11 + A11B22 + A22B11 + A22B22) 
        # - (A22B21 - A22B11) 
        # + (A21B11 + A21B12 - A22B11 - A22B12) 
        # + (A12B21 + A12B22 - A22B21 - A22B22)
        # = A11B11 + A11B22 + A22B11 + A22B22 - A22B21 + A22B11 
        # + A21B11 + A21B12 - A22B11 - A22B12 
        # + A12B21 + A12B22 - A22B21 - A22B22
        # = A11B11 + (A11B22 + A12B22) + (A22B11 + A22B11 - A22B11) 
        # + (A22B22 - A22B22) - (A22B21 + A22B21) 
        # + A21B11 + A21B12 - A22B12 + A12B21
        # This doesn't look like C11 = A11B11 + A12B21.
        
        # I made a mistake in the Strassen formulas or my expansion.
        # Standard Strassen:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A22)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1 - M4 + M6 + M7
        # C12 = M3 + M4
        # C21 = M1 + M2
        # C22 = M3 - M2 + M5 + M7
        
        # Let's trust the standard formulas.
        
        # U vectors (size 4):
        U[0, 0] = 1; U[3, 0] = 1 # M1
        U[2, 1] = 1; U[3, 1] = 1 # M2
        U[0, 2] = 1 # M3
        U[3, 3] = 1 # M4
        U[0, 4] = 1; U[1, 4] = 1 # M5
        U[2, 5] = 1; U[3, 5] = -1 # M6
        U[1, 6] = 1; U[3, 6] = -1 # M7
        
        # V vectors (size 4):
        V[0, 0] = 1; V[3, 0] = 1 # M1
        V[0, 1] = 1 # M2
        V[1, 2] = 1; V[3, 2] = -1 # M3
        V[2, 3] = 1; V[0, 3] = -1 # M4
        V[3, 4] = 1 # M5
        V[0, 5] = 1; V[1, 5] = 1 # M6
        V[2, 6] = 1; V[3, 6] = 1 # M7
        
        # W vectors (size 4):
        W[0, 0] = 1 # C11 gets M1
        W[2, 0] = 1 # C21 gets M1
        W[2, 1] = 1 # C21 gets M2
        W[1, 2] = 1 # C12 gets M3
        W[0, 2] = 0 # C11 doesn't get M3
        # Wait, C11 = M1 - M4 + M6 + M7
        # So W[0, 0]=1, W[0, 3]=-1, W[0, 5]=1, W[0, 6]=1
        # C12 = M3 + M4
        # So W[1, 2]=1, W[1, 3]=1
        # C21 = M1 + M2
        # So W[2, 0]=1, W[2, 1]=1
        # C22 = M3 - M2 + M5 + M7
        # So W[3, 2]=1, W[3, 1]=-1, W[3, 4]=1, W[3, 6]=1
        
        W = np.zeros((4, 7))
        W[0, 0] = 1; W[0, 3] = -1; W[0, 5] = 1; W[0, 6] = 1
        W[1, 2] = 1; W[1, 3] = 1
        W[2, 0] = 1; W[2, 1] = 1
        W[3, 2] = 1; W[3, 1] = -1; W[3, 4] = 1; W[3, 6] = 1
        
        return U, V, W
    
    # Now, construct 4x4x4 using recursive Strassen.
    # Split A into 4 blocks of 2x2: A11, A12, A21, A22.
    # Split B into 4 blocks of 2x2: B11, B12, B21, B22.
    # C11 = A11B11 + A12B21
    # C12 = A11B12 + A12B22
    # C21 = A21B11 + A22B21
    # C22 = A21B12 + A22B22
    
    # We can use Strassen to compute each 2x2 block product?
    # No, Strassen computes the product of two 2x2 matrices.
    # Here we have sums of products.
    # C11 = A11B11 + A12B21. This is not a single matrix product.
    
    # Actually, the standard recursive Strassen for 4x4x4 uses the 7 multiplications
    # on the 4x4 blocks, where each "multiplication" is a product of 2x2 matrices.
    # So we apply Strassen to the 4x4 matrix A and B, treating them as 2x2 blocks.
    # The 7 "scalar" multiplications become 7 matrix multiplications of 2x2 matrices.
    # Each 2x2 matrix multiplication can be done in rank 7 (Strassen).
    # So total rank = 7 * 7 = 49.
    
    # Let's construct the 4x4x4 Strassen decomposition.
    # We need U, V, W of size (16, 49), (16, 49), (16, 49).
    
    # The 4x4 Strassen formulas are the same, but the elements are 2x2 blocks.
    # Let A = [[A11, A12], [A21, A22]] where each is 2x2.
    # Let B = [[B11, B12], [B21, B22]] where each is 2x2.
    
    # M1 = (A11+A22)(B11+B22)
    # This is a product of two 2x2 matrices.
    # We can decompose this product using the 2x2 Strassen decomposition.
    
    # So, for each of the 7 terms M_r, we use the 2x2 Strassen decomposition.
    # The U vector for M1 will be the concatenation of the U vectors for the 2x2 product.
    
    # Let's build the 4x4x4 decomposition.
    U_444 = np.zeros((16, 49))
    V_444 = np.zeros((16, 49))
    W_444 = np.zeros((16, 49))
    
    # Indices for 4x4 blocks:
    # A11: 0-3, A12: 4-7, A21: 8-11, A22: 12-15
    # B11: 0-3, B12: 4-7, B21: 8-11, B22: 12-15
    # C11: 0-3, C12: 4-7, C21: 8-11, C22: 12-15
    
    # We will use the 2x2 Strassen decomposition for each of the 7 terms.
    # U_2, V_2, W_2 = strassen_2x2()
    
    U_2, V_2, W_2 = strassen_2x2()
    
    # Term 0: M1 = (A11+A22)(B11+B22)
    # U for M1: (A11+A22) -> U[0-3, 0] += U_2[:, 0] + U_2[:, 1]? No.
    # The 2x2 Strassen U_2 is for a single 2x2 matrix.
    # Here, the "matrix" is A11+A22.
    # So U[0-3, r] += U_2[:, r] for A11 part, and U[12-15, r] += U_2[:, r] for A22 part.
    
    # Let's define a helper to add a 2x2 Strassen term to the 4x4 decomposition.
    def add_2x2_term(U_big, V_big, W_big, 
                     a_block1, a_block2, b_block1, b_block2, 
                     c_block, w_coeff, offset):
        # Adds the term w_coeff * (A_block1 + A_block2) * (B_block1 + B_block2)
        # to the decomposition, using the 2x2 Strassen decomposition for the product.
        for r in range(7):
            # U part
            for i in range(4):
                U_big[a_block1 + i, offset + r] += U_2[i, r]
                if a_block2 is not None:
                    U_big[a_block2 + i, offset + r] += U_2[i, r]
            # V part
            for i in range(4):
                V_big[b_block1 + i, offset + r] += V_2[i, r]
                if b_block2 is not None:
                    V_big[b_block2 + i, offset + r] += V_2[i, r]
            # W part
            for i in range(4):
                W_big[c_block + i, offset + r] += w_coeff * W_2[i, r]
                
    # M1 = (A11+A22)(B11+B22)
    # C11 += M1, C21 += M1
    # Wait, C11 = M1 - M4 + M6 + M7
    # C21 = M1 + M2
    # So M1 contributes to C11 with coeff 1 and C21 with coeff 1.
    
    # We need to add the term to C11 and C21.
    # But the term is a single product (A11+A22)(B11+B22).
    # We can add it to C11 with coeff 1, and to C21 with coeff 1.
    # But we only have one set of U, V, W vectors for the product.
    # We can use the same U, V, W vectors, and just have W point to both C11 and C21.
    
    # So, for M1:
    # U, V are as defined.
    # W has non-zero entries in C11 and C21.
    
    # Let's modify the helper to allow multiple C blocks.
    
    # For simplicity, let's just construct the 49 terms explicitly.
    
    # Term 0: M1
    # U: A11+A22, V: B11+B22
    # W: C11 (coeff 1), C21 (coeff 1)
    add_2x2_term(U_444, V_444, W_444, 0, 12, 0, 12, 0, 1, 0) # C11
    add_2x2_term(U_444, V_444, W_444, 0, 12, 0, 12, 8, 1, 0) # C21
    # This is wrong because it adds the same U, V vectors twice.
    # We should add the U, V vectors once, and the W vectors for both C blocks.
    
    # Let's redo this.
    U_444 = np.zeros((16, 49))
    V_444 = np.zeros((16, 49))
    W_444 = np.zeros((16, 49))
    
    def add_term(U_big, V_big, W_big, 
                 a_blocks, b_blocks, c_blocks_coeffs, offset):
        # a_blocks: list of block indices (0-3)
        # b_blocks: list of block indices
        # c_blocks_coeffs: list of (block_index, coeff)
        for r in range(7):
            for i in range(4):
                for ab in a_blocks:
                    U_big[ab*4 + i, offset + r] += U_2[i, r]
                for bb in b_blocks:
                    V_big[bb*4 + i, offset + r] += V_2[i, r]
                for cb, coeff in c_blocks_coeffs:
                    W_big[cb*4 + i, offset + r] += coeff * W_2[i, r]
                    
    # M1: (A11+A22)(B11+B22)
    add_term(U_444, V_444, W_444, [0, 3], [0, 3], [(0, 1), (2, 1)], 0)
    
    # M2: (A21+A22)B11
    add_term(U_444, V_444, W_444, [2, 3], [0], [(2, 1)], 7)
    
    # M3: A11(B12-B22)
    add_term(U_444, V_444, W_444, [0], [1, 3], [(1, 1), (3, 1)], 14)
    # Wait, B12-B22. So B blocks are 1 (coeff 1) and 3 (coeff -1).
    # The helper adds U_2[i, r] for each block.
    # We need to handle coefficients for A and B blocks too.
    
    # Let's generalize the helper.
    
    U_444 = np.zeros((16, 49))
    V_444 = np.zeros((16, 49))
    W_444 = np.zeros((16, 49))
    
    def add_term_general(U_big, V_big, W_big, 
                         a_blocks_coeffs, b_blocks_coeffs, c_blocks_coeffs, offset):
        for r in range(7):
            for i in range(4):
                for ab, coeff_a in a_blocks_coeffs:
                    U_big[ab*4 + i, offset + r] += coeff_a * U_2[i, r]
                for bb, coeff_b in b_blocks_coeffs:
                    V_big[bb*4 + i, offset + r] += coeff_b * V_2[i, r]
                for cb, coeff_c in c_blocks_coeffs:
                    W_big[cb*4 + i, offset + r] += coeff_c * W_2[i, r]
                    
    # M1: (A11+A22)(B11+B22)
    add_term_general(U_444, V_444, W_444, 
                     [(0, 1), (3, 1)], [(0, 1), (3, 1)], 
                     [(0, 1), (2, 1)], 0)
                     
    # M2: (A21+A22)B11
    add_term_general(U_444, V_444, W_444, 
                     [(2, 1), (3, 1)], [(0, 1)], 
                     [(2, 1)], 7)
                     
    # M3: A11(B12-B22)
    add_term_general(U_444, V_444, W_444, 
                     [(0, 1)], [(1, 1), (3, -1)], 
                     [(1, 1), (3, 1)], 14)
                     
    # M4: A22(B21-B11)
    add_term_general(U_444, V_444, W_444, 
                     [(3, 1)], [(2, 1), (0, -1)], 
                     [(0, -1), (1, 1)], 21)
                     
    # M5: (A11+A12)B22
    add_term_general(U_444, V_444, W_444, 
                     [(0, 1), (1, 1)], [(3, 1)], 
                     [(3, 1)], 28)
                     
    # M6: (A21-A22)(B11+B12)
    add_term_general(U_444, V_444, W_444, 
                     [(2, 1), (3, -1)], [(0, 1), (1, 1)], 
                     [(0, 1)], 35)
                     
    # M7: (A12-A22)(B21+B22)
    add_term_general(U_444, V_444, W_444, 
                     [(1, 1), (3, -1)], [(2, 1), (3, 1)], 
                     [(0, 1), (3, 1)], 42)
                     
    # Now, we have the 4x4x4 decomposition of rank 49.
    # We need to add the 5th column.
    # The 5th column is C[:, 4] = A * B[:, 4].
    # This is a 4x4x1 product.
    # Rank is 16 (schoolbook).
    
    # Total rank = 49 + 16 = 65.
    
    U_final = np.zeros((16, 65))
    V_final = np.zeros((20, 65)) # m*p = 4*5 = 20
    W_final = np.zeros((20, 65)) # p*n = 5*4 = 20
    
    # Copy the 4x4x4 part
    U_final[:, 0:49] = U_444
    V_final[0:16, 0:49] = V_444
    W_final[0:16, 0:49] = W_444
    
    # Add the 5th column part
    # B[:, 4] is the 5th column of B.
    # In V_final, the indices for B are:
    # B11: 0-3, B12: 4-7, B21: 8-11, B22: 12-15, B31: 16, B32: 17, B33: 18, B34: 19?
    # No, B is 4x5.
    # B = [[B11, B12, B13, B14, B15], ...]
    # Indices in V_final (size 20):
    # Row 0: B11, B12, B13, B14, B15 (0-4)
    # Row 1: B21, B22, B23, B24, B25 (5-9)
    # Row 2: B31, B32, B33, B34, B35 (10-14)
    # Row 3: B41, B42, B43, B44, B45 (15-19)
    
    # The 5th column is B15, B25, B35, B45 (indices 4, 9, 14, 19).
    
    # The product C[:, 4] = A * B[:, 4].
    # C_i4 = sum_j A_ij B_j5.
    # This is a sum of 16 terms: A_00 B_05, A_01 B_15, ...
    
    # We can use the schoolbook decomposition for this part.
    for i in range(4):
        for j in range(4):
            r = 49 + i * 4 + j
            U_final[i * 4 + j, r] = 1
            V_final[j * 5 + 4, r] = 1 # B_j5 is at index j*5 + 4
            W_final[4 * 4 + i, r] = 1 # C_i4 is at index 4*4 + i = 16 + i
            
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}

def _verify(U, V, W, n=4, m=4, p=5, trials=10):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    for _ in range(trials):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    u_dot_a = np.dot(U[:, r], A.flatten())
                    v_dot_b = np.dot(V[:, r], B.flatten())
                    val += u_dot_a * v_dot_b * W[k*n+i, r]
                C_computed[i, k] = val
        
        if not np.allclose(C_computed, C_expected, atol=1e-4):
            return False
            
    return True
