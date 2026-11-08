<<<<<<< SEARCH
def _build_block_decomposition(n, m, p):
    # n=6, m=6, p=6
    # Split into 4x4, 4x2, 2x4, 2x2 blocks
    # A00 (4x4), A01 (4x2), A10 (2x4), A11 (2x2)
    # B00 (4x4), B01 (4x2), B10 (2x4), B11 (2x2)
    
    # Products:
    # P1 = A00 B00 (4x4 * 4x4) -> Rank 16 (Naive)
    # P2 = A01 B10 (4x2 * 2x4) -> Rank 2 (Naive)
    # P3 = A00 B01 (4x4 * 4x2) -> Rank 4 (Naive)
    # P4 = A01 B11 (4x2 * 2x2) -> Rank 2 (Naive)
    # P5 = A10 B00 (2x4 * 4x4) -> Rank 2 (Naive)
    # P6 = A11 B10 (2x2 * 2x4) -> Rank 2 (Naive)
    # P7 = A10 B01 (2x4 * 4x2) -> Rank 2 (Naive)
    # P8 = A11 B11 (2x2 * 2x2) -> Rank 7 (Strassen)
    
    total_rank = 16 + 2 + 4 + 2 + 2 + 2 + 2 + 7
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_idx = 0
    
    # Helper to add a naive product of A[i1:i2, j1:j2] * B[j1:j2, k1:k2] to C[i1:i2, k1:k2]
    # But we need to map to the global U, V, W.
    
    # P1: A[0:4, 0:4] * B[0:4, 0:4] -> C[0:4, 0:4]
    # Naive 4x4
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[(0+i)*6 + (0+j)][col_idx] = 1
                V[(0+j)*6 + (0+k)][col_idx] = 1
                W[(0+k)*6 + (0+i)][col_idx] = 1
                col_idx += 1
                
    # P2: A[0:4, 4:6] * B[4:6, 0:4] -> C[0:4, 0:4]
    # A01 is 4x2, B10 is 2x4
    # C[0:4, 0:4] += A01 B10
    for i in range(4):
        for j in range(2): # j is 0..1 in A01, maps to 4..5 in A
            for k in range(4):
                U[(0+i)*6 + (4+j)][col_idx] = 1
                V[(4+j)*6 + (0+k)][col_idx] = 1
                W[(0+k)*6 + (0+i)][col_idx] = 1
                col_idx += 1
    # Wait, the loop structure for naive product of 4x2 * 2x4 is:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # j ranges 0..1.
    # So 4*1*4 = 16 terms? No.
    # The rank of 4x2*2x4 is 2?
    # C[i,k] = A[i,0]B[0,k] + A[i,1]B[1,k]
    # This is a sum of 2 rank-1 tensors?
    # Tensor T[i,j,k] where j is the shared index.
    # The decomposition is sum_j (A[:,j] B[j,:]).
    # Each term is a rank-1 tensor?
    # Yes.
    # So rank is 2.
    # My loop above adds 4*2*4 = 32 terms?
    # No, the loop `for i, for j, for k` adds 1 term for each (i,j,k).
    # This is the naive decomposition of the *full* product, which has rank 4*2*4 = 32?
    # No, the naive decomposition of m x n x p has rank m*n*p.
    # For 4x2*2x4, naive rank is 4*2*4 = 32.
    # But I said rank 2.
    # Why?
    # Because C[i,k] = sum_j A[i,j] B[j,k].
    # This is a linear combination of 2 matrices: A[:,0]B[0,:] and A[:,1]B[1,:].
    # So the tensor is a sum of 2 rank-1 tensors?
    # Yes.
    # So the rank is 2.
    # My loop `for i, for j, for k` creates 32 terms.
    # This is inefficient.
    # I need to add 2 terms.
    # Term 1: A[:,0] B[0,:]
    # Term 2: A[:,1] B[1,:]
    
    # Let's restart the construction with correct ranks.
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    col_idx = 0
    
    # P1: 4x4 * 4x4 (Naive, Rank 16)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i*6+j][col_idx] = 1
                V[j*6+k][col_idx] = 1
                W[k*6+i][col_idx] = 1
                col_idx += 1

    # P2: 4x2 * 2x4 (Rank 2)
    # Terms: j=0, j=1
    for j in range(2):
        for i in range(4):
            U[i*6 + (4+j)][col_idx] = 1
        for k in range(4):
            V[(4+j)*6 + k][col_idx] = 1
        for i in range(4):
            for k in range(4):
                W[k*6 + i][col_idx] = 1
        col_idx += 1

    # P3: 4x4 * 4x2 (Rank 4)
    # C[i,k] = sum_j A[i,j] B[j,k]
    # j=0..3
    for j in range(4):
        for i in range(4):
            U[i*6 + j][col_idx] = 1
        for k in range(2):
            V[j*6 + (4+k)][col_idx] = 1
        for i in range(4):
            for k in range(2):
                W[(4+k)*6 + i][col_idx] = 1
        col_idx += 1

    # P4: 4x2 * 2x2 (Rank 2)
    # C[i,k] = sum_j A[i,j] B[j,k]
    # j=0..1, k=0..1
    for j in range(2):
        for i in range(4):
            U[i*6 + (4+j)][col_idx] = 1
        for k in range(2):
            V[(4+j)*6 + (4+k)][col_idx] = 1
        for i in range(4):
            for k in range(2):
                W[(4+k)*6 + i][col_idx] = 1
        col_idx += 1

    # P5: 2x4 * 4x4 (Rank 2)
    # C[i,k] = sum_j A[i,j] B[j,k]
    # i=0..1, j=0..3, k=0..3
    # Rank 2?
    # C[i,k] = A[i,0]B[0,k] + ... + A[i,3]B[3,k]
    # This is sum of 4 terms?
    # No, the shared index is j.
    # j ranges 0..3.
    # So rank is 4?
    # Wait, A is 2x4, B is 4x4.
    # C is 2x4.
    # C[i,k] = sum_{j=0}^3 A[i,j] B[j,k].
    # This is a sum of 4 rank-1 tensors?
    # Yes.
    # So rank is 4.
    
    # I underestimated the ranks.
    # P5: 2x4 * 4x4 -> Rank 4.
    # P6: 2x2 * 2x4 -> Rank 2.
    # P7: 2x4 * 4x2 -> Rank 4.
    # P8: 2x2 * 2x2 -> Rank 7.
    
    # Total Rank = 16 + 2 + 4 + 2 + 4 + 2 + 4 + 7 = 41.
    
    # I will implement this correctly.
    
    U = [[0] * 41 for _ in range(n * m)]
    V = [[0] * 41 for _ in range(m * p)]
    W = [[0] * 41 for _ in range(p * n)]
    col_idx = 0
    
    # P1: 4x4 * 4x4 (Naive, Rank 16)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i*6+j][col_idx] = 1
                V[j*6+k][col_idx] = 1
                W[k*6+i][col_idx] = 1
                col_idx += 1

    # P2: 4x2 * 2x4 (Rank 2)
    for j in range(2):
        for i in range(4):
            U[i*6 + (4+j)][col_idx] = 1
        for k in range(4):
            V[(4+j)*6 + k][col_idx] = 1
        for i in range(4):
            for k in range(4):
                W[k*6 + i][col_idx] = 1
        col_idx += 1

    # P3: 4x4 * 4x2 (Rank 4)
    for j in range(4):
        for i in range(4):
            U[i*6 + j][col_idx] = 1
        for k in range(2):
            V[j*6 + (4+k)][col_idx] = 1
        for i in range(4):
            for k in range(2):
                W[(4+k)*6 + i][col_idx] = 1
        col_idx += 1

    # P4: 4x2 * 2x2 (Rank 2)
    for j in range(2):
        for i in range(4):
            U[i*6 + (4+j)][col_idx] = 1
        for k in range(2):
            V[(4+j)*6 + (4+k)][col_idx] = 1
        for i in range(4):
            for k in range(2):
                W[(4+k)*6 + i][col_idx] = 1
        col_idx += 1

    # P5: 2x4 * 4x4 (Rank 4)
    for j in range(4):
        for i in range(2):
            U[(4+i)*6 + j][col_idx] = 1
        for k in range(4):
            V[j*6 + k][col_idx] = 1
        for i in range(2):
            for k in range(4):
                W[k*6 + (4+i)][col_idx] = 1
        col_idx += 1

    # P6: 2x2 * 2x4 (Rank 2)
    for j in range(2):
        for i in range(2):
            U[(4+i)*6 + (4+j)][col_idx] = 1
        for k in range(4):
            V[(4+j)*6 + k][col_idx] = 1
        for i in range(2):
            for k in range(4):
                W[k*6 + (4+i)][col_idx] = 1
        col_idx += 1

    # P7: 2x4 * 4x2 (Rank 4)
    for j in range(4):
        for i in range(2):
            U[(4+i)*6 + j][col_idx] = 1
        for k in range(2):
            V[j*6 + (4+k)][col_idx] = 1
        for i in range(2):
            for k in range(2):
                W[(4+k)*6 + (4+i)][col_idx] = 1
        col_idx += 1

    # P8: 2x2 * 2x2 (Strassen, Rank 7)
    # Strassen patterns for 2x2
    # M0 = (A00+A11)(B00+B11)
    # ...
    # I will use the standard Strassen implementation for the 2x2 block.
    
    # Local indices: i,j,k in 0..1
    # Global indices: 4+i, 4+j, 4+k
    
    strassen_patterns = [
        # M0
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M1
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M2
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # M3
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M4
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M5
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M6
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    for u_coeffs, v_coeffs, w_coeffs in strassen_patterns:
        for (pi, pj), val in u_coeffs.items():
            U[(4+pi)*6 + (4+pj)][col_idx] += val
        for (pi, pj), val in v_coeffs.items():
            V[(4+pi)*6 + (4+pj)][col_idx] += val
        for (pi, pj), val in w_coeffs.items():
            W[(4+pj)*6 + (4+pi)][col_idx] += val
        col_idx += 1
        
    return {"U": U, "V": V, "W": W}
=======
def _build_strassen_2x2_3x3(n, m, p):
    # n=6, m=6, p=6
    # Block size 3
    # 2x2 blocks of 3x3 matrices
    # Strassen reduces 8 multiplications to 7
    # Each multiplication is 3x3 * 3x3, naive rank 27
    # Total rank = 7 * 27 = 189
    
    total_rank = 189
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_idx = 0
    block_size = 3
    
    # Helper to add a naive 3x3 * 3x3 product
    # U_block at (i_off, j_off), V_block at (j_off, k_off), W_block at (k_off, i_off)
    # The tensors are added to the global U, V, W at the specified column indices
    def add_naive_product(i_off, j_off, k_off, coeff_u, coeff_v, coeff_w):
        nonlocal col_idx
        for i in range(block_size):
            for j in range(block_size):
                for k in range(block_size):
                    # U coefficient
                    if coeff_u != 0:
                        U[(i_off + i) * m + (j_off + j)][col_idx] += coeff_u
                    # V coefficient
                    if coeff_v != 0:
                        V[(j_off + j) * p + (k_off + k)][col_idx] += coeff_v
                    # W coefficient
                    if coeff_w != 0:
                        W[(k_off + k) * n + (i_off + i)][col_idx] += coeff_w
                    col_idx += 1

    # Strassen formulas for 2x2 block multiplication
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # We implement each M_k as a sum of naive products.
    # Note: Since we are building the tensor decomposition, we cannot just "multiply" matrices.
    # We must expand the linear combinations.
    # E.g., M1 = (A11 + A22)(B11 + B22) = A11B11 + A11B22 + A22B11 + A22B22
    # Each term AxxByy is a 3x3 * 3x3 product.
    # So M1 contributes 4 naive products.
    # M2 = (A21 + A22)B11 = A21B11 + A22B11. (2 products)
    # M3 = A11(B12 - B22) = A11B12 - A11B22. (2 products)
    # M4 = A22(B21 - B11) = A22B21 - A22B11. (2 products)
    # M5 = (A11 + A12)B22 = A11B22 + A12B22. (2 products)
    # M6 = (A21 - A11)(B11 + B12) = A21B11 + A21B12 - A11B11 - A11B12. (4 products)
    # M7 = (A12 - A22)(B21 + B22) = A12B21 + A12B22 - A22B21 - A22B22. (4 products)
    
    # Total naive products: 4 + 2 + 2 + 2 + 2 + 4 + 4 = 20?
    # Wait, Strassen is 7 multiplications. If we expand them naively, we get more terms than the naive 8 multiplications (which is 8*27=216).
    # 20 * 27 = 540? No, that's wrong.
    # The "multiplication" in Strassen is an operation. In tensor decomposition, each "multiplication" of 3x3 matrices must be decomposed.
    # The rank of a 3x3 matrix multiplication is 27.
    # So if we use 7 such decompositions, the total rank is 7 * 27 = 189.
    # This assumes we can "combine" the linear terms within the U, V, W matrices for each of the 7 slots.
    # Yes, for each of the 7 Strassen steps, we define one set of U, V, W tensors.
    # For M1, the U tensor is (A11+A22), V is (B11+B22).
    # We can represent (A11+A22) as a single vector in the U space by setting coefficients to 1 in the A11 block and 1 in the A22 block.
    # So, for each Strassen term, we add ONE set of 27 rank-1 tensors (the naive decomposition of 3x3*3x3) to the global decomposition.
    
    # Let's define the 7 terms with their U, V, W block coefficients.
    # Block indices:
    # 11: (0,0), 12: (0,1), 21: (1,0), 22: (1,1)
    # Offsets:
    # 11: i=0, j=0, k=0
    # 12: i=0, j=3, k=0
    # 21: i=3, j=0, k=0
    # 22: i=3, j=3, k=0
    
    # Term 1: M1 = (A11 + A22)(B11 + B22)
    # U: A11 (1), A22 (1)
    # V: B11 (1), B22 (1)
    # W: C11 (1), C12 (1), C21 (1), C22 (1) -- Wait, W is the output contribution.
    # The output of M1 is added to C11, C12, C21, C22.
    # C11 += M1, C12 += M1, C21 += M1, C22 += M1.
    # So W should have 1s in C11, C12, C21, C22 blocks.
    
    def add_strassen_term(u_blocks, v_blocks, w_blocks):
        nonlocal col_idx
        # u_blocks: list of (block_idx, coeff)
        # v_blocks: list of (block_idx, coeff)
        # w_blocks: list of (block_idx, coeff)
        
        # We need to add 27 rank-1 tensors for this term.
        # The U vector for this term is fixed.
        # U_vec has length 36.
        # We fill U_vec, V_vec, W_vec based on the blocks.
        
        U_vec = [0.0] * (n * m)
        V_vec = [0.0] * (m * p)
        W_vec = [0.0] * (p * n)
        
        for (bi, bj), coeff in u_blocks:
            i_off = bi * block_size
            j_off = bj * block_size
            for i in range(block_size):
                for j in range(block_size):
                    U_vec[(i_off + i) * m + (j_off + j)] = coeff
        
        for (bj, bk), coeff in v_blocks:
            i_off = bj * block_size
            j_off = bk * block_size
            for i in range(block_size):
                for j in range(block_size):
                    V_vec[(i_off + i) * p + (j_off + j)] = coeff
        
        for (bk, bi), coeff in w_blocks:
            i_off = bk * block_size
            j_off = bi * block_size
            for i in range(block_size):
                for j in range(block_size):
                    W_vec[(i_off + i) * n + (j_off + j)] = coeff
        
        # Now add 27 rank-1 tensors using these vectors?
        # No, the rank-1 tensors are formed by the outer products of the basis vectors.
        # The naive decomposition of 3x3*3x3 uses 27 terms.
        # Term (i,j,k): U has 1 at (i,j), V has 1 at (j,k), W has 1 at (k,i).
        # But here U, V, W are linear combinations.
        # The tensor for this Strassen term is T = sum_{i,j,k} U[i,j] V[j,k] W[k,i] e_i e_j e_k?
        # No.
        # The decomposition is:
        # T = sum_{r=1}^{27} u_r \otimes v_r \otimes w_r
        # where u_r, v_r, w_r are the standard basis vectors scaled by the coefficients in U_vec, V_vec, W_vec?
        # No.
        # If U_vec = sum a_i e_i, V_vec = sum b_j f_j, W_vec = sum c_k g_k.
        # Then the product is (sum a_i e_i) (sum b_j f_j) (sum c_k g_k)?
        # No, the tensor is T_{ijk} = U_{ij} V_{jk} W_{ki}.
        # If we fix U, V, W as matrices, T is a fixed tensor.
        # We need to decompose THIS specific tensor T into rank-1 tensors.
        # The rank of a generic 3x3x3 tensor is 27? No, the rank of the multiplication tensor is 27.
        # But here T is NOT the multiplication tensor.
        # T is the tensor corresponding to the linear map (A,B) -> (A11+A22)(B11+B22) mapped to the output blocks.
        # This T is a sum of 4 multiplication tensors (A11B11, A11B22, A22B11, A22B22).
        # Each of those has rank 27.
        # So the total rank added by M1 is 4 * 27 = 108?
        # If so, total rank would be huge.
        
        # CORRECTION:
        # The Strassen algorithm works by reducing the number of *scalar* multiplications.
        # In tensor decomposition, we are looking for the rank of the tensor.
        # The rank of the 6x6 multiplication tensor is known to be at most 189 (using Strassen on 3x3 blocks, assuming 3x3 rank is 27).
        # Wait, if rank(3x3) = 27, then rank(6x6) <= 7 * 27 = 189.
        # This implies that the tensor T_M1 can be decomposed into 27 rank-1 tensors, not 108.
        # How?
        # Because the "multiplication" (A11+A22)(B11+B22) is a single bilinear operation in the context of the algorithm, but in tensor decomposition, we must express the entire tensor as a sum of rank-1 tensors.
        # The tensor T_M1 is defined by T_{ijk} = (A_{ij} + A_{i+3, j+3}) (B_{jk} + B_{j+3, k+3}) W_{ki} ...
        # Actually, T_M1 is a specific tensor.
        # It is the sum of 4 tensors: T_{11,11}, T_{11,22}, T_{22,11}, T_{22,22}.
        # Each T_{xx,yy} is a "shifted" multiplication tensor.
        # The rank of a sum of tensors is at most the sum of ranks.
        # So rank(T_M1) <= 4 * 27 = 108.
        # Then total rank <= 108 + 2*27 + 2*27 + 2*27 + 2*27 + 4*27 + 4*27 = 108 + 54 + 54 + 54 + 54 + 108 + 108 = 540.
        # This is worse than naive 216.
        
        # THERE IS A MISCONCEPTION.
        # Strassen's algorithm reduces the number of multiplications of *sub-matrices*.
        # If the sub-matrices are 3x3, and we use the optimal decomposition for 3x3 (rank 27), then the total rank is 7 * 27 = 189.
        # This is only true if the "multiplication" of the sub-matrices is performed using the rank-27 decomposition.
        # In the tensor decomposition of the 6x6 product, we are allowed to use ANY rank-1 tensors.
        # The construction is:
        # 1. Take the 7 Strassen terms.
        # 2. For each term, which involves multiplying two 3x3 sub-matrices (which are linear combinations of A's blocks), we use the rank-27 decomposition of the 3x3 multiplication tensor.
        # 3. The "inputs" to the 3x3 multiplication are not just A11, but A11+A22.
        # 4. However, the tensor decomposition is linear.
        # 5. We can simply take the 27 rank-1 tensors for the 3x3 multiplication, and "scale" or "shift" them appropriately.
        # 6. Specifically, for M1 = (A11+A22)(B11+B22), the tensor is T = T_{11,11} + T_{11,22} + T_{22,11} + T_{22,22}.
        # 7. We add the 27 tensors for T_{11,11}, then the 27 for T_{11,22}, etc.
        # 8. This adds 108 tensors.
        # 9. So the total rank is 540.
        
        # WAIT.
        # If I use Strassen, I am effectively saying I need 7 "black box" multiplications of 3x3 matrices.
        # If I replace each black box with a rank-27 decomposition, I add 27 rank-1 tensors per black box.
        # Total 7 * 27 = 189.
        # Why did I think it was 108 for M1?
        # Because M1 is ONE black box.
        # The black box computes (A11+A22)(B11+B22).
        # The tensor for this black box is the sum of 4 multiplication tensors.
        # Does the rank of this sum equal the rank of a single multiplication tensor?
        # No.
        # BUT, in the context of the Strassen algorithm, we are not decomposing the tensor T_M1 directly.
        # We are decomposing the overall 6x6 tensor.
        # The 6x6 tensor is a sum of 7 tensors T_M1 ... T_M7.
        # Each T_Mk is a "structured" tensor.
        # If we decompose T_M1 into 108 rank-1 tensors, and T_M2 into 54, etc., the total is 540.
        
        # Is it possible that rank(T_M1) < 108?
        # T_M1 = T_{11,11} + T_{11,22} + T_{22,11} + T_{22,22}.
        # These tensors have disjoint support in the U and V indices?
        # T_{11,11}: U has support in A11, V in B11, W in C blocks.
        # T_{11,22}: U in A11, V in B22.
        # They share U support (A11).
        # They are linearly independent.
        # So rank(T_M1) is likely 108.
        
        # Therefore, using Strassen with naive 3x3 decompositions gives rank 540, which is BAD.
        
        # The only way Strassen helps is if the sub-matrix multiplication rank is significantly lower than the naive rank, OR if we use a different strategy.
        # For 2x2, Strassen rank is 7. Naive is 8.
        # If we use 2x2 blocks (size 3), and assume rank(3x3) = R.
        # Naive 6x6 rank = 8 * R.
        # Strassen 6x6 rank = 7 * R.
        # So Strassen is always better IF R is the same.
        # My previous calculation of 540 vs 216 was based on R=27.
        # Naive 6x6 using 2x2 blocks of 3x3: 8 * 27 = 216.
        # Strassen 6x6 using 2x2 blocks of 3x3: 7 * 27 = 189.
        # Why did I think M1 adds 108?
        # Because I expanded M1 into 4 products.
        # But in the Strassen algorithm, M1 is ONE product.
        # The "product" is of the matrices (A11+A22) and (B11+B22).
        # The tensor corresponding to this product is T_{sum}.
        # Is rank(T_{sum}) = 27?
        # T_{sum} is the tensor for multiplying two 3x3 matrices X and Y, where X = A11+A22 and Y = B11+B22.
        # The multiplication tensor for 3x3 is T_{3x3}.
        # T_{sum} is just T_{3x3} evaluated on the subspace of X and Y.
        # Actually, T_{sum} IS T_{3x3} (just relabeled indices).
        # The rank is 27.
        # So M1 contributes 27 rank-1 tensors.
        # M2 contributes 27.
        # ...
        # M7 contributes 27.
        # Total 189.
        
        # So the implementation is:
        # For each Strassen term, we add 27 rank-1 tensors.
        # The U vector for these tensors is the identity-like pattern for the 3x3 block, BUT mapped to the global indices of the linear combination.
        # Wait.
        # If X = A11 + A22, then the U vector for the term "A11 part" is 1s in A11, and "A22 part" is 1s in A22.
        # But we only add 27 tensors.
        # How do we cover both A11 and A22?
        # We can't with just 27 tensors if they are disjoint.
        # The 27 tensors for 3x3 multiplication have U vectors that are basis vectors e_{ij} of the 3x3 space.
        # If X is a sum of two blocks, the U vectors must be in the span of the basis vectors of those blocks.
        # The rank of the multiplication tensor is 27.
        # This means there are 27 rank-1 tensors that sum to T_{3x3}.
        # If we "embed" T_{3x3} into the 6x6 space such that the input X is A11+A22, the U vectors will have support in A11 AND A22.
        # Specifically, for each of the 27 rank-1 tensors u_r \otimes v_r \otimes w_r, the vector u_r will have non-zero entries in both A11 and A22 blocks (specifically, the same pattern, just shifted).
        # So we add 27 tensors, each with U having support in 2 blocks (6 non-zeros), V in 2 blocks (6 non-zeros), W in 4 blocks (12 non-zeros).
        # This is valid.
        
        # So, for each Strassen term, we add 27 tensors.
        # Total 189.
        
        pass # This logic confirms 189 is achievable.
        
    # Implementation of the 7 terms with 27 tensors each.
    
    # Term 1: M1
    # U: A11 (1), A22 (1)
    # V: B11 (1), B22 (1)
    # W: C11 (1), C12 (1), C21 (1), C22 (1)
    _add_term(U, V, W, col_idx, 
              [(0,0,1), (1,1,1)], # U blocks (bi, bj, coeff)
              [(0,0,1), (1,1,1)], # V blocks (bj, bk, coeff)
              [(0,0,1), (0,1,1), (1,0,1), (1,1,1)]) # W blocks (bk, bi, coeff)
    col_idx += 27

    # Term 2: M2
    # U: A21 (1), A22 (1)
    # V: B11 (1)
    # W: C21 (1), C22 (1)
    _add_term(U, V, W, col_idx,
              [(1,0,1), (1,1,1)],
              [(0,0,1)],
              [(0,1,1), (1,1,1)])
    col_idx += 27

    # Term 3: M3
    # U: A11 (1)
    # V: B12 (1), B22 (-1)
    # W: C12 (1)
    _add_term(U, V, W, col_idx,
              [(0,0,1)],
              [(0,1,1), (1,1,-1)],
              [(0,1,1)])
    col_idx += 27

    # Term 4: M4
    # U: A22 (1)
    # V: B21 (1), B11 (-1)
    # W: C22 (1)
    _add_term(U, V, W, col_idx,
              [(1,1,1)],
              [(1,0,1), (0,0,-1)],
              [(1,1,1)])
    col_idx += 27

    # Term 5: M5
    # U: A11 (1), A12 (1)
    # V: B22 (1)
    # W: C12 (1), C22 (1)
    _add_term(U, V, W, col_idx,
              [(0,0,1), (0,1,1)],
              [(1,1,1)],
              [(0,1,1), (1,1,1)])
    col_idx += 27

    # Term 6: M6
    # U: A21 (1), A11 (-1)
    # V: B11 (1), B12 (1)
    # W: C22 (1)
    _add_term(U, V, W, col_idx,
              [(1,0,1), (0,0,-1)],
              [(0,0,1), (0,1,1)],
              [(1,1,1)])
    col_idx += 27

    # Term 7: M7
    # U: A12 (1), A22 (-1)
    # V: B21 (1), B22 (1)
    # W: C11 (1)
    _add_term(U, V, W, col_idx,
              [(0,1,1), (1,1,-1)],
              [(1,0,1), (1,1,1)],
              [(0,0,1)])
    col_idx += 27

    return {"U": U, "V": V, "W": W}

def _add_term(U, V, W, start_col, u_blocks, v_blocks, w_blocks):
    # u_blocks: list of (bi, bj, coeff)
    # v_blocks: list of (bj, bk, coeff)
    # w_blocks: list of (bk, bi, coeff)
    # Adds 27 rank-1 tensors for a 3x3 multiplication
    # The 27 tensors are the naive decomposition of 3x3*3x3.
    # For each (i,j,k) in 0..2:
    #   u_r has coeff at (bi+i, bj+j) for each (bi,bj,coeff) in u_blocks?
    #   No.
    #   The U vector for the r-th tensor is formed by taking the basis vector e_{ij} of the 3x3 space
    #   and mapping it to the global space using the block offsets and coefficients.
    #   So U_global[i_global, j_global] = sum_{(bi,bj,c) in u_blocks} c * delta(i_global, bi*3+i) * delta(j_global, bj*3+j)
    
    for i in range(3):
        for j in range(3):
            for k in range(3):
                col = start_col + (i*3 + j)*3 + k # 0..26
                
                # Build U vector
                for (bi, bj, coeff) in u_blocks:
                    gi = bi * 3 + i
                    gj = bj * 3 + j
                    U[gi * 6 + gj][col] += coeff
                    
                # Build V vector
                for (bj, bk, coeff) in v_blocks:
                    gi = bj * 3 + j
                    gj = bk * 3 + k
                    V[gi * 6 + gj][col] += coeff
                    
                # Build W vector
                for (bk, bi, coeff) in w_blocks:
                    gi = bk * 3 + k
                    gj = bi * 3 + i
                    W[gi * 6 + gj][col] += coeff

>>>>>>> REPLACE

