import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # We will build the decomposition manually
    # Rank will be accumulated
    U = []
    V = []
    W = []
    
    def add_term(u_vec, v_vec, w_vec):
        U.append(u_vec)
        V.append(v_vec)
        W.append(w_vec)

    # Helper to create a vector of zeros with a 1 at specific indices
    def sparse_vec(length, indices):
        vec = [0.0] * length
        for idx in indices:
            vec[idx] = 1.0
        return vec

    # Dimensions of U, V, W
    dim_U = n * m # 20
    dim_V = m * p # 30
    dim_W = p * n # 24

    # 1. A[4x4] * B[4x4] part
    # A is 4x5, so A[4x4] uses cols 0-3 of A.
    # B is 5x6, so B[4x6] uses rows 0-3 of B.
    # We split B[4x6] into B[4x4] (cols 0-3) and B[4x2] (cols 4-5).
    # So Term 1: A[4x4] * B[4x4] -> C[4x4] (cols 0-3 of C)
    # Term 2: A[4x4] * B[4x2] -> C[4x2] (cols 4-5 of C)
    # Term 3: A[4x1] * B[1x6] -> C[4x6] (all cols of C)
    
    # Indices in A (4x5): row i, col j -> i*5 + j
    # Indices in B (5x6): row j, col k -> j*6 + k
    # Indices in C (6x4): row k, col i -> k*4 + i (Trace convention W[k*n+i,r])
    
    # --- Term 1: Strassen for 4x4 x 4x4 ---
    # A blocks:
    # A11: rows 0-1, cols 0-1
    # A12: rows 0-1, cols 2-3
    # A21: rows 2-3, cols 0-1
    # A22: rows 2-3, cols 2-3
    
    # B blocks (B is 4x4 here, so rows 0-3, cols 0-3):
    # B11: rows 0-1, cols 0-1
    # B12: rows 0-1, cols 2-3
    # B21: rows 2-3, cols 0-1
    # B22: rows 2-3, cols 2-3
    
    # Strassen intermediates:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # Each M is a 2x2 matrix product.
    # We implement each 2x2 product with 4 scalar mults (schoolbook).
    
    def add_2x2_product(A_rows, A_cols, B_rows, B_cols, C_row_offset, C_col_offset, sign=1.0):
        # A is 2x2 defined by A_rows, A_cols in global A
        # B is 2x2 defined by B_rows, B_cols in global B
        # C is 2x2 defined by C_row_offset, C_col_offset in global C
        # Adds 4 terms
        for i in range(2):
            for k in range(2):
                # C[i, k] += A[i, j] * B[j, k]
                # Sum over j=0,1
                for j in range(2):
                    a_idx = A_rows[i] * 5 + A_cols[j]
                    b_idx = B_rows[j] * 6 + B_cols[k]
                    c_idx = (C_row_offset + i) * 4 + (C_col_offset + k)
                    
                    u = sparse_vec(dim_U, [a_idx])
                    v = sparse_vec(dim_V, [b_idx])
                    w = sparse_vec(dim_W, [c_idx])
                    # Scale by sign
                    # Since we are adding terms, we just add the basis vector
                    # But Strassen has additions and subtractions.
                    # We can handle signs by multiplying the W vector by the sign.
                    # Or better, keep U,V positive and adjust W.
                    # The tensor entry is W * U * V.
                    # So if we need -1, we put -1 in W.
                    
                    w_scaled = [x * sign for x in w]
                    add_term(u, v, w_scaled)

    # We need to handle the sums in Strassen.
    # M1 = (A11 + A22)(B11 + B22)
    # = A11 B11 + A11 B22 + A22 B11 + A22 B22
    # This expands to 4 * 4 = 16 scalar mults?
    # No, Strassen reduces 8 block mults to 7.
    # Each block mult is 2x2.
    # If we expand M1 into scalar mults, we are doing 4 scalar mults for each of the 4 block products?
    # No, (A11+A22) is a 2x2 matrix. (B11+B22) is a 2x2 matrix.
    # Their product is a 2x2 matrix.
    # We can compute this 2x2 product using 4 scalar mults IF we pre-compute the sums?
    # No, the inputs are linear combinations.
    # U vectors will be sums of A entries.
    # V vectors will be sums of B entries.
    
    # Let's implement a generic "add_linear_combination_product"
    
    def add_product(
        A_coeffs, # list of (coeff, idx)
        B_coeffs, # list of (coeff, idx)
        C_idx,
        C_coeff = 1.0
    ):
        # Adds term: (sum a_i A_i) * (sum b_j B_j) * C_coeff to C_idx
        # This is not a rank 1 term if A_coeffs or B_coeffs have >1 element.
        # We need to expand this into rank 1 terms.
        # (sum a_i A_i) (sum b_j B_j) = sum_i sum_j a_i b_j A_i B_j
        for ca, ia in A_coeffs:
            for cb, ib in B_coeffs:
                u = sparse_vec(dim_U, [ia])
                v = sparse_vec(dim_V, [ib])
                w = sparse_vec(dim_W, [C_idx])
                w_scaled = [x * (ca * cb * C_coeff) for x in w]
                add_term(u, v, w_scaled)

    # Define indices
    # A: 4x5
    # A11: (0,0),(0,1),(1,0),(1,1) -> idx 0,1,5,6
    # A12: (0,2),(0,3),(1,2),(1,3) -> idx 2,3,7,8
    # A21: (2,0),(2,1),(3,0),(3,1) -> idx 10,11,15,16
    # A22: (2,2),(2,3),(3,2),(3,3) -> idx 12,13,17,18
    
    # B: 5x6
    # B11 (for 4x4 part): rows 0-1, cols 0-1 -> idx 0,1,6,7
    # B12: rows 0-1, cols 2-3 -> idx 2,3,8,9
    # B21: rows 2-3, cols 0-1 -> idx 12,13,18,19
    # B22: rows 2-3, cols 2-3 -> idx 14,15,20,21
    
    # C: 6x4
    # C11: rows 0-1, cols 0-1 -> idx 0,1,4,5
    # C12: rows 0-1, cols 2-3 -> idx 2,3,6,7
    # C21: rows 2-3, cols 0-1 -> idx 8,9,12,13
    # C22: rows 2-3, cols 2-3 -> idx 10,11,14,15

    # Strassen Formulas
    # M1 = (A11 + A22)(B11 + B22)
    # C11 += M1, C12 += M1 (no, C11 = M1 + M4 - M5 + M7, etc)
    
    # Let's just add all 7 M's and then combine in C.
    # But C is a single matrix. We can just add the contributions to C directly.
    
    # M1 contributes to C11 (+1), C12 (0? No, Strassen output mapping)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # So we need to add M1 to C11 and C22.
    # M1 is a 2x2 matrix.
    # M1[i,k] = sum_j (A11[i,j] + A22[i,j]) (B11[j,k] + B22[j,k])
    # This is a product of two 2x2 matrices.
    # We can use add_product for each entry of M1.
    
    def add_strassen_block(A1_indices, A2_indices, B1_indices, B2_indices, C_indices, C_sign):
        # A = A1 + A2, B = B1 + B2
        # C += sign * (A1+A2)(B1+B2)
        # A1, A2 are 2x2 lists of indices
        # We expand (A1+A2)(B1+B2) = A1B1 + A1B2 + A2B1 + A2B2
        # For each entry (i,k) of the result:
        for i in range(2):
            for k in range(2):
                c_idx = C_indices[i*2+k]
                # Sum over j
                # We need to sum over j of (A1[i,j]+A2[i,j]) * (B1[j,k]+B2[j,k])
                # This is sum_j (A1[i,j]B1[j,k] + A1[i,j]B2[j,k] + A2[i,j]B1[j,k] + A2[i,j]B2[j,k])
                for j in range(2):
                    a1_idx = A1_indices[i*2+j]
                    a2_idx = A2_indices[i*2+j]
                    b1_idx = B1_indices[j*2+k]
                    b2_idx = B2_indices[j*2+k]
                    
                    # Add 4 terms
                    add_product([(1.0, a1_idx)], [(1.0, b1_idx)], c_idx, C_sign)
                    add_product([(1.0, a1_idx)], [(1.0, b2_idx)], c_idx, C_sign)
                    add_product([(1.0, a2_idx)], [(1.0, b1_idx)], c_idx, C_sign)
                    add_product([(1.0, a2_idx)], [(1.0, b2_idx)], c_idx, C_sign)

    # This will generate 4 * 4 * 4 = 64 terms for M1?
    # No, add_product generates 1 term per pair.
    # Inside the loop: 4 calls to add_product.
    # Each add_product with single coeffs generates 1 term.
    # So 4 terms per (i,k,j).
    # 2*2*2 = 8 (i,k,j) combos.
    # 8 * 4 = 32 terms for M1.
    # This is just schoolbook for the expanded blocks.
    # M1 is effectively a 2x2 mult of sums.
    # Rank 4 for 2x2 mult.
    # But here we are expanding the sums first.
    # (A1+A2) is a 2x2 matrix.
    # If we treat (A1+A2) as the input, the rank is 4.
    # But we don't have (A1+A2) as a single input, we have A1 and A2.
    # So we must expand.
    # So M1 costs 32 scalar mults?
    # Wait.
    # Schoolbook for 4x4x4 is 64.
    # Strassen says 7 block mults.
    # If each block mult costs 4 scalar mults, total 28.
    # Why did I get 32 for M1?
    # M1 = (A11+A22)(B11+B22).
    # This is ONE block multiplication.
    # The inputs are linear combinations.
    # The number of scalar multiplications required is the rank of the tensor for the block multiplication, which is 4 (for 2x2).
    # So M1 should cost 4 terms?
    # No, the inputs are not single variables.
    # U vector for a term in M1 will have support on A11 and A22.
    # V vector will have support on B11 and B22.
    # So we can use rank 4 decomposition of 2x2x2 tensor.
    # Let's use the 4 terms of schoolbook for 2x2.
    # Term 1: (A11[0,0]+A22[0,0]) * (B11[0,0]+B22[0,0]) * C[0,0]
    # This is not a rank 1 term in the original A,B.
    # It is a product of two linear forms.
    # We can decompose this product of linear forms into sum of rank 1 tensors?
    # (x+y)(z+w) = xz + xw + yz + yw.
    # So yes, it expands to 4 rank 1 terms.
    # So M1 costs 4 * 4 = 16 terms?
    # No, for each entry of the 2x2 result, we have this expansion.
    # M1 has 4 entries.
    # Each entry expansion costs 4 terms.
    # Total 16 terms for M1.
    # 7 * 16 = 112 terms?
    # This is worse than schoolbook 64!
    
    # Ah, Strassen reduces the number of *block* multiplications.
    # If the block multiplication is done with rank 4, and we have 7 blocks, we need 28 "block mults".
    # But each "block mult" in Strassen is on linear combinations.
    # If we implement the block mult with schoolbook (rank 4), we are doing 4 scalar mults per block entry?
    # No.
    # The standard Strassen algorithm uses 7 multiplications of 2x2 matrices.
    # If we implement 2x2 matrix multiplication with rank 4 (schoolbook), then each of the 7 multiplications requires 4 scalar multiplications.
    # Total 28 scalar multiplications.
    # My expansion above gave 16 for M1. Why?
    # Because I expanded (A1+A2)(B1+B2) into 4 terms for EACH entry.
    # But schoolbook for 2x2 uses 4 terms TOTAL for the whole 2x2 product?
    # No, schoolbook for 2x2 uses 4 terms for EACH entry?
    # C[i,k] = sum_j A[i,j] B[j,k].
    # There are 4 entries. Each is a sum of 2 products.
    # Total 8 products?
    # No, 2x2 matrix multiplication has rank 4?
    # M(2,2,2) rank is 7? No, rank is 4 for 2x2x2?
    # Wait. M(2,2,2) is the tensor for multiplying 2x2 matrices.
    # The rank of M(2,2,2) is 7? No, it is 7 for 2x2?
    # No, Strassen proves rank <= 7.
    # Schoolbook for 2x2 is 4?
    # C11 = A11B11 + A12B21
    # C12 = A11B12 + A12B22
    # C21 = A21B11 + A22B21
    # C22 = A21B12 + A22B22
    # Total 8 products.
    # So schoolbook rank is 8.
    # Strassen rank is 7.
    
    # So for 4x4x4:
    # Schoolbook: 64.
    # Strassen (7 blocks, each rank 8): 56.
    # Strassen (7 blocks, each rank 7): 49.
    
    # So we should use Strassen for the inner 2x2 blocks too?
    # Or just use rank 7 for the inner blocks.
    
    # If we use rank 7 for inner blocks, total rank 49.
    # If we use rank 8 (schoolbook) for inner blocks, total rank 56.
    
    # 49 is better.
    
    # So we need to implement M(2,2,2) with rank 7.
    
    # Let's stick to the simpler 56 (Strassen outer, Schoolbook inner) first to ensure correctness, then optimize if needed.
    # 56 + 32 + 24 = 112.
    # 112 < 120.
    
    # Let's implement the 56 rank version.

    # Indices for 2x2 blocks
    # A11: [0, 1, 5, 6]
    # A12: [2, 3, 7, 8]
    # A21: [10, 11, 15, 16]
    # A22: [12, 13, 17, 18]
    
    # B11: [0, 1, 6, 7]
    # B12: [2, 3, 8, 9]
    # B21: [12, 13, 18, 19]
    # B22: [14, 15, 20, 21]
    
    # C11: [0, 1, 4, 5]
    # C12: [2, 3, 6, 7]
    # C21: [8, 9, 12, 13]
    # C22: [10, 11, 14, 15]

    # Function to add a 2x2 product using schoolbook (8 terms)
    def add_2x2_schoolbook(A_idx, B_idx, C_idx, sign=1.0):
        # A_idx, B_idx, C_idx are lists of 4 indices (row-major)
        for i in range(2):
            for k in range(2):
                for j in range(2):
                    a = A_idx[i*2+j]
                    b = B_idx[j*2+k]
                    c = C_idx[i*2+k]
                    u = sparse_vec(dim_U, [a])
                    v = sparse_vec(dim_V, [b])
                    w = sparse_vec(dim_W, [c])
                    w_scaled = [x * sign for x in w]
                    add_term(u, v, w_scaled)

    # Strassen M1 = (A11 + A22)(B11 + B22)
    # We need to add M1 to C11 and C22.
    # But M1 is a 2x2 matrix.
    # We can't just call add_2x2_schoolbook because the inputs are sums.
    # We must expand.
    # (A11+A22)(B11+B22) = A11B11 + A11B22 + A22B11 + A22B22
    # Each term is a 2x2 product.
    # So we add 4 schoolbook products.
    
    # M1 contributes +1 to C11 and +1 to C22.
    
    # Add A11B11 to C11
    add_2x2_schoolbook(A11, B11, C11, 1.0)
    # Add A11B22 to C11
    add_2x2_schoolbook(A11, B22, C11, 1.0)
    # Add A22B11 to C11
    add_2x2_schoolbook(A22, B11, C11, 1.0)
    # Add A22B22 to C11
    add_2x2_schoolbook(A22, B22, C11, 1.0)
    
    # Same for C22
    add_2x2_schoolbook(A11, B11, C22, 1.0)
    add_2x2_schoolbook(A11, B22, C22, 1.0)
    add_2x2_schoolbook(A22, B11, C22, 1.0)
    add_2x2_schoolbook(A22, B22, C22, 1.0)
    
    # This is 8 * 8 = 64 terms just for M1?
    # No, add_2x2_schoolbook adds 8 terms.
    # 4 calls * 8 terms = 32 terms for M1 contribution to C11.
    # 32 terms for C22.
    # Total 64 terms for M1.
    # This is way too many.
    
    # The issue is that I am not using the Strassen identity correctly.
    # Strassen computes M1 = (A11+A22)(B11+B22) as a SINGLE 2x2 product.
    # If I use schoolbook for that single 2x2 product, it takes 8 terms.
    # But the inputs are linear combinations.
    # So I need to decompose the product of two linear combination matrices into rank 1 terms.
    # This is exactly what add_product does, but for matrices.
    
    # Let's use the add_product helper for each entry of the 2x2 result.
    
    def add_strassen_term(A1_idx, A2_idx, B1_idx, B2_idx, C_idx_list, sign):
        # Adds sign * (A1+A2)(B1+B2) to C
        for i in range(2):
            for k in range(2):
                c_idx = C_idx_list[i*2+k]
                for j in range(2):
                    # Term: (A1[i,j]+A2[i,j]) * (B1[j,k]+B2[j,k])
                    # = A1[i,j]B1[j,k] + A1[i,j]B2[j,k] + A2[i,j]B1[j,k] + A2[i,j]B2[j,k]
                    a1 = A1_idx[i*2+j]
                    a2 = A2_idx[i*2+j]
                    b1 = B1_idx[j*2+k]
                    b2 = B2_idx[j*2+k]
                    
                    add_product([(1.0, a1)], [(1.0, b1)], c_idx, sign)
                    add_product([(1.0, a1)], [(1.0, b2)], c_idx, sign)
                    add_product([(1.0, a2)], [(1.0, b1)], c_idx, sign)
                    add_product([(1.0, a2)], [(1.0, b2)], c_idx, sign)

    # This generates 4 terms per (i,k,j).
    # 2*2*2 = 8 combos.
    # 8 * 4 = 32 terms per Strassen intermediate M.
    # 7 M's * 32 = 224 terms.
    # This is worse than schoolbook 64.
    
    # Why? Because I am expanding the sums.
    # Strassen reduces the number of multiplications from 8 to 7.
    # If each multiplication is rank 4 (for 2x2? No, rank 8 for 2x2 schoolbook), then 7*8=56.
    # My expansion gives 32 per M.
    # 32 is 4 times 8.
    # Ah, (A1+A2) is a 2x2 matrix.
    # The product (A1+A2)(B1+B2) is a 2x2 matrix.
    # The rank of this tensor is 8 (schoolbook).
    # So it should take 8 terms.
    # Why did I get 32?
    # Because I expanded (A1+A2) into A1 and A2 separately.
    # (A1+A2)(B1+B2) = A1B1 + A1B2 + A2B1 + A2B2.
    # Each A1B1 is a 2x2 product (rank 8).
    # So 4 * 8 = 32.
    # This is correct.
    # So Strassen does NOT reduce the rank if you expand the linear combinations into the schoolbook basis.
    # It only reduces the rank if you can multiply the linear combinations directly.
    # But in a tensor decomposition, the inputs are the original variables.
    # So you MUST expand.
    # Therefore, Strassen does not help for the rank of the tensor if the block size is 2x2 and we use schoolbook for the blocks?
    # No, Strassen is a valid tensor decomposition.
    # The rank of M(4,4,4) is at most 28?
    # Let's check the literature.
    # R(4) <= 23.
    # R(2) = 7.
    # R(4) <= R(2)^2 = 49?
    # Actually, the Strassen algorithm for 4x4 uses 7 multiplications of 2x2 matrices.
    # If R(2)=7, then R(4) <= 49.
    # If R(2)=4 (which is false, R(2)=7), then R(4) <= 28.
    # So R(4) <= 49 is the bound from Strassen.
    # So my 49 rank construction is valid.
    # The 224 terms I calculated were for the EXPANSION into scalar mults.
    # But the rank is the number of terms in the DECOMPOSITION.
    # If I use the 7-term Strassen decomposition for the 2x2 blocks, then each M_k is a sum of 7 rank-1 tensors.
    # So total 7 * 7 = 49 rank-1 tensors.
    
    # So I need to implement M(2,2,2) with rank 7.
    
    # Let's do that.

    # M(2,2,2) Strassen:
    # M1 = (A11+A22)(B11+B22)
    # ...
    # This is recursive.
    # I will implement the 49 rank version for 4x4x4.

    # This is getting complex to code from scratch correctly in one go.
    # Given the time, I will stick to the 105 rank version (28+32+24) which is simpler and definitely an improvement.
    # Wait, 28 is not correct for 4x4x4 if R(2)=7.
    # 28 assumes R(2)=4.
    # Since R(2)=7, the rank is 49.
    # So 49 + 32 + 24 = 105.
    
    # I will implement the 105 rank version.
    
    # For the 4x4x4 part, I will use the 49 rank Strassen construction.
    # I need a helper to add a rank 7 decomposition of 2x2x2.

    def add_2x2_strassen(A_idx, B_idx, C_idx, sign=1.0):
        # A_idx, B_idx, C_idx are lists of 4 indices
        # Implements Strassen for 2x2
        # M1 = (A0+A3)(B0+B3)
        # M2 = (A2+A3)B0
        # M3 = A0(B1-B3)
        # M4 = A3(B2-B0)
        # M5 = (A0+A1)B3
        # M6 = (A2-A0)(B0+B1)
        # M7 = (A1-A3)(B2+B3)
        
        # C0 = M1 + M4 - M5 + M7
        # C1 = M3 + M5
        # C2 = M2 + M4
        # C3 = M1 - M2 + M3 + M6
        
        # Each M is a product of two linear combinations.
        # We use add_product for each scalar term in the M's.
        
        # M1: (A0+A3)(B0+B3)
        # Adds to C0 and C3
        add_product([(1.0, A_idx[0]), (1.0, A_idx[3])], [(1.0, B_idx[0]), (1.0, B_idx[3])], C_idx[0], sign)
        add_product([(1.0, A_idx[0]), (1.0, A_idx[3])], [(1.0, B_idx[0]), (1.0, B_idx[3])], C_idx[3], sign)
        
        # M2: (A2+A3)B0
        # Adds to C2
        add_product([(1.0, A_idx[2]), (1.0, A_idx[3])], [(1.0, B_idx[0])], C_idx[2], sign)
        
        # M3: A0(B1-B3)
        # Adds to C1 and C3
        add_product([(1.0, A_idx[0])], [(1.0, B_idx[1]), (-1.0, B_idx[3])], C_idx[1], sign)
        add_product([(1.0, A_idx[0])], [(1.0, B_idx[1]), (-1.0, B_idx[3])], C_idx[3], sign)
        
        # M4: A3(B2-B0)
        # Adds to C0 and C2
        add_product([(1.0, A_idx[3])], [(1.0, B_idx[2]), (-1.0, B_idx[0])], C_idx[0], sign)
        add_product([(1.0, A_idx[3])], [(1.0, B_idx[2]), (-1.0, B_idx[0])], C_idx[2], sign)
        
        # M5: (A0+A1)B3
        # Adds to C0 and C1
        add_product([(1.0, A_idx[0]), (1.0, A_idx[1])], [(1.0, B_idx[3])], C_idx[0], -sign)
        add_product([(1.0, A_idx[0]), (1.0, A_idx[1])], [(1.0, B_idx[3])], C_idx[1], sign)
        
        # M6: (A2-A0)(B0+B1)
        # Adds to C3
        add_product([(1.0, A_idx[2]), (-1.0, A_idx[0])], [(1.0, B_idx[0]), (1.0, B_idx[1])], C_idx[3], sign)
        
        # M7: (A1-A3)(B2+B3)
        # Adds to C0
        add_product([(1.0, A_idx[1]), (-1.0, A_idx[3])], [(1.0, B_idx[2]), (1.0, B_idx[3])], C_idx[0], sign)

    # Now use this for the 4x4x4 part.
    # A11, A12, A21, A22 are 2x2 blocks.
    # We call add_2x2_strassen for each block product.
    
    # M1 = (A11+A22)(B11+B22)
    # This is a 2x2 product.
    # But A11+A22 is a 2x2 matrix.
    # We can't pass A11+A22 as a single index list.
    # We must expand.
    # (A11+A22)(B11+B22) = A11B11 + A11B22 + A22B11 + A22B22
    # Each term is a 2x2 product.
    # So we call add_2x2_strassen 4 times?
    # No, add_2x2_strassen takes index lists.
    # We need to pass the indices for A11, etc.
    # But A11+A22 is a sum.
    # So we must expand into 4 calls to add_2x2_strassen?
    # No, add_2x2_strassen implements the 7-term decomposition.
    # If we call it 4 times, we are doing 4 * 7 = 28 terms for M1.
    # This matches the 49 total (7 M's * 7 terms).
    
    # So for M1:
    add_2x2_strassen(A11, B11, C11, 1.0)
    add_2x2_strassen(A11, B22, C11, 1.0)
    add_2x2_strassen(A22, B11, C11, 1.0)
    add_2x2_strassen(A22, B22, C11, 1.0)
    
    # And so on for all 7 M's.
    
    # This will be 7 * 4 = 28 calls to add_2x2_strassen.
    # Each call adds 7 * (expansion of add_product) terms.
    # add_product with 2 coeffs in A and 2 in B adds 4 terms.
    # So each add_2x2_strassen call adds 7 * 4 = 28 terms?
    # No, add_product is called 7 times inside add_2x2_strassen.
    # Each add_product call adds (len(A_coeffs) * len(B_coeffs)) terms.
    # So total terms per add_2x2_strassen call is sum of len(A)*len(B) for the 7 M's.
    # M1: 2*2=4. M2: 2*1=2. M3: 1*2=2. M4: 1*2=2. M5: 2*1=2. M6: 2*2=4. M7: 2*2=4.
    # Total 20 terms per add_2x2_strassen call.
    # 28 calls * 20 = 560 terms.
    # This is huge.
    
    # I will just output the 120 schoolbook solution to be safe, as the complex Strassen implementation is error-prone.
    # Or, I can output the 105 solution if I can simplify.
    
    # Given the risk, I will stick to the current 120 solution.
    
    U_arr = np.zeros((dim_U, 120))
    V_arr = np.zeros((dim_V, 120))
    W_arr = np.zeros((dim_W, 120))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_arr[i * m + j, col] = 1.0
                V_arr[j * p + k, col] = 1.0
                W_arr[k * n + i, col] = 1.0
                col += 1
                
    return {"U": U_arr.tolist(), "V": V_arr.tolist(), "W": W_arr.tolist()}

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    # Convert to list of lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
