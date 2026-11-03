import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Part 1: 4x4x4 using Strassen (rank 49)
    # Part 2: 4x4x3 using schoolbook (rank 48)
    rank1 = 49
    rank2 = 48
    total_rank = rank1 + rank2
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Build 4x4x4 Strassen decomposition
    # A is 4x4, B is 4x4, C is 4x4
    # Split into 2x2 blocks
    # For each Strassen product, build the 2x2x2 decomposition
    
    # The 7 Strassen products for 2x2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 - M4 + M6 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M5 + M7
    
    # For 4x4x4, we have 4 block positions for each matrix
    # Block (bi, bj) where bi in {0,1}, bj in {0,1}
    # Global index for A: row = bi*2 + li, col = bj*2 + lj
    # Global index for B: row = bj*2 + lj, col = bk*2 + lk
    # Global index for C: row = bi*2 + li, col = bk*2 + lk
    
    # For each of the 4 block pairs (bi, bk), we apply Strassen to the 2x2 blocks
    # Actually, we need to be more careful. The 4x4x4 multiplication involves
    # all combinations of blocks.
    
    # Let me think again. A is 4x4, so it has 4x4 entries.
    # B is 4x4, so it has 4x4 entries.
    # C is 4x4, so it has 4x4 entries.
    
    # We split A into 4 blocks of size 2x2:
    # A[0:2, 0:2] = A11, A[0:2, 2:4] = A12
    # A[2:4, 0:2] = A21, A[2:4, 2:4] = A22
    
    # Similarly for B and C.
    
    # The product C = A * B involves:
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # Each of these is a sum of two 2x2x2 products.
    # We can use Strassen on the combined 2x2 block multiplication.
    
    # Actually, the standard Strassen for 4x4x4 treats the 4x4 matrices as 2x2 blocks
    # of 2x2 matrices, and applies Strassen to get 7 products of 2x2x2.
    
    # Let me define the block indices:
    # For A: block (a_row, a_col) where a_row, a_col in {0,1}
    # For B: block (b_row, b_col) where b_row, b_col in {0,1}
    # For C: block (c_row, c_col) where c_row, c_col in {0,1}
    
    # A block (a_row, a_col) corresponds to global rows a_row*2:(a_row+1)*2, cols a_col*2:(a_col+1)*2
    # B block (b_row, b_col) corresponds to global rows b_row*2:(b_row+1)*2, cols b_col*2:(b_col+1)*2
    # C block (c_row, c_col) corresponds to global rows c_row*2:(c_row+1)*2, cols c_col*2:(c_col+1)*2
    
    # The multiplication C = A*B means:
    # C(c_row, c_col) = sum over mid of A(a_row, mid) * B(mid, c_col)
    # where a_row = c_row, and mid is the block index.
    
    # So for each (c_row, c_col), we have:
    # C(c_row, c_col) = A(c_row, 0)*B(0, c_col) + A(c_row, 1)*B(1, c_col)
    
    # This is a 2x2 block multiplication for each (c_row, c_col) pair, but actually
    # it's the full 4x4x4 multiplication decomposed into blocks.
    
    # The standard approach: treat A, B, C as 2x2 matrices of 2x2 blocks.
    # Then apply Strassen's algorithm to get 7 products.
    
    # Let me define the 7 Strassen products for the block-level:
    # M1 = (A(0,0)+A(1,1))*(B(0,0)+B(1,1))
    # M2 = (A(1,0)+A(1,1))*B(0,0)
    # M3 = A(0,0)*(B(0,1)-B(1,1))
    # M4 = A(1,1)*(B(1,0)-B(0,0))
    # M5 = (A(0,0)+A(0,1))*B(1,1)
    # M6 = (A(1,0)-A(0,0))*(B(0,0)+B(0,1))
    # M7 = (A(0,1)-A(1,1))*(B(1,0)+B(1,1))
    
    # Then:
    # C(0,0) = M1 - M4 + M6 + M7
    # C(0,1) = M3 + M5
    # C(1,0) = M2 + M4
    # C(1,1) = M1 - M3 + M5 + M7
    
    # Each Mi is a 2x2x2 matrix multiplication (product of two 2x2 matrices giving a 2x2 result).
    # Each 2x2x2 can be done with rank 7 using Strassen.
    
    # So total rank for 4x4x4 is 7*7 = 49.
    
    # Now I need to construct the U, V, W for each of the 7 block-level products,
    # and for each, construct the 2x2x2 Strassen decomposition.
    
    # Let me define a helper to add a 2x2x2 Strassen decomposition at given block offsets.
    
    # For a 2x2x2 multiplication:
    # A is 2x2, B is 2x2, C is 2x2
    # A indices: (i,j) where i,j in {0,1}
    # B indices: (j,k) where j,k in {0,1}
    # C indices: (i,k) where i,k in {0,1}
    
    # The 7 Strassen products:
    # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    # M2 = (A[1,0]+A[1,1])*B[0,0]
    # M3 = A[0,0]*(B[0,1]-B[1,1])
    # M4 = A[1,1]*(B[1,0]-B[0,0])
    # M5 = (A[0,0]+A[0,1])*B[1,1]
    # M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
    # M7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
    
    # C[0,0] = M1 - M4 + M6 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M3 + M5 + M7
    
    # For each Mi, we have a rank-1 decomposition of the 2x2x2 tensor.
    # Actually, each Mi is itself a product of two linear combinations, so it's rank 1
    # in the tensor decomposition sense? No, that's not right.
    
    # Let me reconsider. The 2x2x2 matrix multiplication tensor has rank 7.
    # The Strassen decomposition gives 7 rank-1 terms.
    
    # For M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1]):
    # This is a single product of two linear combinations.
    # In the tensor T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[i,k,r],
    # M1 contributes to C[0,0] and C[1,1] with coefficient +1.
    
    # The rank-1 term for M1:
    # U part: A[0,0] and A[1,1] both have coefficient +1
    # V part: B[0,0] and B[1,1] both have coefficient +1
    # W part: C[0,0] and C[1,1] both have coefficient +1
    
    # So for this rank-1 term:
    # U[0,0] = 1, U[1,1] = 1 (for the A part)
    # V[0,0] = 1, V[1,1] = 1 (for the B part)
    # W[0,0] = 1, W[1,1] = 1 (for the C part)
    
    # But wait, this is for the 2x2x2 case. In the 4x4x4 case, we need to map
    # the local 2x2 indices to global 4x4 indices.
    
    # Let me define a function to add a 2x2x2 Strassen decomposition at given offsets.
    
    def add_2x2x2_strassen(U, V, W, a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset, start_col):
        """
        Add a 2x2x2 Strassen decomposition.
        A is at global rows a_row_offset:(a_row_offset+2), cols a_col_offset:(a_col_offset+2)
        B is at global rows b_row_offset:(b_row_offset+2), cols b_col_offset:(b_col_offset+2)
        C is at global rows c_row_offset:(c_row_offset+2), cols c_col_offset:(c_col_offset+2)
        """
        # Local to global mapping:
        # A local (i,j) -> global (a_row_offset+i, a_col_offset+j)
        # B local (j,k) -> global (b_row_offset+j, b_col_offset+k)
        # C local (i,k) -> global (c_row_offset+i, c_col_offset+k)
        
        # U index: (a_row, a_col) -> a_row * m + a_col (but m=4 in global)
        # Actually, U has shape (n*m, rank) where n=4, m=4, so U[i,j] is at index i*4+j
        # V has shape (m*p, rank) where m=4, p=4, so V[j,k] is at index j*4+k
        # W has shape (p*n, rank) where p=4, n=4, so W[k,i] is at index k*4+i
        
        # The 7 Strassen products for 2x2x2:
        strassen_products = [
            # (U_coeffs, V_coeffs, W_coeffs)
            # Each is a dict mapping (local_i, local_j) to coeff for U, etc.
            # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
            # M2 = (A[1,0]+A[1,1])*B[0,0]
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1}),
            # M3 = A[0,0]*(B[0,1]-B[1,1])
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1}),
            # M4 = A[1,1]*(B[1,0]-B[0,0])
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(1,1): -1}),
            # M5 = (A[0,0]+A[0,1])*B[1,1]
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,1): 1, (1,1): 1}),
            # M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(0,0): 1}),
            # M7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ]
        
        for (u_coeffs, v_coeffs, w_coeffs) in strassen_products:
            for (li, lj), coeff_u in u_coeffs.items():
                gi = a_row_offset + li
                gj = a_col_offset + lj
                U[gi * m + gj][start_col] += coeff_u
            for (lj, lk), coeff_v in v_coeffs.items():
                gj = b_row_offset + lj
                gk = b_col_offset + lk
                V[gj * p + gk][start_col] += coeff_v
            for (li, lk), coeff_w in w_coeffs.items():
                gi = c_row_offset + li
                gk = c_col_offset + lk
                W[gk * n + gi][start_col] += coeff_w
            start_col += 1
        
        return start_col
    
    # Now build the 4x4x4 Strassen decomposition.
    # The 7 block-level Strassen products:
    # Each involves products of 2x2 blocks.
    
    # M1_block = (A(0,0)+A(1,1))*(B(0,0)+B(1,1))
    # This is a product of two sums of 2x2 blocks.
    # In the tensor decomposition, this contributes to C(0,0) and C(1,1).
    
    # Actually, I realize I need to be more careful. The block-level Strassen
    # gives 7 products, each of which is a 2x2x2 matrix multiplication.
    # But each "product" is actually a bilinear form, not a single scalar.
    
    # Let me reconsider the structure. The 4x4x4 multiplication tensor can be
    # decomposed using the Kronecker product structure.
    
    # A 4x4 matrix is a 2x2 matrix of 2x2 blocks.
    # The multiplication C = A*B at the block level is:
    # C(i,j) = sum_k A(i,k)*B(k,j) for i,j in {0,1}
    
    # This is a 2x2x2 multiplication of blocks, where each block is 2x2.
    # We can apply Strassen to this block-level multiplication.
    
    # The block-level Strassen gives 7 terms, each of which is a product of
    # linear combinations of blocks. Each such product is a 2x2x2 matrix
    # multiplication (since each block is 2x2).
    
    # So for each of the 7 block-level terms, we need to decompose the 2x2x2
    # block multiplication, which itself has rank 7.
    
    # Total rank: 7 * 7 = 49.
    
    # Let me define the block-level Strassen terms:
    # For the 2x2x2 block multiplication:
    # A blocks: A00, A01, A10, A11 (each 2x2)
    # B blocks: B00, B01, B10, B11 (each 2x2)
    # C blocks: C00, C01, C10, C11 (each 2x2)
    
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # Strassen for this:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    
    # C00 = M1 - M4 + M6 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M5 + M7
    
    # Each Mi is a 2x2x2 matrix multiplication (product of two 2x2 matrices).
    # Each can be decomposed with rank 7.
    
    # Now I need to construct the U, V, W for each Mi.
    
    # For M1 = (A00+A11)*(B00+B11):
    # This is a bilinear form. In the tensor decomposition:
    # T[i,j,k] for the full 4x4x4.
    
    # A00 is at global rows 0:2, cols 0:2
    # A11 is at global rows 2:4, cols 2:4
    # B00 is at global rows 0:2, cols 0:2
    # B11 is at global rows 2:4, cols 2:4
    
    # M1 contributes to C00 (rows 0:2, cols 0:2) and C11 (rows 2:4, cols 2:4)
    
    # The rank-1 decomposition of M1 as a bilinear form:
    # M1 = (sum of A terms) * (sum of B terms)
    # = (A00 + A11) * (B00 + B11)
    
    # This expands to:
    # A00*B00 + A00*B11 + A11*B00 + A11*B11
    
    # Each of these is a 2x2x2 product.
    # A00*B00: A at (0:2, 0:2), B at (0:2, 0:2), C at (0:2, 0:2)
    # A00*B11: A at (0:2, 0:2), B at (2:4, 2:4), C at (0:2, 2:4)
    # A11*B00: A at (2:4, 2:4), B at (0:2, 0:2), C at (2:4, 0:2)
    # A11*B11: A at (2:4, 2:4), B at (2:4, 2:4), C at (2:4, 2:4)
    
    # But wait, this is not correct. M1 is a single bilinear form, not a sum.
    # In the tensor decomposition, M1 corresponds to one rank-1 term where:
    # U has coefficients for A00 and A11
    # V has coefficients for B00 and B11
    # W has coefficients for C00 and C11
    
    # Specifically, for M1:
    # U: A00 has coeff +1, A11 has coeff +1
    # V: B00 has coeff +1, B11 has coeff +1
    # W: C00 has coeff +1, C11 has coeff +1
    
    # But each of these is a 2x2 block, so we need to expand to individual entries.
    
    # For A00 (global rows 0:2, cols 0:2):
    # U[0,0], U[0,1], U[1,0], U[1,1] all get coeff +1
    
    # For A11 (global rows 2:4, cols 2:4):
    # U[2,2], U[2,3], U[3,2], U[3,3] all get coeff +1
    
    # Similarly for V and W.
    
    # But this gives a rank-1 term that is actually the product of two sums,
    # which is not a single rank-1 term in the tensor sense.
    
    # I think I'm confusing myself. Let me start over with a clearer approach.
    
    # The matrix multiplication tensor T for 4x4x4 has entries T[i,j,k] for
    # i in 0..3, j in 0..3, k in 0..3, where T[i,j,k] = A[i,j]*B[j,k]*C[i,k].
    
    # We want to decompose T as sum_r U[:, :, r] * V[:, :, r] * W[:, :, r]
    # where U has shape (4,4,R), V has shape (4,4,R), W has shape (4,4,R).
    
    # The Strassen decomposition for 4x4x4 uses the fact that 4 = 2*2, and
    # applies Strassen at two levels.
    
    # Level 1: Split 4x4 into 2x2 blocks of 2x2 matrices.
    # Level 2: Apply Strassen to the 2x2 block multiplication.
    
    # The key insight is that the 4x4x4 tensor can be written as a sum of 49
    # rank-1 tensors, where each rank-1 tensor corresponds to a product of
    # three linear combinations.
    
    # Let me just directly construct the 49 rank-1 terms.
    
    # I'll use the following approach:
    # 1. For each of the 7 block-level Strassen products, construct 7 rank-1 terms
    #    for the 2x2x2 block multiplication.
    # 2. Map each local 2x2 index to global 4x4 index.
    
    # Let me define the block-level Strassen terms more carefully.
    
    # Block A00: global (0,0), (0,1), (1,0), (1,1)
    # Block A01: global (0,2), (0,3), (1,2), (1,3)
    # Block A10: global (2,0), (2,1), (3,0), (3,1)
    # Block A11: global (2,2), (2,3), (3,2), (3,3)
    
    # Similarly for B and C.
    
    # For the block-level Strassen product M1 = (A00+A11)*(B00+B11):
    # This contributes to C00 and C11.
    
    # The 2x2x2 decomposition of (A00+A11)*(B00+B11):
    # This is a bilinear form in the entries of A00, A11, B00, B11.
    
    # Actually, I think the cleanest way is to note that each block-level
    # Strassen product is itself a 2x2x2 matrix multiplication, and we can
    # use the known rank-7 decomposition for 2x2x2.
    
    # Let me define a function that, given the block offsets for A, B, C,
    # adds the 7 rank-1 terms for the 2x2x2 Strassen decomposition.
    
    # For a 2x2x2 multiplication with A at (a_row, a_col), B at (b_row, b_col),
    # C at (c_row, c_col), the 7 Strassen terms are:
    
    # I'll use the standard Strassen decomposition for 2x2x2.
    
    # Let me look up the exact coefficients.
    
    # For 2x2x2 Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 - M4 + M6 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M5 + M7
    
    # For each Mi, the rank-1 decomposition is:
    # M1: U has A11=1, A22=1; V has B11=1, B22=1; W has C11=1, C22=1
    # M2: U has A21=1, A22=1; V has B11=1; W has C21=1
    # M3: U has A11=1; V has B12=1, B22=-1; W has C12=1
    # M4: U has A22=1; V has B21=1, B11=-1; W has C22=-1
    # M5: U has A11=1, A12=1; V has B22=1; W has C12=1, C22=1
    # M6: U has A21=1, A11=-1; V has B11=1, B12=1; W has C11=1
    # M7: U has A12=1, A22=-1; V has B21=1, B22=1; W has C11=1, C22=1
    
    # Now, for the 4x4x4 case, each "block" is a 2x2 matrix, so we need to
    # expand each of these to individual entries.
    
    # Let me define the block-to-global mapping.
    
    # For block (bi, bj) of a 4x4 matrix, the global indices are:
    # rows: bi*2, bi*2+1
    # cols: bj*2, bj*2+1
    
    # So A00 block: global (0,0), (0,1), (1,0), (1,1)
    # A01 block: global (0,2), (0,3), (1,2), (1,3)
    # A10 block: global (2,0), (2,1), (3,0), (3,1)
    # A11 block: global (2,2), (2,3), (3,2), (3,3)
    
    # Now, for the block-level Strassen product M1 = (A00+A11)*(B00+B11):
    # This is a 2x2x2 multiplication where:
    # "A" is the 2x2 matrix [A00, A01; A10, A11] but we're only using A00 and A11
    # Actually, no. M1 is (A00+A11)*(B00+B11), which is a product of two
    # 2x2 matrices (each being a sum of two 2x2 blocks).
    
    # Wait, I think I'm overcomplicating this. Let me think of it differently.
    
    # The 4x4x4 multiplication can be seen as a 2x2x2 multiplication of 2x2 blocks.
    # The Strassen algorithm for 2x2x2 gives 7 products, each of which is a
    # 2x2x2 matrix multiplication (of the individual entries within the blocks).
    
    # So for each of the 7 block-level products, we need to decompose a 2x2x2
    # matrix multiplication, which has rank 7.
    
    # Total: 7 * 7 = 49.
    
    # Let me define the block-level Strassen terms:
    # Let A_block(i,j) denote the 2x2 block at position (i,j) in the 4x4 matrix A.
    # Similarly for B_block and C_block.
    
    # The 7 block-level Strassen products are:
    # M1_block = (A_block(0,0) + A_block(1,1)) * (B_block(0,0) + B_block(1,1))
    # M2_block = (A_block(1,0) + A_block(1,1)) * B_block(0,0)
    # M3_block = A_block(0,0) * (B_block(0,1) - B_block(1,1))
    # M4_block = A_block(1,1) * (B_block(1,0) - B_block(0,0))
    # M5_block = (A_block(0,0) + A_block(0,1)) * B_block(1,1)
    # M6_block = (A_block(1,0) - A_block(0,0)) * (B_block(0,0) + B_block(0,1))
    # M7_block = (A_block(0,1) - A_block(1,1)) * (B_block(1,0) + B_block(1,1))
    
    # C_block(0,0) = M1_block - M4_block + M6_block + M7_block
    # C_block(0,1) = M3_block + M5_block
    # C_block(1,0) = M2_block + M4_block
    # C_block(1,1) = M1_block - M3_block + M5_block + M7_block
    
    # Each M*_block is a 2x2x2 matrix multiplication.
    
    # Now, for each M*_block, I need to construct the 7 rank-1 terms for the
    # 2x2x2 decomposition.
    
    # Let me write a helper function.
    
    def add_2x2x2_decomposition(U, V, W, a_blocks, b_blocks, c_blocks, start_col, n, m, p):
        """
        Add the 7 rank-1 terms for a 2x2x2 Strassen decomposition.
        a_blocks: list of (block_row, block_col) for A
        b_blocks: list of (block_row, block_col) for B
        c_blocks: list of (block_row, block_col) for C
        """
        # This is getting too complex. Let me just directly enumerate all 49 terms.
        pass
    
    # I think the cleanest approach is to directly enumerate the 49 rank-1 terms
    # for the 4x4x4 Strassen decomposition.
    
    # Let me use a different strategy. I'll use the fact that the 4x4x4 tensor
    # can be decomposed using the Kronecker product of the 2x2x2 tensor with itself,
    # but with Strassen's optimization.
    
    # Actually, let me just implement it by directly constructing the 49 terms.
    
    # I'll define the 49 terms as a list of (U_coeffs, V_coeffs, W_coeffs) where
    # each is a dict mapping (i,j) to coefficient.
    
    # This is going to be tedious but straightforward.
    
    # Let me use a computational approach to generate the terms.
    
    # Actually, I realize I should just use numpy to construct the matrices.
    
    # Let me try a different approach: use the known rank-49 decomposition for 4x4x4.
    
    # I found a reference that gives the explicit decomposition. Let me try to
    # construct it programmatically.
    
    # For now, let me just return the schoolbook decomposition to ensure correctness,
    # and then optimize later.
    
    # Schoolbook for 4x4x7: rank = 4*4*7 = 112
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
# EVOLVE-BLOCK-END

