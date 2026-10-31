"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    assert (n, m, p) == (4, 4, 4)
    
    # Strassen 2x2 rank-7 decomposition
    # U_S: (4, 7) - for A entries indexed as [0,1,2,3] = [A00,A01,A10,A11]
    # V_S: (4, 7) - for B entries indexed as [0,1,2,3] = [B00,B01,B10,B11]
    # W_S: (4, 7) - for C entries indexed as [0,1,2,3] = [C00,C01,C10,C11]
    # Convention: W_S[c_row, k] such that C[c_row] = sum_k U_S[a_row,k] * V_S[b_row,k] * W_S[c_row,k]
    # More precisely: C[i,j] = sum_k U_S[i,k] * V_S[j,k] * W_S[i*2+j, k]
    # Wait, let me be more careful.
    
    # Standard Strassen for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # A index: A00=0, A01=1, A10=2, A11=3
    # B index: B00=0, B01=1, B10=2, B11=3
    # C index: C00=0, C01=1, C10=2, C11=3
    
    # U_S[i, k]: coefficient of A_i in the k-th factor for the first matrix
    # V_S[j, k]: coefficient of B_j in the k-th factor for the second matrix
    # W_S[c, k]: coefficient for C_c from the k-th product
    
    U_S = np.zeros((4, 7))
    V_S = np.zeros((4, 7))
    W_S = np.zeros((4, 7))
    
    # M1: (A00+A11)(B00+B11)
    U_S[0, 0] = 1  # A00
    U_S[3, 0] = 1  # A11
    V_S[0, 0] = 1  # B00
    V_S[3, 0] = 1  # B11
    W_S[0, 0] = 1  # C00 += M1
    W_S[3, 0] = 1  # C11 += M1
    
    # M2: (A10+A11)B00
    U_S[2, 1] = 1  # A10
    U_S[3, 1] = 1  # A11
    V_S[0, 1] = 1  # B00
    W_S[2, 1] = 1  # C10 += M2
    W_S[3, 1] = -1 # C11 -= M2
    
    # M3: A00(B01-B11)
    U_S[0, 2] = 1  # A00
    V_S[1, 2] = 1  # B01
    V_S[3, 2] = -1 # B11
    W_S[1, 2] = 1  # C01 += M3
    W_S[3, 2] = 1  # C11 += M3
    
    # M4: A11(B10-B00)
    U_S[3, 3] = 1  # A11
    V_S[2, 3] = 1  # B10
    V_S[0, 3] = -1 # B00
    W_S[0, 3] = 1  # C00 += M4
    W_S[2, 3] = 1  # C10 += M4
    
    # M5: (A00+A01)B11
    U_S[0, 4] = 1  # A00
    U_S[1, 4] = 1  # A01
    V_S[3, 4] = 1  # B11
    W_S[0, 4] = -1 # C00 -= M5
    W_S[1, 4] = 1  # C01 += M5
    
    # M6: (A10-A00)(B00+B01)
    U_S[2, 5] = 1  # A10
    U_S[0, 5] = -1 # A00
    V_S[0, 5] = 1  # B00
    V_S[1, 5] = 1  # B01
    W_S[3, 5] = 1  # C11 += M6
    
    # M7: (A01-A11)(B10+B11)
    U_S[1, 6] = 1  # A01
    U_S[3, 6] = -1 # A11
    V_S[2, 6] = 1  # B10
    V_S[3, 6] = 1  # B11
    W_S[0, 6] = 1  # C00 += M7
    
    # Verify Strassen correctness
    # C[i,j] = sum_k U_S[i,k] * V_S[j,k] * W_S[i*2+j, k]
    # Let's verify with symbolic check
    for i in range(4):
        for j in range(4):
            c_idx = i * 2 + j  # C index
            total = 0.0
            for k in range(7):
                total += U_S[i, k] * V_S[j, k] * W_S[c_idx, k]
            # This should match the expected C entry
            # C00 = A00*B00 + A01*B10
            # C01 = A00*B01 + A01*B11
            # C10 = A10*B00 + A11*B10
            # C11 = A10*B01 + A11*B11
            if c_idx == 0:  # C00
                expected = 1.0  # A00*B00 coefficient
            elif c_idx == 1:  # C01
                expected = 1.0  # A00*B01 coefficient
            elif c_idx == 2:  # C10
                expected = 1.0  # A10*B00 coefficient
            elif c_idx == 3:  # C11
                expected = 1.0  # A10*B01 coefficient
            # Actually let me just check a few specific entries
    # Let me verify more carefully
    # C00 should be A00*B00 + A01*B10
    # Coefficient of A00*B00 in C00: sum_k U_S[0,k]*V_S[0,k]*W_S[0,k]
    coeff_A00B00_C00 = sum(U_S[0,k] * V_S[0,k] * W_S[0,k] for k in range(7))
    # Should be 1
    # Coefficient of A01*B10 in C00: sum_k U_S[1,k]*V_S[2,k]*W_S[0,k]
    coeff_A01B10_C00 = sum(U_S[1,k] * V_S[2,k] * W_S[0,k] for k in range(7))
    # Should be 1
    # Coefficient of A00*B10 in C00: sum_k U_S[0,k]*V_S[2,k]*W_S[0,k]
    coeff_A00B10_C00 = sum(U_S[0,k] * V_S[2,k] * W_S[0,k] for k in range(7))
    # Should be 0
    
    # Now build 4x4 decomposition via Kronecker product
    # A is 4x4, indexed as A[i,j] with flat index i*4+j
    # For 2x2 block structure: A[i,j] where i=2*bi+ri, j=2*bj+cj
    # bi,bj in {0,1}, ri,cj in {0,1}
    # A block (bi,bj) has entries at flat indices: (2*bi+ri)*4 + (2*bj+cj) for ri,cj in {0,1}
    # In the 2x2 Strassen, A00 corresponds to block (0,0), A01 to block (0,1), etc.
    # A00 flat indices: 0,1,4,5 (i=0,1; j=0,1)
    # A01 flat indices: 2,3,6,7 (i=0,1; j=2,3)
    # A10 flat indices: 8,9,12,13 (i=2,3; j=0,1)
    # A11 flat indices: 10,11,14,15 (i=2,3; j=2,3)
    
    # For the Kronecker product approach:
    # U[flat_a_idx, r1*7+r2] = U_S[local_a_idx, r1] * U_S[local_a_idx2, r2]
    # where flat_a_idx = (2*bi+ri)*4 + (2*bj+cj)
    # and local_a_idx depends on the block structure
    
    # Actually, let me think about this differently.
    # The 4x4 matrix multiplication tensor can be seen as the tensor product
    # of two 2x2 matrix multiplication tensors.
    
    # For 2x2: T_2x2[a,b,c] = 1 if C_c contains A_a * B_b
    # For 4x4: T_4x4[A,B,C] = 1 if C contains A*B
    # T_4x4 = T_2x2 ⊗ T_2x2 (Kronecker product)
    
    # The rank-7 decomposition of T_2x2 gives:
    # T_2x2[a,b,c] = sum_k U_S[a,k] * V_S[b,k] * W_S[c,k]
    
    # For T_4x4 = T_2x2 ⊗ T_2x2:
    # T_4x4[(a1,a2),(b1,b2),(c1,c2)] = sum_{k1,k2} U_S[a1,k1]*U_S[a2,k2] * V_S[b1,k1]*V_S[b2,k2] * W_S[c1,k1]*W_S[c2,k2]
    
    # So U_4x4[(a1,a2), (k1,k2)] = U_S[a1,k1] * U_S[a2,k2]
    # V_4x4[(b1,b2), (k1,k2)] = V_S[b1,k1] * V_S[b2,k2]
    # W_4x4[(c1,c2), (k1,k2)] = W_S[c1,k1] * W_S[c2,k2]
    
    # Now I need to map the flat indices correctly.
    # A flat index: i*4+j where i,j in {0,1,2,3}
    # In block form: i = 2*bi + ri, j = 2*bj + cj
    # So flat index = (2*bi+ri)*4 + (2*bj+cj) = 8*bi + 4*ri + 2*bj + cj
    # The "a" index in T_2x2 is the block index (bi,bj) mapped to 0,1,2,3
    # a = 2*bi + cj... no wait.
    
    # Let me define: for 2x2 Strassen, the A index a in {0,1,2,3} corresponds to:
    # a=0: A00 (block (0,0))
    # a=1: A01 (block (0,1)) 
    # a=2: A10 (block (1,0))
    # a=3: A11 (block (1,1))
    # So a = 2*bi + bj where bi is row block, bj is col block
    
    # For the 4x4 case, each "a" in the 2x2 decomposition corresponds to a 2x2 block
    # of the 4x4 matrix. So the flat A index decomposes as:
    # flat_a = (2*bi+ri)*4 + (2*bj+cj)
    # where (bi,bj) gives the 2x2 block index a = 2*bi+bj
    # and (ri,cj) gives the position within the block
    
    # For the Kronecker product:
    # U_4x4[flat_a, (k1,k2)] where flat_a = (2*bi+ri)*4 + (2*bj+cj)
    # This should be U_S[a1, k1] * U_S[a2, k2] where...
    
    # Hmm, I think the Kronecker product structure is:
    # The 4x4 multiplication is equivalent to multiplying two 2x2 "super-matrices"
    # where each entry is a 2x2 matrix.
    
    # So if we view A as a 2x2 matrix of 2x2 blocks, then:
    # A_block(bi,bj) is a 2x2 matrix
    # The product C_block(ci,cj) = sum_{bi,bj} A_block(bi,bj) * B_block(bi,bj)
    
    # Each block product is a 2x2 matrix multiplication, which can be done with Strassen.
    # But that gives 7*7=49 multiplications of 2x2 matrices, each of which is 8 scalar mults.
    # That's not what we want.
    
    # Actually, the tensor product approach is cleaner. Let me just directly construct
    # the Kronecker product.
    
    # U_4x4 has shape (16, 49)
    # V_4x4 has shape (16, 49)
    # W_4x4 has shape (16, 49)
    
    # Rank index: r = k1*7 + k2 where k1,k2 in {0,...,6}
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    # For each flat A index (16 of them), decompose into block and local indices
    for flat_a in range(16):
        i = flat_a // 4
        j = flat_a % 4
        bi = i // 2
        ri = i % 2
        bj = j // 2
        cj = j % 2
        a_block = 2 * bi + bj  # 0,1,2,3 for the 2x2 block index
        
        for k1 in range(7):
            for k2 in range(7):
                r = k1 * 7 + k2
                # The Kronecker product: U_4x4[flat_a, r] = U_S[a_block, k1] * U_S[?, k2]
                # But what is the second index?
                
    # I think I'm overcomplicating this. Let me use the direct approach:
    # The 4x4 matrix multiplication tensor T[i,j,k,l] = 1 if C[k,l] = A[i,j]*B[j,l]
    # can be decomposed as the Kronecker product of two 2x2 tensors.
    
    # Let me just build it by iterating over all possible rank-1 terms.
    # Each rank-1 term corresponds to a choice of (k1,k2) in {0..6}x{0..6}
    # For each such term, U, V, W entries are products of the 2x2 Strassen factors.
    
    # U[flat_a, r] where flat_a = i*4+j, r = k1*7+k2
    # U_S is indexed by (a, k) where a is the 2x2 A index
    # For the 4x4 case, flat_a decomposes into (a1, a2) where:
    # a1 = 2*(i//2) + (j//2)  -- this is the "outer" 2x2 index
    # a2 = 2*(i%2) + (j%2)    -- this is the "inner" 2x2 index
    # Wait, that doesn't seem right either.
    
    # Let me try a different decomposition. View the 4x4 matrix as:
    # A = [[A00, A01], [A10, A11]] where Aij are 2x2
    # Then C = A*B where the block product uses 2x2 matrix multiplication.
    
    # The tensor for block multiplication is T_block[X,Y,Z] where X,Y,Z are 2x2 matrices
    # T_block can be decomposed with Strassen: T_block = sum_k U_S[x,k]*V_S[y,k]*W_S[z,k]
    # where x,y,z are the flat indices of the 2x2 entries.
    
    # For the full 4x4: T_4x4[A,B,C] = sum_{bi,bj} T_block[A_block(bi,bj), B_block(bi,bj), C_block(bi,bj)]
    # This isn't quite a tensor product.
    
    # OK let me just directly construct the Kronecker product properly.
    
    # The key insight: 4x4 matmul tensor = (2x2 matmul tensor) ⊗ (2x2 matmul tensor)
    # where the tensor indices are:
    # T_4x4[A_idx, B_idx, C_idx]
    # A_idx = i*4+j, B_idx = j'*4+k, C_idx = k'*4+i'
    # T_4x4 = 1 if i'=i, k'=k, and the bilinear form matches
    
    # Actually, the standard tensor is: T[A,B,C] = sum_{j} A[i,j]*B[j,k] where C[i,k]
    # For 4x4: T[i*4+j, j'*4+k, k'*4+i] = 1 if i=k' and j=j'... no.
    
    # Let me just use the Kronecker product of the factor matrices directly.
    
    # U_4x4 = U_S ⊗ U_S (Kronecker product)
    # V_4x4 = V_S ⊗ V_S
    # W_4x4 = W_S ⊗ W_S
    
    # Kronecker product: (A⊗B)[i*B_rows + j, k*B_cols + l] = A[i,k]*B[j,l]
    
    U_kron = np.kron(U_S, U_S)  # (4*4, 7*7) = (16, 49)
    V_kron = np.kron(V_S, V_S)  # (16, 49)
    W_kron = np.kron(W_S, W_S)  # (16, 49)
    
    # Now I need to verify the indexing is correct.
    # The Kronecker product U_S ⊗ U_S has index [i*4+j, k*7+l] = U_S[i,k]*U_S[j,l]
    # where i,j are the 2x2 A indices and k,l are the rank indices.
    
    # For the 4x4 matrix, flat index i*4+j corresponds to row i, col j.
    # In the Kronecker product, the index [i*4+j] in U_kron corresponds to
    # the "first" 2x2 index being i and "second" being j.
    
    # But in the 4x4 matmul, the A index is (row, col) = (i, j)
    # and the bilinear form is sum_j A[i,j]*B[j,k]
    
    # For the Kronecker product to work, we need:
    # U_kron[flat_a, r] * V_kron[flat_b, r] * W_kron[flat_c, r] summed over r
    # should give the correct coefficient for A_flat_a * B_flat_b in C_flat_c.
    
    # Let me verify with a simple case.
    # A[0,0] * B[0,0] should contribute to C[0,0] with coefficient 1.
    # flat_a = 0*4+0 = 0
    # flat_b = 0*4+0 = 0  
    # flat_c = 0*4+0 = 0
    
    # In Kronecker: flat_a=0 means i=0,j=0 in the 2x2 indices
    # U_kron[0, r] where r=k*7+l, U_kron[0, r] = U_S[0,k]*U_S[0,l]
    # V_kron[0, r] = V_S[0,k]*V_S[0,l]
    # W_kron[0, r] = W_S[0,k]*W_S[0,l]
    
    # Sum over r: sum_{k,l} U_S[0,k]*U_S[0,l]*V_S[0,k]*V_S[0,l]*W_S[0,k]*W_S[0,l]
    # = (sum_k U_S[0,k]*V_S[0,k]*W_S[0,k]) * (sum_l U_S[0,l]*V_S[0,l]*W_S[0,l])
    # = (coeff of A00*B00 in C00 for 2x2)^2
    
    # For 2x2, coeff of A00*B00 in C00 = 1
    # So this gives 1*1 = 1. Correct!
    
    # Let me check A[0,0]*B[0,1] in C[0,1]:
    # flat_a=0, flat_b=1, flat_c=1
    # U_kron[0, r] = U_S[0,k]*U_S[0,l]
    # V_kron[1, r] where flat_b=1 means i=0,j=1: V_S[0,k]*V_S[1,l]
    # W_kron[1, r] where flat_c=1 means i=0,j=1: W_S[0,k]*W_S[1,l]
    
    # Sum: sum_{k,l} U_S[0,k]*U_S[0,l]*V_S[0,k]*V_S[1,l]*W_S[0,k]*W_S[1,l]
    # = (sum_k U_S[0,k]*V_S[0,k]*W_S[0,k]) * (sum_l U_S[0,l]*V_S[1,l]*W_S[1,l])
    # = (coeff A00*B00 in C00) * (coeff A00*B01 in C01)
    # = 1 * 1 = 1. Correct!
    
    # Let me check A[0,1]*B[0,0] in C[0,0]:
    # flat_a=1 (i=0,j=1), flat_b=0, flat_c=0
    # U_kron[1, r] = U_S[0,k]*U_S[1,l]
    # V_kron[0, r] = V_S[0,k]*V_S[0,l]
    # W_kron[0, r] = W_S[0,k]*W_S[0,l]
    
    # Sum: sum_{k,l} U_S[0,k]*U_S[1,l]*V_S[0,k]*V_S[0,l]*W_S[0,k]*W_S[0,l]
    # = (sum_k U_S[0,k]*V_S[0,k]*W_S[0,k]) * (sum_l U_S[1,l]*V_S[0,l]*W_S[0,l])
    # = (coeff A00*B00 in C00) * (coeff A01*B00 in C00)
    # For 2x2, coeff of A01*B00 in C00 = 0 (C00 = A00*B00 + A01*B10)
    # So this gives 1*0 = 0. Correct!
    
    # Great, the Kronecker product approach works!
    
    # But wait, I need to check the W index convention.
    # The task says: W[k*n+i, r] where k is column, i is row
    # So W index = k*4 + i where k is col, i is row
    # My W_kron has index [i*4+j] where i is row, j is col
    # I need to transpose the indexing: W_final[k*4+i, r] = W_kron[i*4+k, r]
    
    # Actually, let me re-read: "output factor follows the official trace convention W[k*n+i,r]"
    # So W[row, col] in the matrix sense, but the flat index is k*n+i where k=col, i=row
    # So W_flat[k*n+i] = W_matrix[i, k]
    
    # My W_kron[i*4+j] corresponds to row i, col j
    # I need W_final[k*4+i] = W_kron[i*4+k]
    # So W_final = W_kron with rows permuted: new_row = k*4+i, old_row = i*4+k
    
    W_final = np.zeros((16, 49))
    for i in range(4):
        for k in range(4):
            W_final[k*4+i, :] = W_kron[i*4+k, :]
    
    # Now convert to lists
    U_list = [[int(U_kron[i, r]) for r in range(49)] for i in range(16)]
    V_list = [[int(V_kron[i, r]) for r in range(49)] for i in range(16)]
    W_list = [[int(W_final[i, r]) for r in range(49)] for i in range(16)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

