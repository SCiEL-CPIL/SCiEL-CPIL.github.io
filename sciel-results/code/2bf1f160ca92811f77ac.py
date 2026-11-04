import time
import sys

# EVOLVE-BLOCK-START

def construct():
    n, m, p = 4, 4, 7
    # Part 1: 4x4x4 using Strassen (Rank 56)
    # Part 2: 4x4x3 using Trivial (Rank 48)
    
    rank = 56 + 48
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    t = 0
    
    # Strassen terms for 2x2 blocks
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Each term is a product of two linear combinations of blocks.
    # We decompose the resulting 2x2 matrix product using trivial rank 8.
    
    # The trivial decomposition of a 2x2 matrix product C = AB is:
    # Sum over 8 scalar multiplications:
    # (i,j) in A, (j,k) in B -> contributes to C[i,k]
    # Specifically:
    # A[0,0]B[0,0] -> C[0,0]
    # A[0,0]B[0,1] -> C[0,1]
    # A[0,1]B[1,0] -> C[0,0]
    # A[0,1]B[1,1] -> C[0,1]
    # A[1,0]B[0,0] -> C[1,0]
    # A[1,0]B[0,1] -> C[1,1]
    # A[1,1]B[1,0] -> C[1,0]
    # A[1,1]B[1,1] -> C[1,1]
    
    # For a Strassen term M_t = L_A * L_B, where L_A and L_B are 2x2 matrices of linear combos of blocks.
    # We expand L_A and L_B into global A and B indices.
    # L_A[i,j] is a linear combo of A[2I+i, 2J+j].
    # L_B[j,k] is a linear combo of B[2J+j, 2K+k].
    
    # We add 8 rank-1 tensors for each Strassen term.
    # For each (i_block, j_block, k_block) in 0..1:
    # The term is L_A[i_block, j_block] * L_B[j_block, k_block] contributing to C[i_block, k_block].
    # Wait, the trivial decomposition is sum over j_block?
    # No, the trivial decomposition of the 2x2 product uses 8 terms, one for each (i, j, k) combination?
    # No, the 8 terms are the 8 scalar multiplications in the 2x2 product.
    # The scalar multiplications are:
    # (0,0,0), (0,0,1), (0,1,0), (0,1,1), (1,0,0), (1,0,1), (1,1,0), (1,1,1)
    # Where indices are (i, j, k) such that A[i,j] * B[j,k] contributes to C[i,k].
    # Yes, these are the 8 terms.
    
    strassen_terms = [
        { 'A': [(0,0,1), (1,1,1)], 'B': [(0,0,1), (1,1,1)] },
        { 'A': [(1,0,1), (1,1,1)], 'B': [(0,0,1)] },
        { 'A': [(0,0,1)], 'B': [(0,1,1), (1,1,-1)] },
        { 'A': [(1,1,1)], 'B': [(1,0,1), (0,0,-1)] },
        { 'A': [(0,0,1), (0,1,1)], 'B': [(1,1,1)] },
        { 'A': [(1,0,1), (0,0,-1)], 'B': [(0,0,1), (0,1,1)] },
        { 'A': [(0,1,1), (1,1,-1)], 'B': [(1,0,1), (1,1,1)] }
    ]
    
    for term in strassen_terms:
        A_comb = term['A']
        B_comb = term['B']
        
        # Build L_A and L_B as 2x2 matrices of coefficients
        # L_A[I][J] is the coefficient for block A[I][J]
        L_A = [[0]*2 for _ in range(2)]
        for I, J, c in A_comb:
            L_A[I][J] += c
            
        L_B = [[0]*2 for _ in range(2)]
        for J, K, c in B_comb:
            L_B[J][K] += c
            
        # Iterate over the 8 trivial terms for the 2x2 product
        for i_block in range(2):
            for j_block in range(2):
                for k_block in range(2):
                    # The term is L_A[i_block, j_block] * L_B[j_block, k_block]
                    # This contributes to C[i_block, k_block]
                    
                    # Check if coefficients are zero
                    if L_A[i_block][j_block] == 0 or L_B[j_block][k_block] == 0:
                        continue
                        
                    # We need to add a rank-1 tensor.
                    # U vector: non-zero at A elements in block (i_block, j_block)
                    # V vector: non-zero at B elements in block (j_block, k_block)
                    # W vector: non-zero at C elements in block (i_block, k_block)
                    
                    # But L_A[i_block, j_block] is a linear combo.
                    # In the trivial decomposition, we have separate terms for each scalar.
                    # Here, the "scalar" is the whole block?
                    # No, the block is 2x2.
                    # The linear combo L_A[i_block, j_block] applies to the whole block.
                    # So we add ONE rank-1 tensor for the whole block product?
                    # No, a rank-1 tensor corresponds to ONE scalar multiplication.
                    # The block product is 4 scalar multiplications.
                    # So we need 4 rank-1 tensors for each (i_block, j_block, k_block)?
                    # That would be 7 * 8 * 4 = 224. Too high.
                    
                    # Wait, the trivial decomposition of 2x2x2 (scalar) has rank 8.
                    # The 8 terms are the 8 scalar multiplications.
                    # For the block case, the "scalar" is a 2x2 matrix.
                    # The rank of the 2x2x2 tensor (where entries are 2x2 matrices) is 8?
                    # No, the rank is the sum of the ranks of the scalar decompositions.
                    # If we treat the blocks as scalars, the rank is 8.
                    # But each "scalar" is a 2x2 matrix, which has rank 4 (trivial) or 7 (Strassen).
                    # So total rank is 8 * 4 = 32?
                    # Or 8 * 7 = 56?
                    
                    # Let's use trivial for the inner 2x2.
                    # So for each of the 8 block-terms, we add 4 rank-1 tensors (one for each element in the 2x2 block).
                    # Total 7 * 8 * 4 = 224.
                    # This is worse than 112.
                    
                    # I am stuck again. The rank of [4,4,4] is 49.
                    # I will just use the known 49 rank decomposition.
                    pass

    # Fallback to Trivial 112
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    t = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][t] = 1
                V[v_idx][t] = 1
                W[w_idx][t] = 1
                t += 1
                
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
