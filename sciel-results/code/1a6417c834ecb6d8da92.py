"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 7
    # Use Kronecker product structure:
    # Decompose 2x2x7 into (2x2x2) ⊗ (1x1x7)
    # The 2x2x2 part uses Strassen (rank 7)
    # The 1x1x7 part is trivial (rank 7, one term per column)
    # Total rank = 7 * 7 = 49
    
    rank_2x2x2 = 7
    rank_1x1x7 = p  # 7
    rank = rank_2x2x2 * rank_1x1x7  # 49
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 49
    V = [[0] * rank for _ in range(m * p)]  # 14 x 49
    W = [[0] * rank for _ in range(p * n)]  # 14 x 49
    
    # Strassen decomposition for 2x2x2
    # U2[i*2+j, c], V2[j*2+k, c], W2[k*2+i, c]
    # Strassen terms (u_coeffs, v_coeffs, w_coeffs) where each is length 4
    strassen_terms = [
        # M1 = A11 * (B12 + B22)
        ([1, 0, 0, 0], [0, 1, 0, 1], [0, 1, 0, 0]),
        # M2 = (A21 + A22) * B11
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        # M3 = (A11 + A12) * B22
        ([1, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]),
        # M4 = A22 * (B11 + B21)
        ([0, 0, 0, 1], [1, 0, 1, 0], [0, 0, 0, 1]),
        # M5 = (A11 + A22) * (B11 + B22)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M6 = (A11 - A21) * (B12 + B22)
        ([1, 0, -1, 0], [0, 1, 0, 1], [0, 0, 0, 0]),
        # M7 = (A12 - A22) * (B21 + B22)
        ([0, 1, 0, -1], [0, 0, 1, 1], [0, 1, 0, 0]),
    ]
    
    # For the Kronecker product structure:
    # The 2x2x2 part handles the 2x2 matrix multiplication structure
    # The 1x1x7 part handles the 7 columns
    # 
    # For each Strassen term t (0..6) and each column l (0..6):
    #   U[i*2+j, t*7+l] = u_coeffs[i*2+j]
    #   V[j*7+l, t*7+l] = v_coeffs[j*2]  (for the 2x2x2 part, j is 0 or 1, k is 0 or 1)
    #   W[l*2+i, t*7+l] = w_coeffs[k*2+i]
    
    # Wait, I need to be more careful with the Kronecker structure.
    # 
    # The tensor for 2x2x7 is: sum_{i,j,l} A[i,j] * B[j,l] * C[i,l]
    # 
    # Using Kronecker product: think of this as (2x2x2) ⊗ (1x1x7)
    # where the 1x1x7 part just indexes the columns.
    # 
    # For each Strassen term t and each column l:
    #   The A-part contribution: u_coeffs_t[i*2+j]
    #   The B-part contribution: v_coeffs_t[j*2] for the 2x2x2 part, but B is 2x7
    #   The C-part contribution: w_coeffs_t[k*2+i] for the 2x2x2 part, but C is 2x7
    # 
    # Actually, let me think of it differently. The Kronecker product structure
    # means we factor the tensor as a product of two smaller tensors.
    # 
    # Tensor T for 2x2x7: T[(i,j), (j,l), (l,i)] = 1
    # 
    # We can write this as: T = T_2x2x2 ⊗ T_1x1x7
    # where T_2x2x2[(i,j), (j,k), (k,i)] = 1 for i,j,k in {0,1}
    # and T_1x1x7[(a,b), (b,c), (c,a)] = 1 for a,b,c in {0} (trivial)
    # 
    # But this doesn't quite work because the dimensions don't match.
    # 
    # Let me try a different approach. The key insight is that we can use the
    # Strassen decomposition for the 2x2 part and replicate it for each column.
    # 
    # For each Strassen term t (0..6):
    #   For each column l (0..6):
    #     Create a term that handles the contribution of column l using Strassen term t
    # 
    # But this would give 7*7 = 49 terms, which is the Kronecker product rank.
    # 
    # However, I realize that this approach may not be correct because the Strassen
    # decomposition is for 2x2x2, not for 2x2x7. Let me reconsider.
    # 
    # Actually, the correct approach is:
    # The 2x2x7 tensor can be viewed as 7 copies of the 2x2x1 tensor (one for each column).
    # The 2x2x1 tensor has rank 2 (schoolbook).
    # So naively, 7*2 = 14 terms.
    # 
    # But we can do better by sharing the A-matrix structure across columns.
    # 
    # Let me just implement the schoolbook 28-rank decomposition, which is known to be correct.
    
    # Schoolbook decomposition for 2x2x7
    for i in range(n):
        for j in range(m):
            for l in range(p):
                col = i * m * p + j * p + l
                U[i * m + j][col] = 1
                V[j * p + l][col] = 1
                W[l * n + i][col] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
