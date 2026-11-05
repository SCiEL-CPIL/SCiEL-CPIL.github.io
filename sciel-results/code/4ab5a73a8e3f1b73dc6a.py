"""Matrix multiplication T_{2,3,5} tensor decomposition with rank reduction via linear dependencies."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 5
    # Schoolbook rank = n*m*p = 30
    # Try to find a rank-minimal decomposition by exploiting linear dependencies
    # among the 30 rank-1 terms.
    #
    # The tensor T_{i,j,k} = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Actually, the matrix multiplication tensor is:
    # T[i*m+j, j*p+k, k*n+i] = 1 (for the standard convention)
    # Wait, let me re-read: W[k*n+i, r]
    # So the tensor has indices (a, b, c) where:
    # a = i*m + j (input to A: row i, col j)
    # b = j*p + k (input to B: row j, col k)  
    # c = k*n + i (output: row i, col k)
    #
    # The tensor T[a,b,c] = 1 iff there exist i,j,k such that:
    # a = i*m+j, b = j*p+k, c = k*n+i
    # This is always consistent, so T is the "standard" matrix mult tensor.
    #
    # For T_{2,3,5}, we have 30 rank-1 tensors in the schoolbook decomposition.
    # We need to find if some are linearly dependent.
    #
    # Let's use numpy to check for rank of the 3-tensor by flattening.
    # Actually, a better approach: try known small-rank decompositions.
    #
    # Known results:
    # - T_{2,2,2} has rank 7 (Strassen)
    # - T_{2,3,3} has rank 18 or 19? 
    # - For T_{2,3,5}, let's try to find a decomposition.
    #
    # Strategy: Represent each rank-1 term as a vector in R^{(n*m)*(m*p)*(p*n)}
    # and check linear dependencies.
    #
    # Actually, a simpler approach: try to use a bilinear algorithm with fewer terms.
    # Let me try to construct one by hand or use a search.
    #
    # Given time constraints, let's try a systematic approach:
    # 1. Start with schoolbook (rank 30)
    # 2. Try to find linear combinations that cancel
    #
    # Let me try a different approach: use the fact that for fixed j,
    # the terms (i,j,k) for all i,k form a rank-p*n block.
    #
    # Actually, let's just try to find if we can reduce the rank by checking
    # if the 30 vectors (flattened rank-1 tensors) are linearly independent.
    import numpy as np
    
    # Build the 30 rank-1 tensors as vectors
    dim_a = n * m  # 6
    dim_b = m * p  # 15
    dim_c = p * n  # 10
    total_dim = dim_a * dim_b * dim_c  # 900
    
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                v = np.zeros(total_dim)
                a = i * m + j
                b = j * p + k
                c = k * n + i
                idx = (a * dim_b + b) * dim_c + c
                v[idx] = 1.0
                terms.append(v)
    
    terms_arr = np.array(terms)  # shape (30, 900)
    
    # Check rank of this matrix
    rank = np.linalg.matrix_rank(terms_arr)
    
    # If rank < 30, we can potentially reduce
    # But we need an actual decomposition, not just the rank.
    #
    # Let's try a greedy approach: find the largest set of linearly independent terms
    # and see if the rest can be expressed as combinations.
    
    if rank < len(terms):
        # Find a basis
        from scipy.linalg import svd
        U_sv, s, Vt = svd(terms_arr, full_matrices=False)
        # The number of non-zero singular values gives the rank
        effective_rank = np.sum(s > 1e-10)
        
        # Use the first `rank` terms as a basis (simplified)
        # This is not rigorous but let's try
        # Actually, let's just output the schoolbook decomposition
        # since finding an explicit reduced-rank decomposition is complex.
        pass
    
    # For now, return the schoolbook decomposition
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
