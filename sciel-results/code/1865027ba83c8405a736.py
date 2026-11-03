"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 5
    
    # Rank-30 construction for 3x3 * 3x5 matrix multiplication
    # Use 6 rank-1 terms per output column (5 columns * 6 = 30)
    # For each column k, compute C[:,k] = A * B[:,k] using 6 rank-1 terms
    # via a 3x2 x 2x3 factorization approach
    
    rank = 30
    U = [[0] * rank for _ in range(n * m)]  # 9 x 30
    V = [[0] * rank for _ in range(m * p)]  # 15 x 30
    W = [[0] * rank for _ in range(p * n)]  # 15 x 30
    
    # For each output column k (0..4), use 6 terms
    # Term structure for column k, sub-term s (0..5):
    # We use the identity: sum_{j=0}^{2} A[i,j]*B[j,k] 
    # = (A[i,0]+A[i,1])*B[0,k] + (A[i,1]+A[i,2])*B[1,k] + (A[i,0]+A[i,2])*B[2,k]
    #   - A[i,0]*B[1,k] - A[i,0]*B[2,k] - A[i,1]*B[0,k] - A[i,1]*B[2,k] - A[i,2]*B[0,k] - A[i,2]*B[1,k]
    # This is getting complex. Let me use a simpler valid construction.
    
    # Simpler: use schoolbook but merge terms across columns
    # For each (i,j) pair, use terms that cover multiple k values
    
    # Actually, let me use a clean rank-30 construction:
    # 30 terms where term t corresponds to:
    #   i = t % 3
    #   j = (t // 3) % 3
    #   k = (t // 9) % 5  -- but this gives 45 combos, not 30
    
    # Let me use a different indexing:
    # 30 terms, each handling a specific (i, j, k) or combination
    
    # Valid rank-30 construction using explicit terms:
    # I'll use 10 terms per "band" of 5 columns, with 3 bands
    
    # Band 0 (terms 0-9): handle rows i=0, columns k=0..4
    # Band 1 (terms 10-19): handle rows i=1, columns k=0..4  
    # Band 2 (terms 20-29): handle rows i=2, columns k=0..4
    
    # For each row i and column k, C[i,k] = sum_j A[i,j]*B[j,k]
    # Use 10 terms per row to cover all 5 columns with 3 j-values each
    
    # For row i, use terms that compute all 5 outputs C[i,0..4]
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    
    # Use 10 rank-1 terms:
    # Term 0: (A[i,0]+A[i,1]) * B[0,k] for k=0..4 -> 5 terms? No, each term is one product
    
    # Each rank-1 term is: u * v * w where u is a vector in R^9, v in R^15, w in R^15
    # For a single (i,j,k) contribution: u has 1 at (i,j), v has 1 at (j,k), w has 1 at (k,i)
    
    # To get rank 30, I need 30 such terms that exactly sum to the tensor
    
    # Let me just use 30 of the 45 schoolbook terms and verify... but that won't be complete
    
    # Final approach: use a verified rank-30 construction from explicit formulas
    
    # Terms 0-4: For i=0, j=0, k=0..4
    # Terms 5-9: For i=0, j=1, k=0..4
    # Terms 10-14: For i=0, j=2, k=0..4
    # Terms 15-19: For i=1, j=0, k=0..4
    # Terms 20-24: For i=1, j=1, k=0..4
    # Terms 25-29: For i=1, j=2, k=0..4
    # This only covers i=0,1 (15 terms for 2 rows * 3 cols * 5 outputs = 30, but I have 30 terms for 6 (i,j) pairs)
    # I'm missing i=2 entirely!
    
    # I need 9 (i,j) pairs * 5 k values = 45 for schoolbook
    # To get 30, I need to merge 15 terms
    
    # Let me use a different strategy: merge terms that share the same A or B factor
    
    # For fixed i and j, the term A[i,j]*B[j,k] for k=0..4 can be written as:
    # A[i,j] * (B[j,0], B[j,1], B[j,2], B[j,3], B[j,4])
    # This is already a single rank-1 term if I allow B to be a vector!
    # Wait, no - in the tensor decomposition, each term is a product of three vectors
    
    # The tensor T[i,j,k] = delta_{i,j,k} in the sense that T = sum_r U_r * V_r * W_r
    # where U_r is a vector in R^{n*m}, V_r in R^{m*p}, W_r in R^{p*n}
    
    # For schoolbook: T = sum_{i,j,k} e_{i,j} * e_{j,k} * e_{k,i} (45 terms)
    
    # To reduce rank, I need to find linear dependencies
    
    # For now, let me return a valid rank-30 construction using a simple method:
    # Use 6 terms per (i,j) pair for the 5 k-values, with some sharing
    
    # Actually, I'll just return a clean rank-30 that I can verify:
    # Use the fact that the 3x5 matrix B can be factored as B = B1 * B2 where B1 is 3x2 and B2 is 2x5
    # Then A*B = A*B1*B2, and I can compute A*B1 (3x2) with 6 terms and then multiply by B2
    
    # This gives: rank(A*B1) + rank((A*B1)*B2) ... no, this doesn't directly give a tensor rank
    
    # I'll implement a valid rank-30 construction using explicit term definitions
    
    # Define 30 terms:
    terms = []
    
    # For each of the 30 terms, specify (u_indices, v_indices, w_indices)
    # where each is a list of (index, value) pairs
    
    # I'll use a construction where I merge schoolbook terms in pairs
    
    # Merge (i,j,k1) and (i,j,k2) into one term with V having two nonzeros
    # But this changes the W vector to have two nonzeros, which is fine
    
    # Term merging: for fixed (i,j), merge k=0,1 into one term, k=2,3 into another, k=4 separate
    # This gives 3 terms per (i,j) instead of 5, so 9*3 = 27 terms
    # But wait, this doesn't work because W needs to be a single vector
    
    # I think the cleanest valid construction is rank 30 via:
    # 5 groups of 6 terms, each group handling one output column with 6 rank-1 terms
    
    # For column k, C[:,k] = A * B[:,k]
    # Compute this with 6 rank-1 terms using:
    # Let B_col = B[:,k] (a 3-vector)
    # C_col = A * B_col (a 3-vector)
    # This is a 3x3 * 3x1 = 3x1 multiplication, which has rank 3 (schoolbook)
    # So I need 3 terms per column, total 15 terms. But I want 30.
    
    # I'm overcomplicating this. Let me just return a valid rank-30 construction
    # that I can verify by construction.
    
    # Use 30 terms where:
    # For t in 0..29:
    #   i = t % 3
    #   j = (t // 3) % 3
    #   k = (t // 9) % 5  -- gives 45 values, but I only have 30 terms
    
    # Let me use a different mapping:
    # For t in 0..29:
    #   i = t % 3
    #   j = (t // 3) % 3
    #   k = (t // 9) % 5  -- this gives k in 0..4 for t in 0..44, but I only have 30
    
    # I'll use 30 terms with a custom mapping that covers all needed (i,j,k) triples
    
    # Actually, I realize I should just use a simple valid construction.
    # Let me use rank 30 with the following explicit terms:
    
    # Terms 0-4: i=0, j=0, k=0..4
    # Terms 5-9: i=0, j=1, k=0..4
    # Terms 10-14: i=0, j=2, k=0..4
    # Terms 15-19: i=1, j=0, k=0..4
    # Terms 20-24: i=1, j=1, k=0..4
    # Terms 25-29: i=1, j=2, k=0..4
    
    # This covers i=0,1 but not i=2! I need 45 terms for all 9 (i,j) pairs.
    
    # I cannot do rank 30 with this simple scheme. Let me use rank 45 (schoolbook)
    # but implement it cleanly, or find a valid rank < 45.
    
    # After reflection, the minimal rank for 3x3*3x5 is not trivially less than 45.
    # Let me return a valid rank-45 schoolbook construction.
    
    rank = 45
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
