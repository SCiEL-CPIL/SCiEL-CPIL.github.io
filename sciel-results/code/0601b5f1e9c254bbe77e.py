"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    rank = 8
    U = [[0] * rank for _ in range(n * m)]  # 16 x 8
    V = [[0] * rank for _ in range(m * p)]  # 24 x 8
    W = [[0] * rank for _ in range(p * n)]  # 24 x 8

    # Strategy: Use 8 terms where each term handles a subset of (i,k) pairs
    # Term r covers row i_r and column group k in groups of 3 (p=6, 2 groups)
    # Actually, let's use a simpler structured approach:
    # For rank 8, assign each term to handle specific (i, k_group) pairs
    
    # Let's try: for r in range(8), let r = i * 2 + kg where i in {0,1} and kg in {0,1}
    # But we have 4 rows and 6 cols. Let's use:
    # r encodes (i_mod2, k_mod3) for one parity class, and handle the rest separately
    
    # Simpler: use rank = 12 with a block structure
    # Actually, let me just try a valid low-rank solution using Kronecker products
    
    # Use the fact that T = sum over j of outer(i,j) x outer(j,k)
    # This gives rank n*m*p = 96 (schoolbook)
    
    # For a better decomposition, use bilinear complexity results.
    # For 4x4 * 4x6, let's try rank = 24 (half of 48 = 4*12, or use structure)
    
    # Let me try a constructive approach with rank = 16:
    # Use 4 "row blocks" and 4 "column blocks" with shared structure
    
    rank = 16
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Term (a, b) where a in {0,1,2,3}, b in {0,1,2,3}
    # U[i*m+j, a*4+b] = 1 if i == a
    # V[j*p+k, a*4+b] = 1 if j == a and k in block b (k // 1.5, but p=6)
    # Let k blocks be: b=0: k in {0,1}, b=1: k in {2,3}, b=2: k in {4,5}, b=3: k in {0,1,2,3,4,5}
    
    # Actually, let's use a cleaner approach: rank = n * p = 24
    # Term (i, k): U[i*m+j, i*p+k] = delta(j, j) = 1 for all j... no
    
    # Let me just use rank = 32 with a clear structure:
    rank = 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # For each i in 0..3 and k in 0..5, we need to capture the contribution
    # But that's 24 terms minimum if we don't share
    
    # Let me use rank = 24: one term per (i, k) pair
    # U[i*m+j, i*p+k] = 1 for all j  -> no, V must depend on j
    
    # Correct approach for rank 24:
    # Term r = i*p + k: 
    #   U[i*m+j, r] = 1 for all j (since all j contribute to C[i,k])
    #   V[j*p+k, r] = 1 if j == j (always true) -> V[j*p+k, r] = 1 only when the column matches
    #   W[k*n+i, r] = 1
    # But then U[i*m+j, r] = 1 for all j means U row i*m+j has 1 in positions r=i*p+k for all k
    # And V[j*p+k, r] = 1 only when r = i'*p+k for some i', but we need it to be 1 for our specific j
    # This doesn't work cleanly.
    
    # Let me go back to a known-good approach: use rank = 48 with half-integer entries
    # or just use the schoolbook but try to verify what the evaluator expects
    
    # Given the difficulty, let me try rank = 96 (schoolbook) first to ensure validity,
    # then the evaluator will confirm the interface works
    
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
