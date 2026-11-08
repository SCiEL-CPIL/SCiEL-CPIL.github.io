"""Matrix multiplication 2x3 by 3x3 with rank-17 decomposition."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 3
    rank = 17
    
    U = [[0] * rank for _ in range(n * m)]  # 6 x 17
    V = [[0] * rank for _ in range(m * p)]  # 9 x 17
    W = [[0] * rank for _ in range(p * n)]  # 6 x 17
    
    # Schoolbook would give 18 terms. We save 1 rank.
    # Terms are indexed by (i, j, k) for i in 0..1, j in 0..2, k in 0..2
    # U[i*m + j, col] = 1, V[j*p + k, col] = 1, W[k*n + i, col] = 1
    
    # We'll use 17 terms. The savings comes from the structure.
    # One known construction: use 17 specific triples.
    
    # Let me enumerate 17 terms that form a valid decomposition.
    # The multiplication tensor T[i,j,k,l] = delta_{j,k} * delta_{i,l} for 
    # C[i,l] = sum_j A[i,j] B[j,l]
    # In our indexing: U row = i*m+j, V row = j*p+k, W row = k*n+i
    # Term (i,j,k) contributes to U[i*m+j], V[j*p+k], W[k*n+i]
    
    # For rank 17, we need to find 17 such that the sum of outer products
    # equals the tensor. One way: take all 18 schoolbook terms and note that
    # there's a linear relation allowing us to eliminate one.
    
    # Explicit rank-17 construction for M_{2,3,3}:
    # We use the following 17 terms (i, j, k):
    terms = [
        (0, 0, 0), (0, 0, 1), (0, 0, 2),
        (0, 1, 0), (0, 1, 1), (0, 1, 2),
        (0, 2, 0), (0, 2, 1), (0, 2, 2),
        (1, 0, 0), (1, 0, 1), (1, 0, 2),
        (1, 1, 0), (1, 1, 1), (1, 1, 2),
        (1, 2, 0), (1, 2, 1),
    ]
    # That's only 17 terms but we're missing (1,2,2). This won't work directly.
    
    # Let me use a different approach. The known rank-17 solution uses
    # half-integer or integer coefficients. Let me construct it properly.
    
    # Actually, for M_{2,3,3}, the rank is 17 and a construction is:
    # Use 17 rank-1 tensors. Here's one explicit construction:
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # 17 terms with coefficients. Using the standard construction:
    # For each of the 18 schoolbook positions, we have a term, but we combine
    # two of them using a linear identity.
    
    # Simple approach: just use rank 18 first, then try to verify if 17 works.
    # Given time constraints, let me provide the rank-17 solution from known results.
    
    # Known rank-17 decomposition for 2x3 * 3x3:
    # The 17 terms are constructed as follows. Let's label them c=0..16.
    
    # Terms 0-8: all terms for i=0 (9 terms)
    for c, (i, j, k) in enumerate([(0,0,0),(0,0,1),(0,0,2),(0,1,0),(0,1,1),(0,1,2),(0,2,0),(0,2,1),(0,2,2)]):
        U[i * m + j][c] = 1
        V[j * p + k][c] = 1
        W[k * n + i][c] = 1
    
    # Terms 9-16: 8 terms for i=1, but we need to cover all 9 positions (1,j,k)
    # We can't just drop one. Instead, use a modified construction.
    
    # Let me restart with a cleaner approach using the actual known solution.
    # The rank of M_{2,3,3} is 17. Here's a valid construction:
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Using the construction from Landsberg's work or similar:
    # We can write the 18 schoolbook terms and apply a linear transformation
    # that reduces the rank by 1.
    
    # For now, let me use a verified rank-17 construction.
    # The key idea: for the i=1 row, we use 8 terms plus modify one i=0 term.
    
    # Here's a concrete rank-17 solution:
    # c=0: (0,0,0) U[0]=1, V[0]=1, W[0]=1
    # c=1: (0,0,1) U[0]=1, V[1]=1, W[1]=1  -> W row = 1*2+0=2
    # c=2: (0,0,2) U[0]=1, V[2]=1, W[2]=1  -> W row = 2*2+0=4
    # c=3: (0,1,0) U[1]=1, V[3]=1, W[3]=1  -> W row = 0*2+0=0, but that conflicts
    # 
    # I'm making errors in indexing. Let me be very careful.
    # U index: i*m + j = i*3 + j
    # V index: j*p + k = j*3 + k  
    # W index: k*n + i = k*2 + i
    #
    # For term (i,j,k): U[i*3+j], V[j*3+k], W[k*2+i]
    
    # Let me just use rank 18 which is definitely correct, then attempt 17.
    # Given the complexity, let me provide a correct rank-18 first, then optimize.
    
    # Actually, let me just hardcode the known rank-17 solution properly.
    
    # After careful construction, here is a rank-17 solution:
    # We use 17 terms. The trick is that one of the W entries can be shared.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]  
    W = [[0] * rank for _ in range(p * n)]
    
    # 17 terms:
    # (0,0,0), (0,0,1), (0,0,2), (0,1,0), (0,1,1), (0,1,2), (0,2,0), (0,2,1), (0,2,2) = 9 terms for i=0
    # (1,0,0), (1,0,1), (1,0,2), (1,1,0), (1,1,1), (1,1,2), (1,2,0), (1,2,1) = 8 terms for i=1
    # Total: 17 terms. But we're missing (1,2,2).
    
    # This naive selection doesn't work. We need a more clever construction.
    
    # Let me use the actual known solution. The rank-17 decomposition of M_{2,3,3}
    # uses the following 17 rank-1 tensors with integer coefficients:
    
    # I'll use a verified construction. The 17 terms are:
    terms_17 = [
        # (i, j, k, coeff_U, coeff_V, coeff_W)
        (0, 0, 0, 1, 1, 1),
        (0, 0, 1, 1, 1, 1),
        (0, 0, 2, 1, 1, 1),
        (0, 1, 0, 1, 1, 1),
        (0, 1, 1, 1, 1, 1),
        (0, 1, 2, 1, 1, 1),
        (0, 2, 0, 1, 1, 1),
        (0, 2, 1, 1, 1, 1),
        (0, 2, 2, 1, 1, 1),
        (1, 0, 0, 1, 1, 1),
        (1, 0, 1, 1, 1, 1),
        (1, 0, 2, 1, 1, 1),
        (1, 1, 0, 1, 1, 1),
        (1, 1, 1, 1, 1, 1),
        (1, 1, 2, 1, 1, 1),
        (1, 2, 0, 1, 1, 1),
        (1, 2, 1, 1, 1, 1),
    ]
    
    # This is missing (1,2,2). This construction is invalid.
    
    # Let me just use rank 18 which is guaranteed correct, and accept that.
    # The score will be 18. If the optimal is 17, we lose a bit, but we have a valid solution.
    
    rank = n * m * p  # 18
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
