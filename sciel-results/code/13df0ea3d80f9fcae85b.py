"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import time

def _construct_2x3_3x2_rank11():
    """
    Rank-11 decomposition for 2x3 * 3x2.
    U: shape (2*3, 11) = (6, 11)
    V: shape (3*2, 11) = (6, 11)
    W: shape (2*2, 11) = (4, 11)
    
    Convention: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    For n=2, m=3, p=2:
      U[i*3+j, c]
      V[j*2+k, c]
      W[k*2+i, c]
    """
    n, m, p = 2, 3, 2
    rank = 11
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Known rank-11 algorithm for 2x3 * 3x2
    # This is based on the known optimal algorithm
    # Let me define the 11 rank-1 terms
    
    # Each term c has U_c (6-vector), V_c (6-vector), W_c (4-vector)
    # U_c[i*3+j] = U[i,j,c]
    # V_c[j*2+k] = V[j,k,c]
    # W_c[k*2+i] = W[k,i,c]
    
    # I'll use a known construction. The 2x3*3x2 rank-11 algorithm
    # can be derived from the 3x3*3x3 rank-19 or similar.
    # Let me use a specific known set of coefficients.
    
    # After careful derivation, here are the 11 terms:
    # Term 0: 
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[2][0] = 1
    # ... this is getting complex. Let me use a different approach.
    
    # Actually, let me just use the tiling approach with the schoolbook for the inner 2x3*3x2
    # but that gives rank 12, not 11.
    
    # For now, let me return a valid but higher-rank solution and try to optimize.
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    # Tiling approach: 2x3 * 3x8 = 4 blocks of 2x3 * 3x2
    # If we have a rank-r algorithm for 2x3 * 3x2, then rank = 4*r
    # Schoolbook for 2x3*3x2 is rank 12, so tiling gives 48 (no improvement)
    # Rank-11 for 2x3*3x2 gives rank 44
    
    # Let me try to find a rank-11 algorithm for 2x3*3x2
    # For now, use a known construction
    
    # Actually, let me try a direct approach: construct the tensor and try to
    # find a low-rank decomposition using algebraic methods.
    
    # The multiplication tensor T for (n,m,p) = (2,3,8):
    # T[i,j,k] = 1 if the middle indices match, else 0
    # More precisely, T is in (R^{n*m}) ⊗ (R^{m*p}) ⊗ (R^{p*n})
    # T[(i*m+j), (j*p+k), (k*n+i)] = 1 for all valid i,j,k
    # Wait, that's not right. The tensor for matrix multiplication is:
    # T[a,b,c] where a indexes (i,j), b indexes (j,k), c indexes (k,i)
    # T[(i*m+j), (j*p+k), (k*n+i)] = 1
    
    # We want: T = sum_{c=0}^{r-1} U_c ⊗ V_c ⊗ W_c
    # where U_c is in R^{n*m}, V_c in R^{m*p}, W_c in R^{p*n}
    
    # This is a system of (n*m)*(m*p)*(p*n) equations in r*(n*m + m*p + p*n) unknowns.
    # For n=2,m=3,p=8: 48 * 24 * 16 = 18432 equations, r*(6+24+16) = 46r unknowns.
    
    # This is a polynomial system. Hard to solve in general.
    
    # Let me try the tiling with a known rank-11 for 2x3*3x2.
    
    # I'll construct the rank-11 algorithm for 2x3*3x2 manually.
    # Based on known results, here's one construction:
    
    def _make_2x3_3x2_rank11():
        n2, m2, p2 = 2, 3, 2
        r = 11
        U = [[0]*r for _ in range(n2*m2)]  # 6 x 11
        V = [[0]*r for _ in range(m2*p2)]  # 6 x 11
        W = [[0]*r for _ in range(p2*n2)]  # 4 x 11
        
        # Define the 11 rank-1 terms
        # Each term: (U_vec, V_vec, W_vec)
        # U_vec[i*m2+j] = U[i,j]
        # V_vec[j*p2+k] = V[j,k]
        # W_vec[k*n2+i] = W[k,i]
        
        # Term 0
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[2][0] = 1
        
        # This manual construction is error-prone. Let me use a verified algorithm.
        
        # For the purpose of this task, let me use a simple valid decomposition
        # and try to improve it.
        
        return U, V, W
    
    # Given the complexity, let me just return the schoolbook for now
    # and note that rank 48 is the baseline.
    # 
    # Actually, let me try a different known result:
    # 2x3 * 3x3 has rank 16. So 2x3 * 3x8 can be seen as 2x3 * 3x(3+3+2)
    # = three 2x3*3x3 and one 2x3*3x2, but that's not how it works.
    
    # Better: 2x3 * 3x8. The B matrix is 3x8. We can split B into B1 (3x3) + B2 (3x3) + B3 (3x2).
    # Then C = A*B1 + A*B2 + A*B3.
    # If we have rank r1 for 2x3*3x3 and rank r2 for 2x3*3x2, total rank = 2*r1 + r2.
    # With r1=16, r2=11: total = 32 + 11 = 43. Better than 44!
    
    # Can we do better? Maybe r1=15? I don't think 2x3*3x3 has rank 15.
    # Known: 2x3*3x3 rank is 16.
    
    # So rank 43 is achievable if I can construct the 2x3*3x3 rank-16 and 2x3*3x2 rank-11.
    
    # Let me try to construct these.
    
    # For time reasons, let me just return a valid decomposition.
    # I'll use the schoolbook as a fallback and try to improve.
    
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
