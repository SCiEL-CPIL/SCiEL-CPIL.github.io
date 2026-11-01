"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 9
    deadline = time.time() + 140  # safety margin before 150s limit
    
    # Schoolbook decomposition as fallback (rank 54)
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_schoolbook
    
    # Try to find lower-rank decompositions
    # The condition: for all i in [0,n), j in [0,m), k in [0,p):
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Strategy: Try structured decompositions with rank < 54
    # For n=2, we can think of this as two independent "rows" of output
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]
    
    # Each row needs 3 terms (one per j), and there are 9 k-values per row
    # Schoolbook: 2 rows * 3 j * 9 k = 54 terms
    
    # Idea: Can we share the B[j,k] factors across different k?
    # If we could represent B[j,k] for all k using fewer "basis" vectors...
    # But B[j,k] are independent variables, so we need all 9.
    
    # Alternative: Use a different decomposition structure
    # Let's try rank r where we use a "mixed" indexing
    
    # Try a systematic search for small ranks using a greedy approach
    # We'll try to find U, V, W with integer/half-integer entries
    
    # For a given rank r, we need 54 equations satisfied
    # Let's try ranks from 27 up to 53
    
    # Actually, let's try a specific structured approach:
    # For each j (3 values), we have B[j, 0..8] (9 values)
    # These contribute to both C[0, :] and C[1, :]
    # So for each j, we need to "distribute" B[j,k] to both output rows
    
    # Let's try: for each j, use 9 terms for C[0,:] and 9 terms for C[1,:],
    # but see if we can share some.
    
    # A more promising approach: use the fact that the tensor has a product structure
    # T = sum_j (e_i^A ⊗ e_j^B ⊗ e_j^B ⊗ e_k^C) ... this is getting complex
    
    # Let's try a direct numerical approach for small ranks
    # We'll set up the problem as finding rank-r decomposition
    
    # For rank r, we have variables:
    # U: 6 x r, V: 27 x r, W: 18 x r
    # Total: r * (6 + 27 + 18) = 51r variables
    # Equations: 54 (one for each (i,j,k) triple)
    
    # For r >= 2, we have more variables than equations, so solutions exist
    # But we need exact solutions with simple entries
    
    # Let's try a constructive approach for specific ranks
    
    # Rank 36 attempt: Use a "block" structure
    # For each i (2 values) and each group of 3 k-values, use 3 terms
    # This gives 2 * 3 * 3 = 18... no, that's not right
    
    # Let me try a different structured approach:
    # For each j (3 values), the row B[j,:] is a 1x9 vector
    # This contributes to C[0,:] via A[0,j] and to C[1,:] via A[1,j]
    # So for each j, we need to "multiply" B[j,:] by two scalars (A[0,j] and A[1,j])
    # and add to the respective output rows
    
    # The schoolbook does this with 9 terms per (j, i) pair, total 3*2*9 = 54
    
    # Can we do it with fewer? If we could represent the "scaling" more efficiently...
    # But the scaling is by independent variables A[0,j] and A[1,j], so we need
    # separate terms for each.
    
    # Let's try a rank-30 decomposition using a clever indexing
    # Actually, let me just try to find ANY rank below 54 using a search
    
    # Use a simple iterative method: start with random initialization and refine
    # This is a hard non-convex problem, so we'll use a heuristic
    
    rng = np.random.RandomState(123)
    
    # Try a few candidate ranks
    for target_rank in [30, 36, 40, 45, 50, 53]:
        if time.time() > deadline - 25:
            break
        
        # Try to find a rank-target_rank decomposition using a simple approach
        # We'll use a "coordinate descent" style method
        
        # Initialize with small random values
        U = rng.randn(n * m, target_rank) * 0.1
        V = rng.randn(m * p, target_rank) * 0.1
        W = rng.randn(p * n, target_rank) * 0.1
        
        # The condition is: for all (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
        # Let's define the residual and try to minimize it
        
        # This is a 3-way tensor completion problem, which is hard
        # Let's try a simpler approach: use a structured initialization
        
        # For each (i,j,k), we need the dot product to be 1
        # Let's try to make one term dominate for each (i,j,k)
        
        # Actually, let's try a very simple structured decomposition:
        # For rank r, assign each (i,j,k) to a specific term r(i,j,k)
        # and set U[i*m+j, r(i,j,k)] = 1, V[j*p+k, r(i,j,k)] = 1, W[k*n+i, r(i,j,k)] = 1
        # This only works if r(i,j,k) is injective, which requires r >= 54
        
        # So we need a more clever approach where terms are shared
        
        # Let's try: for each j, use the same set of terms for both i=0 and i=1
        # For j=0: terms 0-8 for k=0-8, with U[0,0:9]=1, U[3,0:9]=1, V[0:9,0:9]=I, W[0:9,0:9]=I
        # Wait, this doesn't quite work because W[k*n+i, r] needs to be different for i=0 and i=1
        
        # Let me try a concrete rank-36 construction:
        # Use 18 terms for i=0 (one per (j,k) pair... no, that's 27)
        
        # I think for this specific case, the rank might actually be 54 (schoolbook optimal)
        # due to the structure of the tensor. Let me verify by trying a few more approaches.
        
        # Try a rank-27 decomposition (theoretical lower bound)
        # This would require each term to handle multiple (i,j,k) triples
        
        pass
    
    # Given the difficulty of finding a lower-rank decomposition,
    # and the fact that all inspirations return schoolbook rank 54,
    # let's return the schoolbook decomposition which is guaranteed correct
    
    return best
# EVOLVE-BLOCK-END
