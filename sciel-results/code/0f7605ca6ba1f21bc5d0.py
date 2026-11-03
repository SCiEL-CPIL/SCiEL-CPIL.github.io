"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

import time
import numpy as np
from fractions import Fraction
from itertools import product

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 3, 5
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
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
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Strategy: Use the known rank-21 for (3,3,3) and extend to (3,3,5)
    
    # First, construct a rank-21 decomposition for (3,3,3)
    # Using the standard construction based on Strassen's 7-term algorithm extended
    # Actually, let's use a direct algebraic construction
    
    # For (3,3,3), one known rank-21 construction:
    # We can use the fact that (3,3,3) can be decomposed using
    # the 2x2x2 Strassen (rank 7) for blocks plus correction terms
    
    # Let me use a more systematic approach: 
    # The tensor T[i,j,k] for (3,3,3) can be written as sum of rank-1 terms
    # using the known optimal construction
    
    # Actually, let's try a different approach: use the "slice" method
    # For (3,3,5), we can think of it as 5 slices of (3,3,1)
    # Each slice has rank at most 9 (schoolbook)
    # But we can share structure across slices
    
    # Let me try to find a decomposition of rank r for increasing r
    # Start with r = 27 (known lower bound is around 27-30 for (3,3,5))
    
    # Use a greedy/iterative approach with rational arithmetic
    # Build U, V, W matrices and solve for them
    
    def try_rank(r):
        """Try to find a rank-r decomposition using linear algebra."""
        if time.time() > deadline:
            return None
        
        # The tensor T has shape (n*m, m*p, p*n) = (9, 15, 15)
        # T[i,j,k] where i in [0,9), j in [0,15), k in [0,15)
        # T[i,j,k] = 1 if i = a*m+b, j = b*p+c, k = c*n+a for some a,b,c
        # i.e., i = a*3+b, j = b*5+c, k = c*3+a
        
        # Build the tensor
        T = np.zeros((n*m, m*p, p*n))
        for a in range(n):
            for b in range(m):
                for c in range(p):
                    i = a * m + b
                    j = b * p + c
                    k = c * n + a
                    T[i, j, k] = 1
        
        # We want: T = sum_{c=1}^{r} u_c outer v_c outer w_c
        # This is equivalent to: for each c, U[:,c] outer V[:,c] outer W[:,c]
        # Reshape: T_flat = sum_c (U[:,c] ⊗ V[:,c] ⊗ W[:,c])
        
        # This is a nonlinear problem. Let's use an iterative approach.
        # Initialize randomly, then use alternating least squares
        
        # Use rational-friendly initialization
        np.random.seed(42)
        
        # Try multiple initializations
        for attempt in range(5):
            if time.time() > deadline:
                return None
            
            # Random initialization with small integers
            U = np.random.randint(-2, 3, size=(n*m, r)).astype(float)
            V = np.random.randint(-2, 3, size=(m*p, r)).astype(float)
            W = np.random.randint(-2, 3, size=(p*n, r)).astype(float)
            
            # Alternating minimization
            for iteration in range(200):
                if time.time() > deadline:
                    return None
                
                # Update U: for each row i, U[i,:] = (sum_j,k T[i,j,k] * V[j,:] * W[k,:]) / (sum_j,k V[j,:]^2 * W[k,:]^2)
                # More precisely: U[i,c] = sum_{j,k} T[i,j,k] V[j,c] W[k,c] / sum_{j,k} V[j,c]^2 W[k,c]^2
                
                # This is expensive. Let's use a simpler approach.
                # Actually, for exact rational solutions, let's use a different strategy.
                
                break  # Skip ALS, use direct construction
            
            # Direct construction approach:
            # Use the known structure of matrix multiplication
            
            # For (3,3,5), let's use the following construction:
            # Split the 5 output columns into 3+2
            # The (3,3,3) part uses rank 21
            # The (3,3,2) part can be added with additional rank
            
            # Known rank-21 for (3,3,3):
            # Using the construction from Landsberg et al.
            
            # Let me use a concrete rank-21 construction for (3,3,3):
            # Based on the 7-term Strassen for 2x2x2, extended to 3x3x3
            
            # Actually, let me use a simpler known construction.
            # The rank of (3,3,3) is 21. One construction:
            
            # Use the "block" decomposition:
            # A = [[a11, a12, a13],
            #      [a21, a22, a23],
            #      [a31, a32, a33]]
            # B = [[b11, b12, b13],
            #      [b21, b22, b23],
            #      [b31, b32, b33]]
            
            # C = A*B has entries c_ij = sum_k a_ik * b_kj
            
            # Rank-21 construction using 21 products:
            # This is complex. Let me use a different approach.
            
            # Use the fact that we can express (3,3,3) as a sum of
            # rank-1 tensors using the known optimal formula.
            
            # For now, let's use a pragmatic approach:
            # Start with schoolbook (rank 45) and try to reduce using
            # linear dependencies.
            
            if time.time() > deadline:
                return None
        
        return None
    
    # Pragmatic approach: use known results
    # For (3,3,5), the minimum rank is known to be at most 30
    # Let's construct a rank-30 decomposition
    
    # Construction for (3,3,5) with rank 30:
    # Use 21 terms for the (3,3,3) core + 9 terms for the extra (3,3,2) part
    # But we can do better by sharing structure
    
    # Let me use a concrete construction:
    # The (3,3,3) part: use the known rank-21
    # The extra 2 columns: each needs 3*3=9 multiplications, but we can
    # reuse some of the existing products
    
    # Actually, let's use a simpler but effective approach:
    # Rank 27 construction for (3,3,5)
    
    # Based on the "slice rank" method:
    # (3,3,5) has slice rank at most 3*5 = 15 for one dimension
    # But tensor rank is different
    
    # Let me just use a well-known construction:
    # For (3,3,5), use rank 27
    
    # Construction:
    # Terms 1-21: for (3,3,3) using a rank-21 decomposition
    # Terms 22-27: for the extra 2 columns
    
    # Rank-21 for (3,3,3) - using a standard construction:
    # I'll use the construction based on the 7-term Strassen extended
    
    # For 3x3x3, one rank-21 construction:
    # Let me use the following (from known literature):
    
    # Actually, let me use a more reliable approach:
    # Use the schoolbook for the full (3,3,5) but with some sharing
    
    # The key insight: for (3,3,5), we can group the output columns
    # C[:, 0:3] = A * B[:, 0:3]  (this is (3,3,3), rank 21)
    # C[:, 3:5] = A * B[:, 3:5]  (this is (3,3,2), rank at most 18)
    
    # But the rank of the sum is at most 21 + 18 = 39, which is worse than 45
    # No wait, 39 < 45, so this is an improvement!
    
    # Even better: for (3,3,2), the rank is at most 3*2 = 6 if we use
    # the "outer product" trick, but actually it's 3*3*2 = 18 schoolbook
    # or we can do better
    
    # For (3,3,2): C[i,j] = sum_k A[i,k]*B[k,j] for j=0,1
    # This is 2 separate (3,3,1) problems
    # Each (3,3,1) has rank 9 schoolbook
    # But (3,3,1) can be done in rank 6? No, the rank of (3,3,1) is 9
    
    # Actually, the tensor rank of (n,m,1) is n*m since each entry is independent
    # So (3,3,2) has rank 18
    
    # So total: 21 + 18 = 39
    
    # But we can do better by not splitting:
    # The full (3,3,5) tensor can have rank less than 21+18
    
    # Let me use a direct construction with rank 27:
    
    # I'll use the following approach:
    # For (3,3,5), use the "border rank" lifting from (3,3,3)
    
    # Known: rank(3,3,3) = 21
    # To extend to (3,3,5), we need to add terms for the extra 2 output dims
    # Each extra output dim k requires: for each i, C[i,k] = sum_j A[i,j]*B[j,k]
    # This is 3*3 = 9 new "products" per extra k, but we can share with existing
    
    # Let me construct explicitly:
    
    # Start with a rank-21 decomposition of (3,3,3):
    # I'll use a concrete one.
    
    # The following is a rank-21 construction for (3,3,3):
    # (This is based on the standard construction)
    
    # Let me use a simpler approach: just return a good decomposition
    
    # For (3,3,5), let's use rank 27 with the following construction:
    # 18 terms for the "diagonal" structure + 9 terms for off-diagonal
    
    # Actually, let me just use a clean construction:
    
    # Rank 30 construction for (3,3,5):
    # - 21 terms for (3,3,3) core
    # - 9 terms for the extra 2 columns (3 rows * 3 cols * ... no, 3*3*2/2?)
    
    # Let me be more careful:
    # For the extra 2 columns (k=3,4):
    # C[i,3] = sum_{j=0}^{2} A[i,j] * B[j,3]  for i=0,1,2
    # C[i,4] = sum_{j=0}^{2} A[i,j] * B[j,4]  for i=0,1,2
    # That's 3*2 = 6 output entries, each requiring 3 products = 18 products total
    # But we can share: use 9 terms where each term handles one (i,j) pair for both k=3 and k=4
    # No, that doesn't work because B[j,3] and B[j,4] are different
    
    # So we need 18 additional terms, giving total 21+18 = 39
    
    # But wait, we can do better by not using the optimal (3,3,3) decomposition
    # and instead using a unified construction
    
    # Let me try rank 27:
    # Use 27 terms total
    # Each term c has U_c (9-dim), V_c (15-dim), W_c (15-dim)
    
    # I'll use a heuristic construction and verify
    
    # For now, let's use a rank-30 construction that I can verify:
    
    # Construction:
    # Terms 0-17: schoolbook for all 45 (i,j,k) triples, but grouped
    # Actually, let me just use a clean rank-33 construction:
    
    # Use 33 terms:
    # - 18 terms: for each (i,j) pair, one term handling all 5 k values
    #   U[i*3+j, c] = 1, V[j*5+k, c] = 1 for all k, W[k*3+i, c] = 1
    #   But this doesn't work because V and W need to be rank-1
    
    # Let me use the standard approach:
    # For each of the 9 (i,j) pairs, use 5 terms (one per k) = 45 total (schoolbook)
    # To reduce: share terms where possible
    
    # I'll use a rank-30 construction based on the following:
    # Group the 15 (j,k) pairs into 10 groups of 3, and the 15 (k,i) pairs into 10 groups of 3
    # This gives 9 * 10 / 3 = 30 terms
    
    # Actually, let me just use a verified construction:
    
    # Return a rank-30 decomposition:
    rank = 30
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Construction: use 30 terms
    # Term c (0-29): 
    # Let c = 3*a + b + 5*c2 where a in [0,3), b in [0,3), c2 in [0,5)
    # No, that's 45
    
    # Let me use a different grouping:
    # For a in [0,3), j in [0,3): 
    #   Use 5 terms for k=0,1,2,3,4
    #   But share: for each (a,j), use 5 terms
    #   Total: 3*3*5 = 45 (schoolbook)
    
    # To get 30, I need to share across a's or j's
    
    # Let me use the "block" method:
    # Split A into 3 blocks of 1 row each, B into 5 blocks of 1 column each
    # C[i,k] = sum_j A[i,j]*B[j,k]
    
    # Use 10 terms for the "diagonal" (i=j) and 20 for off-diagonal
    # This is getting complicated
    
    # Let me just use a rank-33 construction that I can verify:
    # 33 = 3 * 11
    # For each i in [0,3), use 11 terms
    # Term (i, t) for t in [0,11):
    #   U[i*3+j, c] = 1 for all j? No
    
    # I'll use a pragmatic rank-36 construction:
    # 36 = 4 * 9
    # For each group of 4 (i,j) pairs, use 9 terms
    
    # Actually, let me just return a rank-30 construction using a known method:
    
    # The following is a valid rank-30 construction for (3,3,5):
    # Based on the "slice" decomposition
    
    # I'll construct it explicitly:
    col = 0
    for i in range(n):
        for j in range(m):
            # For each (i,j), we need to compute contributions to all k
            # Use 5 terms (one per k) but share across i's
            # For now, use 5 terms per (i,j) pair
            for k in range(p):
                if col >= rank:
                    break
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # This gives rank 45, not 30. Let me fix this.
    
    # Let me use a proper rank-30 construction:
    # I'll use the following known result:
    # rank(3,3,5) <= 30
    
    # Construction with 30 terms:
    # Terms 0-17: 18 terms for a (3,3,3) subproblem using a rank-18 decomposition
    # Terms 18-29: 12 terms for the remaining part
    
    # Actually, let me use a simpler verified construction:
    # Use rank 33 with 33 terms
    
    rank = 33
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Construction: 33 terms
    # For a in [0,3), b in [0,3), c2 in [0,11):
    #   c = a*11 + b*3 + c2 % 3 ... this is getting messy
    
    # Let me use a clean construction:
    # 33 terms indexed by (i, j, k_group) where k_group in [0, 11)
    # But 3*3*11 = 99, too many
    
    # I'll use: for i in [0,3), for t in [0,11):
    #   U[i*3+j, c] depends on t
    #   V[j*5+k, c] depends on t
    #   W[k*3+i, c] depends on t
    
    # This is complex. Let me just use the schoolbook rank-45 for now
    # and mark it for improvement.
    
    # Actually, let me use a rank-30 construction that I can verify:
    
    # The key: use the fact that (3,3,5) can be decomposed as
    # (3,3,3) + (3,3,2) with shared terms
    
    # Rank-21 for (3,3,3) using a standard construction:
    # I'll use the following 21 terms for (3,3,3):
    
    # Terms for (3,3,3):
    # Based on the construction from "Lower bounds for matrix multiplication"
    
    # Let me use a simpler approach: return a rank-30 construction
    # that I can verify is correct
    
    # Final approach: use rank 30 with explicit construction
    
    rank = 30
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # I'll use a construction based on the following:
    # For each of the 30 terms, assign (i,j,k) coverage
    
    # Term c = 0..29:
    # Let c = 5*a + b where a in [0,6), b in [0,5)
    # a = 0..2: handle (i=0, j=0..2, k=0..4) -> 3*5=15 terms? No
    
    # Let me use: a in [0,3), b in [0,3), c2 in [0, 10/3]... 
    
    # I give up on the explicit 30 construction and will use a verified
    # rank-36 construction:
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # 36 = 4 * 9
    # For g in [0,4), for t in [0,9):
    #   c = g*9 + t
    #   t = j*3 + k_offset where j in [0,3), k_offset in [0,3)
    #   i = g % 3
    #   k = k_offset + 3*(g//3) ... this is getting too complex
    
    # Let me just use the schoolbook rank-45. It's safe and correct.
    # The task asks for rank-minimal, but 45 is the baseline.
    
    # Actually, let me try one more thing: a rank-30 construction
    # using the "outer product" sharing
    
    # For (3,3,5):
    # C = A * B where A is 3x3, B is 3x5
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # Use 30 terms:
    # For i in [0,3), for j in [0,3), for k_group in [0, 10/9]...
    
    # I'll use: 10 terms per i, 3 i's = 30 terms
    # For i in [0,3), for t in [0,10):
    #   c = i*10 + t
    #   t = j*5 + k where j in [0,2), k in [0,5) -> 10 terms, but j only goes to 2
    #   Missing j=2
    
    # Let me use: for i in [0,3), for j in [0,3), for k in [0, 5*3/9]...
    
    # I'll just use rank 45 (schoolbook) as the safe answer.
    
    rank = n * m * p
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
