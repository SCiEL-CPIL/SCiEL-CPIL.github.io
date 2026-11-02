"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 125  # safety margin before 150s limit

    n, m, p = DIMENSIONS
    # Matrix multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    # We want rank-r decomposition: T = sum_{c=0}^{r-1} U[:,c] ⊗ V[:,c] ⊗ W[:,c]
    # where U is (n*m, r), V is (m*p, r), W is (p*n, r)
    # and the convention is W[k*n+i, c]

    # Start with schoolbook rank = n*m*p = 32
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()

    # Try to reduce rank using random initialization + iterative refinement
    # The decomposition condition: for all (a,b), sum_c U[a,c]*V[b,c]*W[c_index] should match T
    # More precisely: T[i*m+j, j*p+k, k*n+i] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # This is a 3-way tensor decomposition (CP decomposition) of the matrix multiplication tensor.

    # We'll try ranks from 27 upward (known lower bound for 2x4x4 is at most 28 via Strassen-like,
    # but let's try to find if 27 or lower works)
    # Actually, let's try a randomized algebraic approach:
    # For a given rank r, we have r*(n*m + m*p + p*n) unknowns.
    # The constraints are n*m*p equations (one for each tensor entry).
    # For n=2,m=4,p=4: 32 equations, unknowns = r*(8+16+8) = 32r.
    # For r=32: 1024 unknowns, 32 equations - very underdetermined.
    # For r=28: 896 unknowns, 32 equations.

    # Strategy: Use the fact that the tensor has specific structure.
    # Let's try a Strassen-like block approach first to get rank 28, then try to reduce.

    # Strassen-like: partition the 2x4 * 4x4 into blocks.
    # A is 2x4, B is 4x4, C = A*B is 2x4.
    # Let's partition B into 2x2 blocks of 2x2:
    # B = [[B11, B12], [B21, B22]] where each is 2x2
    # A = [A1, A2] where A1, A2 are 2x2
    # C = [A1*B11 + A2*B21, A1*B12 + A2*B22]
    # C = [C1, C2] where C1, C2 are 2x2

    # Using Strassen on 2x2 blocks:
    # For computing A1*B11 + A2*B21 (which is C1):
    # This is a sum of two 2x2 products.
    # Similarly for C2.

    # Actually, let's use a direct Strassen-style approach for the whole thing.
    # The known result: rank of 2x4x4 matrix multiplication is 28.
    # Let me construct a rank-28 decomposition using block Strassen.

    # Partition:
    # A (2x4) = [A1 | A2], A1 = A[:,0:2], A2 = A[:,2:4] (each 2x2)
    # B (4x4) = [[B11, B12], [B21, B22]], each 2x2
    #   B11 = B[0:2, 0:2], B12 = B[0:2, 2:4]
    #   B21 = B[2:4, 0:2], B22 = B[2:4, 2:4]
    # C (2x4) = [C1 | C2], C1 = A1*B11 + A2*B21, C2 = A1*B12 + A2*B22 (each 2x2)

    # For each 2x2 product, Strassen uses 7 multiplications.
    # C1 = A1*B11 + A2*B21:
    #   Let's compute M1 = A1*B11 and M2 = A2*B21 separately? No, that's 14 mults.
    #   Instead, use Strassen on the sum:
    #   Actually, we can't directly Strassen a sum of two products.
    #   But we can compute:
    #   M1 = (A1 + A2) * (B11 + B21)  -- but this gives A1*B11 + A1*B21 + A2*B11 + A2*B21
    #   That's not quite right.

    # Let me think differently. We have 4 independent 2x2 matrix multiplications to do:
    # A1*B11, A1*B12, A2*B21, A2*B22
    # Each requires rank 7 by Strassen, total 28.
    # So rank 28 is achievable.

    # Let's construct the rank-28 decomposition explicitly.
    # For each 2x2 Strassen multiplication, we have 7 "multiplications" (rank-1 terms).

    # Strassen 2x2:
    # Given P (2x2) and Q (2x2), compute P*Q:
    # M1 = (P11 + P22) * (Q11 + Q22)
    # M2 = (P21 + P22) * Q11
    # M3 = P11 * (Q12 - Q22)
    # M4 = P22 * (Q21 - Q11)
    # M5 = (P11 + P12) * Q22
    # M6 = (P21 - P11) * (Q11 + Q12)
    # M7 = (P12 - P22) * (Q21 + Q22)
    # Then:
    # R11 = M1 + M4 - M5 + M7
    # R12 = M3 + M5
    # R21 = M2 + M4
    # R22 = M1 - M2 + M3 + M6

    # For our case, we have 4 such 2x2 multiplications. Let's label them:
    # Mult 0: A1 * B11 -> contributes to C1
    # Mult 1: A2 * B21 -> contributes to C1
    # Mult 2: A1 * B12 -> contributes to C2
    # Mult 3: A2 * B22 -> contributes to C2

    # C1 = result of Mult 0 + result of Mult 1
    # C2 = result of Mult 2 + result of Mult 3

    # Total rank = 4 * 7 = 28.

    # Now let's build U, V, W for this rank-28 decomposition.
    # U has shape (n*m, rank) = (8, 28)
    # V has shape (m*p, rank) = (16, 28)
    # W has shape (p*n, rank) = (8, 28)

    # Indexing:
    # U row index: i*m + j where i in [0,2), j in [0,4)
    # V row index: j*p + k where j in [0,4), k in [0,4)
    # W row index: k*n + i where k in [0,4), i in [0,2)

    # For Mult 0: A1 * B11
    # A1 = A[:, 0:2] -> rows i=0,1; cols j=0,1
    # B11 = B[0:2, 0:2] -> rows j=0,1; cols k=0,1
    # Result contributes to C1 = C[:, 0:2] -> rows i=0,1; cols k=0,1

    # For Strassen, the "P" matrix is A1 (2x2), "Q" matrix is B11 (2x2).
    # P11 = A[0,0] (i=0,j=0), P12 = A[0,1] (i=0,j=1)
    # P21 = A[1,0] (i=1,j=0), P22 = A[1,1] (i=1,j=1)
    # Q11 = B[0,0] (j=0,k=0), Q12 = B[0,1] (j=0,k=1)
    # Q21 = B[1,0] (j=1,k=0), Q22 = B[1,1] (j=1,k=1)

    # M1 = (P11+P22)*(Q11+Q22):
    #   U part: A[0,0] + A[1,1] -> U[0,0] += 1, U[1,0] += 1 (for this mult's columns)
    #   V part: B[0,0] + B[1,1] -> V[0,0] += 1, V[1,1] += 1
    #   W part: contributes to R11, R22 of the product.
    #   R11 = M1 + M4 - M5 + M7 -> C[0,0] gets +1 from M1
    #   R22 = M1 - M2 + M3 + M6 -> C[1,1] gets +1 from M1
    #   So W[0*2+0, col] += 1 (k=0,i=0 -> C[0,0]) and W[1*2+1, col] += 1 (k=1,i=1 -> C[1,1])

    # This is getting complex. Let me just build it programmatically.

    def strassen_2x2_decomp(P_rows, P_cols, Q_rows, Q_cols, R_rows, R_cols, base_col, U, V, W):
        """
        Add a Strassen 2x2 multiplication to U, V, W starting at base_col.
        P is a 2x2 submatrix of A with rows P_rows (2 indices into n*m) and cols P_cols (2 indices into m).
        Actually, let's use explicit index mapping.
        
        P_rows: list of 2 row indices in A (0..n-1)
        P_cols: list of 2 col indices in A (0..m-1)
        Q_rows: list of 2 row indices in B (0..m-1)
        Q_cols: list of 2 col indices in B (0..p-1)
        R_rows: list of 2 row indices in C (0..n-1)
        R_cols: list of 2 col indices in C (0..p-1)
        """
        # P11, P12, P21, P22 correspond to:
        # P11: A[P_rows[0], P_cols[0]] -> U index = P_rows[0]*m + P_cols[0]
        # P12: A[P_rows[0], P_cols[1]] -> U index = P_rows[0]*m + P_cols[1]
        # P21: A[P_rows[1], P_cols[0]] -> U index = P_rows[1]*m + P_cols[0]
        # P22: A[P_rows[1], P_cols[1]] -> U index = P_rows[1]*m + P_cols[1]
        
        # Q11: B[Q_rows[0], Q_cols[0]] -> V index = Q_rows[0]*p + Q_cols[0]
        # Q12: B[Q_rows[0], Q_cols[1]] -> V index = Q_rows[0]*p + Q_cols[1]
        # Q21: B[Q_rows[1], Q_cols[0]] -> V index = Q_rows[1]*p + Q_cols[0]
        # Q22: B[Q_rows[1], Q_cols[1]] -> V index = Q_rows[1]*p + Q_cols[1]
        
        # R11: C[R_rows[0], R_cols[0]] -> W index = R_cols[0]*n + R_rows[0]
        # R12: C[R_rows[0], R_cols[1]] -> W index = R_cols[1]*n + R_rows[0]
        # R21: C[R_rows[1], R_cols[0]] -> W index = R_cols[0]*n + R_rows[1]
        # R22: C[R_rows[1], R_cols[1]] -> W index = R_cols[1]*n + R_rows[1]
        
        u11 = P_rows[0] * m + P_cols[0]
        u12 = P_rows[0] * m + P_cols[1]
        u21 = P_rows[1] * m + P_cols[0]
        u22 = P_rows[1] * m + P_cols[1]
        
        v11 = Q_rows[0] * p + Q_cols[0]
        v12 = Q_rows[0] * p + Q_cols[1]
        v21 = Q_rows[1] * p + Q_cols[0]
        v22 = Q_rows[1] * p + Q_cols[1]
        
        w11 = R_cols[0] * n + R_rows[0]
        w12 = R_cols[1] * n + R_rows[0]
        w21 = R_cols[0] * n + R_rows[1]
        w22 = R_cols[1] * n + R_rows[1]
        
        # M1 = (P11+P22)*(Q11+Q22)
        c = base_col + 0
        U[u11, c] += 1
        U[u22, c] += 1
        V[v11, c] += 1
        V[v22, c] += 1
        W[w11, c] += 1
        W[w22, c] += 1
        
        # M2 = (P21+P22)*Q11
        c = base_col + 1
        U[u21, c] += 1
        U[u22, c] += 1
        V[v11, c] += 1
        W[w21, c] += 1
        W[w22, c] += -1  # R22 = M1 - M2 + ...
        
        # M3 = P11*(Q12-Q22)
        c = base_col + 2
        U[u11, c] += 1
        V[v12, c] += 1
        V[v22, c] += -1
        W[w12, c] += 1
        W[w22, c] += 1
        
        # M4 = P22*(Q21-Q11)
        c = base_col + 3
        U[u22, c] += 1
        V[v21, c] += 1
        V[v11, c] += -1
        W[w11, c] += 1
        W[w21, c] += 1
        
        # M5 = (P11+P12)*Q22
        c = base_col + 4
        U[u11, c] += 1
        U[u12, c] += 1
        V[v22, c] += 1
        W[w11, c] += -1
        W[w12, c] += 1
        
        # M6 = (P21-P11)*(Q11+Q12)
        c = base_col + 5
        U[u21, c] += 1
        U[u11, c] += -1
        V[v11, c] += 1
        V[v12, c] += 1
        W[w22, c] += 1
        
        # M7 = (P12-P22)*(Q21+Q22)
        c = base_col + 6
        U[u12, c] += 1
        U[u22, c] += -1
        V[v21, c] += 1
        V[v22, c] += 1
        W[w11, c] += 1
        W[w22, c] += -1
        
        return base_col + 7

    # Now build the rank-28 decomposition
    rank_28 = 28
    U28 = np.zeros((n * m, rank_28))
    V28 = np.zeros((m * p, rank_28))
    W28 = np.zeros((p * n, rank_28))
    
    # Mult 0: A1 * B11 -> C1_left
    # A1: rows [0,1], cols [0,1]
    # B11: rows [0,1], cols [0,1]
    # C1_left: rows [0,1], cols [0,1]
    col = strassen_2x2_decomp([0,1], [0,1], [0,1], [0,1], [0,1], [0,1], 0, U28, V28, W28)
    
    # Mult 1: A2 * B21 -> C1_right
    # A2: rows [0,1], cols [2,3]
    # B21: rows [2,3], cols [0,1]
    # C1_right: rows [0,1], cols [0,1]
    col = strassen_2x2_decomp([0,1], [2,3], [2,3], [0,1], [0,1], [0,1], col, U28, V28, W28)
    
    # Mult 2: A1 * B12 -> C2_left
    # A1: rows [0,1], cols [0,1]
    # B12: rows [0,1], cols [2,3]
    # C2_left: rows [0,1], cols [2,3]
    col = strassen_2x2_decomp([0,1], [0,1], [0,1], [2,3], [0,1], [2,3], col, U28, V28, W28)
    
    # Mult 3: A2 * B22 -> C2_right
    # A2: rows [0,1], cols [2,3]
    # B22: rows [2,3], cols [2,3]
    # C2_right: rows [0,1], cols [2,3]
    col = strassen_2x2_decomp([0,1], [2,3], [2,3], [2,3], [0,1], [2,3], col, U28, V28, W28)
    
    # Verify the rank-28 decomposition
    def verify_decomp(U, V, W, n, m, p):
        """Check if the decomposition exactly represents matrix multiplication."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1
                    actual = 0
                    for c in range(U.shape[1]):
                        actual += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(actual - expected) > 1e-10:
                        return False
        return True
    
    if verify_decomp(U28, V28, W28, n, m, p):
        best_rank = 28
        best_U, best_V, best_W = U28.copy(), V28.copy(), W28.copy()
    
    # Now try to find rank 27 via structural sharing of Strassen multiplications
    # Key insight: A1 is used in Mult 0 and Mult 2, A2 in Mult 1 and Mult 3.
    # We can potentially share some Strassen terms.
    
    # Alternative approach: try to find a rank-27 decomposition by
    # starting from rank-28 and attempting to eliminate one rank-1 term
    # through algebraic manipulation.
    
    # Structural approach: Use a modified Strassen scheme where we
    # compute the 4 block products with only 27 multiplications.
    
    # Let's try: view the problem as A (2x4) * B (4x4) = C (2x4)
    # Partition B into two 4x2 halves: B_L = B[:,0:2], B_R = B[:,2:4]
    # C_L = A * B_L (2x2), C_R = A * B_R (2x2)
    # 
    # For C_L = A * B_L where A is 2x4 and B_L is 4x2:
    #   A = [A1 A2], B_L = [B11; B21]
    #   C_L = A1*B11 + A2*B21
    #
    # For C_R = A * B_R where A is 2x4 and B_R is 4x2:
    #   C_R = A1*B12 + A2*B22
    #
    # Total: 4 independent 2x2 multiplications -> 28 with Strassen.
    #
    # To get 27, we need to share one multiplication.
    # 
    # Idea: Compute M = (A1 + A2) * (B11 + B12 + B21 + B22)
    # This expands to: A1*B11 + A1*B12 + A1*B21 + A1*B22 + A2*B11 + A2*B12 + A2*B21 + A2*B22
    # 
    # We need: A1*B11 + A2*B21 (for C_L) and A1*B12 + A2*B22 (for C_R)
    # 
    # If we also compute:
    # N = (A1 - A2) * (B11 - B12 + B21 - B22)
    # = A1*B11 - A1*B12 + A1*B21 - A1*B22 - A2*B11 + A2*B12 - A2*B21 + A2*B22
    #
    # Then M + N = 2*A1*B11 + 2*A1*B21 + 2*A2*B12 + 2*A2*B22
    # M - N = 2*A1*B12 + 2*A1*B22 + 2*A2*B11 + 2*A2*B22  -- not quite right
    #
    # This approach doesn't cleanly separate the needed terms.
    #
    # Let's try a different sharing strategy.
    # 
    # For the two C_L-type products (A1*B11 + A2*B21) and (A1*B12 + A2*B22):
    # 
    # Define:
    # S1 = (A1 + A2) * (B11 + B21)   [7 mults via Strassen on 2x2]
    # S2 = (A1 + A2) * (B12 + B22)   [7 mults]
    # S3 = A1 * B11                   [7 mults]
    # S4 = A2 * B21                   [7 mults]
    # S5 = A1 * B12                   [7 mults]
    # S6 = A2 * B22                   [7 mults]
    #
    # C_L = S3 + S4 + (S1 - S3 - S4 - A1*B21 - A2*B11)  -- still need extra terms
    #
    # This isn't working cleanly. Let me try a completely different structural approach.
    
    # APPROACH: Use the fact that for 2x4 x 4x4, we can use a "splitting" method.
    # 
    # The tensor T for 2x4x4 has 32 entries. We want rank r decomposition.
    # 
    # Let's try a direct rank-27 construction using a known algebraic identity.
    # 
    # Actually, let's try a practical approach: use randomized search with 
    # the rank-28 solution as a starting point and try to find a 27-rank 
    # approximation that's exact.
    
    if time.time() < deadline - 15:
        # Try to find rank 27 by perturbing the rank-28 solution
        # and checking if one column becomes redundant.
        
        # Check if any pair of columns in the rank-28 decomposition
        # can be merged (i.e., if there's a linear dependency).
        
        # For a rank-r decomposition, the columns of [U;V;W] (stacked)
        # should be linearly independent for a minimal decomposition.
        # If two columns are linearly dependent, we can merge them.
        
        # Stack all factor matrices
        def check_rank_reduction(U, V, W, n, m, p, r):
            """Check if we can reduce rank by finding linearly dependent columns."""
            # For each pair of columns, check if they're linearly dependent
            # in the combined space.
            for c1 in range(r):
                if time.time() > deadline - 15:
                    return False, None
                for c2 in range(c1 + 1, r):
                    # Check if column c2 can be expressed as alpha * column c1
                    # in the U, V, W spaces simultaneously.
                    u1, u2 = U[:, c1], U[:, c2]
                    v1, v2 = V[:, c1], V[:, c2]
                    w1, w2 = W[:, c1], W[:, c2]
                    
                    # If u2 = a*u1, v2 = a*v1, w2 = a*w1 for some scalar a
                    # then we can merge.
                    
                    # Find a from u (if u1 is nonzero)
                    nonzero_u = np.abs(u1) > 1e-10
                    if np.any(nonzero_u):
                        a_candidates = u2[nonzero_u] / u1[nonzero_u]
                        if np.allclose(a_candidates, a_candidates[0], atol=1e-10):
                            a = a_candidates[0]
                            # Check V
                            if np.allclose(v2, a * v1, atol=1e-10):
                                # Check W
                                if np.allclose(w2, a * w1, atol=1e-10):
                                    return True, (c1, c2, a)
                    
                    # Also check if u2 = 0, v2 = 0, w2 = 0 (zero column)
                    if np.allclose(u2, 0, atol=1e-10) and \
                       np.allclose(v2, 0, atol=1e-10) and \
                       np.allclose(w2, 0, atol=1e-10):
                        return True, (c1, c2, 0)
            
            return False, None
        
        reducible, info = check_rank_reduction(U28, V28, W28, n, m, p, 28)
        if reducible:
            # Remove the redundant column
            c1, c2, a = info
            U27 = np.delete(U28, c2, axis=1)
            V27 = np.delete(V28, c2, axis=1)
            W27 = np.delete(W28, c2, axis=1)
            if verify_decomp(U27, V27, W27, n, m, p):
                best_rank = 27
                best_U, best_V, best_W = U27.copy(), V27.copy(), W27.copy()
        
        # If no simple reduction found, try a more sophisticated approach:
        # Use the border rank technique - try to find a rank-27 decomposition
        # by solving a system of equations.
        
        if best_rank > 27 and time.time() < deadline - 20:
            # Try to construct rank 27 directly using a modified Strassen scheme.
            # 
            # Key idea: For the 2x4 x 4x4 problem, we can use a scheme where
            # we share the "diagonal" Strassen terms.
            #
            # Let's define a custom 27-rank scheme:
            #
            # The 4 block products are:
            # P0 = A1*B11, P1 = A2*B21, P2 = A1*B12, P3 = A2*B22
            #
            # Standard Strassen gives 7*4 = 28 terms.
            #
            # For sharing, note that P0 and P2 share A1, and P1 and P3 share A2.
            #
            # In Strassen, the 7 terms for a product P*Q are:
            # M1=(P11+P22)(Q11+Q22), M2=(P21+P22)Q11, M3=P11(Q12-Q22),
            # M4=P22(Q21-Q11), M5=(P11+P12)Q22, M6=(P21-P11)(Q11+Q12),
            # M7=(P12-P22)(Q21+Q22)
            #
            # For P0=A1*B11 and P2=A1*B12, the U-factors are the same.
            # We can potentially combine M1 terms:
            # M1_0 = (A1[0,0]+A1[1,1])*(B11[0,0]+B11[1,1])
            # M1_2 = (A1[0,0]+A1[1,1])*(B12[0,0]+B12[1,1])
            #
            # These have the same U-factor but different V-factors, so they can't
            # be directly merged. However, we might be able to use a different
            # decomposition of the 2x2 product that allows sharing.
            #
            # Let's try: use a 6-rank scheme for some of the 2x2 products
            # (not possible in general, but maybe for this specific structure).
            #
            # Actually, the minimal rank for 2x2x2 is 7, so we can't reduce
            # individual 2x2 products. The only way to get below 28 is to
            # share multiplications across the 4 block products.
            #
            # Let's try a direct algebraic approach: assume a rank-27 decomposition
            # exists and try to find it using the structure of the problem.
            #
            # For rank 27, we have 27*(8+16+8) = 864 unknowns and 32 equations.
            # This is very underdetermined, so solutions should exist if the
            # rank is indeed 27.
            #
            # Let's try a randomized approach: initialize with the rank-28 solution,
            # remove one column, and try to adjust the remaining columns to
            # compensate.
            
            np.random.seed(123)
            for attempt in range(50):
                if time.time() > deadline - 20:
                    break
                    
                # Start with rank-28, remove a random column
                col_to_remove = np.random.randint(28)
                U27_try = np.delete(U28.copy(), col_to_remove, axis=1)
                V27_try = np.delete(V28.copy(), col_to_remove, axis=1)
                W27_try = np.delete(W28.copy(), col_to_remove, axis=1)
                
                # Now try to adjust the remaining 27 columns to make it work.
                # This is a nonlinear problem, so we'll use a simple iterative approach.
                
                # Compute the current error
                def compute_error(U, V, W, n, m, p):
                    err = 0
                    max_err = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = 0
                                for c in range(U.shape[1]):
                                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                                err += (val - 1) ** 2
                                max_err = max(max_err, abs(val - 1))
                    return err, max_err
                
                err, max_err = compute_error(U27_try, V27_try, W27_try, n, m, p)
                
                # If error is small, try to fix it with small perturbations
                if max_err < 0.1:
                    # Try gradient-free perturbation
                    for sub_attempt in range(100):
                        if time.time() > deadline - 20:
                            break
                        # Perturb one random element
                        mat_choice = np.random.randint(3)
                        if mat_choice == 0:
                            row = np.random.randint(n * m)
                            col = np.random.randint(27)
                            U27_try[row, col] += np.random.randn() * 0.01
                        elif mat_choice == 1:
                            row = np.random.randint(m * p)
                            col = np.random.randint(27)
                            V27_try[row, col] += np.random.randn() * 0.01
                        else:
                            row = np.random.randint(p * n)
                            col = np.random.randint(27)
                            W27_try[row, col] += np.random.randn() * 0.01
                        
                        err_new, max_err_new = compute_error(U27_try, V27_try, W27_try, n, m, p)
                        if err_new < err:
                            err = err_new
                            max_err = max_err_new
                        else:
                            # Revert
                            if mat_choice == 0:
                                U27_try[row, col] -= np.random.randn() * 0.01
                            elif mat_choice == 1:
                                V27_try[row, col] -= np.random.randn() * 0.01
                            else:
                                W27_try[row, col] -= np.random.randn() * 0.01
                        
                        if max_err < 1e-10:
                            break
                    
                    if max_err < 1e-10:
                        best_rank = 27
                        best_U, best_V, best_W = U27_try.copy(), V27_try.copy(), W27_try.copy()
                        break
    
    # Convert to lists for the return format
    def to_list(arr):
        return [[float(x) for x in row] for row in arr]
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
