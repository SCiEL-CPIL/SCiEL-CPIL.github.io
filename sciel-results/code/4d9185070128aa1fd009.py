"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication.
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 145  # 5s safety margin
    
    def validate_decomposition(U_mat, V_mat, W_mat, rank_val):
        """Check if the decomposition exactly computes matrix multiplication."""
        U_arr = np.array(U_mat, dtype=float)
        V_arr = np.array(V_mat, dtype=float)
        W_arr = np.array(W_mat, dtype=float)
        
        # For all (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_arr[i * m + j, :], V_arr[j * p + k, :] * W_arr[k * n + i, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    def validate_polynomial_identity(U_mat, V_mat, W_mat, rank_val, num_trials=10):
        """
        Schwartz-Zippel style polynomial identity test.
        Verify that the tensor identity holds by evaluating at random points.
        """
        U_arr = np.array(U_mat, dtype=float)
        V_arr = np.array(V_mat, dtype=float)
        W_arr = np.array(W_mat, dtype=float)
        
        for _ in range(num_trials):
            # Random matrices A (n x m) and B (m x p)
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            
            # Compute C = A @ B
            C = A @ B
            
            # Compute using decomposition: C[i,k] = sum_c (sum_j U[i*m+j,c]*A[i,j]) * (sum_j V[j*p+k,c]*B[j,k]) * W[k*n+i,c]
            # Actually: C[i,k] = sum_c U[i,m,j] * A[i,j] * V[j,p,k] * B[j,k] * W[k,n,i]
            # More precisely: for each c, compute intermediate, then sum
            C_decomp = np.zeros((n, p))
            for c in range(rank_val):
                # Compute u_vec[i] = sum_j U[i*m+j, c] * A[i, j]
                u_vec = np.zeros(n)
                for i in range(n):
                    for j in range(m):
                        u_vec[i] += U_arr[i * m + j, c] * A[i, j]
                
                # Compute v_vec[k] = sum_j V[j*p+k, c] * B[j, k]
                v_vec = np.zeros(p)
                for k in range(p):
                    for j in range(m):
                        v_vec[k] += V_arr[j * p + k, c] * B[j, k]
                
                # C[i,k] += u_vec[i] * v_vec[k] * W[k*n+i, c]
                for i in range(n):
                    for k in range(p):
                        C_decomp[i, k] += u_vec[i] * v_vec[k] * W_arr[k * n + i, c]
            
            if not np.allclose(C, C_decomp, atol=1e-9):
                return False
        return True
    
    # Start with schoolbook decomposition
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
    
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    # Verify schoolbook works
    assert validate_decomposition(U, V, W, rank), "Schoolbook decomposition failed validation"
    
    # Strategy: Use Strassen-like algebraic identities to reduce rank
    # For 3x3 * 3x8, we can view this as 8 separate 3x3 * 3x1 multiplications
    # Each 3x3 * 3x1 has rank at least 9, but we can use algebraic tricks
    
    # Key insight: Use the fact that for 3x3 matrices, we can use
    # algebraic identities to reduce the number of multiplications
    
    # Approach 1: Use shared computations across the 8 columns
    # Group columns into pairs and use Strassen-like formulas
    
    # For 3x3 * 3x1, the trivial rank is 9.
    # Can we do better? The lower bound is max(9,3,3) = 9.
    # So 9 is optimal for a single 3x3*3x1.
    
    # But for 3x3 * 3x8, we have 8 such products.
    # If we can share computations, we might reduce total rank.
    
    # Let's try to construct a decomposition with rank < 72
    
    # Method: Use linear combinations of A rows and B columns
    # Define new variables that are linear combinations
    
    # For the 3x3 * 3x8 case, let's try to use the structure
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # We can define:
    # M1 = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]  (but this is per k)
    
    # Better: Use the fact that we can precompute linear combinations
    # of A and B that are shared across k
    
    # Let's try a different approach: use the tensor rank of the
    # matrix multiplication tensor for 3x3x8
    
    # The tensor is: T[i,j,k] = e_{i,j} * f_{j,k} * g_{k,i}
    # where e_{i,j} extracts A[i,j], f_{j,k} extracts B[j,k], g_{k,i} places into C[i,k]
    
    # We want to decompose T as sum_c u_c * v_c * w_c
    
    # For now, let's try to find a rank-64 or lower decomposition
    # using structured search
    
    # Value set for entries
    value_set = [0, 1, -1, 0.5, -0.5, 2, -2, 0.25, -0.25]
    
    # Try structured decompositions based on algebraic identities
    
    # Identity 1: For 3x3 * 3x1, use the following 9-term decomposition
    # This is just the schoolbook, but let's see if we can do better
    # for the 3x3 * 3x8 case by sharing
    
    # Actually, let's try to use the fact that the 8 columns of B
    # can be grouped, and we can use shared intermediate computations
    
    # For example, if we define:
    # S1 = A[0,0] + A[0,1] + A[0,2]
    # S2 = A[1,0] + A[1,1] + A[1,2]  
    # S3 = A[2,0] + A[2,1] + A[2,2]
    
    # And for B:
    # T1_k = B[0,k] + B[1,k] + B[2,k]
    # T2_k = B[0,k] - B[1,k] + B[2,k]
    # etc.
    
    # This is getting complex. Let's try a more systematic approach.
    
    # Use random search with polynomial identity verification
    # Focus on ranks that are achievable
    
    # Lower bound: max(n*m, m*p, p*n) = max(9, 24, 24) = 24
    # But realistically, we need at least 9 per "column group"
    
    # Let's try ranks from 48 to 71 (multiples of 6 or 8)
    candidate_ranks = [48, 52, 56, 60, 64, 68]
    
    for target_rank in candidate_ranks:
        if time.time() > deadline - 10:
            break
        if target_rank >= best_rank:
            continue
        
        # Try structured random search
        max_attempts = 50
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            # Use a smarter initialization: start with schoolbook and perturb
            U_rand = [row[:] for row in U]
            V_rand = [row[:] for row in V]
            W_rand = [row[:] for row in W]
            
            # Truncate to target_rank
            if target_rank < rank:
                U_rand = [row[:target_rank] for row in U_rand]
                V_rand = [row[:target_rank] for row in V_rand]
                W_rand = [row[:target_rank] for row in W_rand]
            
            # Apply random perturbations from value_set
            for i in range(n * m):
                for c in range(target_rank):
                    if random.random() < 0.3:
                        U_rand[i][c] = random.choice(value_set)
            
            for i in range(m * p):
                for c in range(target_rank):
                    if random.random() < 0.3:
                        V_rand[i][c] = random.choice(value_set)
            
            for i in range(p * n):
                for c in range(target_rank):
                    if random.random() < 0.3:
                        W_rand[i][c] = random.choice(value_set)
            
            # Quick check with polynomial identity test
            if validate_polynomial_identity(U_rand, V_rand, W_rand, target_rank, num_trials=3):
                # Verify exactly
                if validate_decomposition(U_rand, V_rand, W_rand, target_rank):
                    best_rank = target_rank
                    best_U, best_V, best_W = U_rand, V_rand, W_rand
                    break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
