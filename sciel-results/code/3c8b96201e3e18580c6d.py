"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 138  # 138s deadline, 12s safety margin
    
    # The matrix multiplication tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Start with schoolbook as fallback
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    # Try to find lower-rank decompositions using structured approaches
    # Key insight: For fixed j, the equations are:
    # For each (i,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # 
    # Let's try rank r where we can exploit the structure.
    # One approach: Use the fact that we can write 1 = sum of products.
    # 
    # Try a block-structured approach:
    # For each j, define vectors u_j[i] = U[i*m+j, :], v_j[k] = V[j*p+k, :], w_{k,i} = W[k*n+i, :]
    # We need: sum_c u_j[i,c] * v_j[k,c] * w_{k,i,c} = 1 for all i,k
    
    # A simple structured decomposition:
    # Let r = n * p + m * p + n * m (or similar)
    # Actually, let's try to use the identity: 1 = 1*1*1
    # and try to share terms across different (i,j,k) triples.
    
    # Approach 1: Try rank = n*m*p - 1 and see if we can merge two terms
    # This is unlikely to work for arbitrary tensors, but let's try structured ranks.
    
    # Approach 2: Use a bilinear algorithm structure
    # For matrix multiplication, there are known lower bounds and constructions.
    # For 2x4 * 4x8, the schoolbook rank is 64.
    # Can we do better? Let's try ranks from 32 to 63.
    
    # Let's try a specific structured approach:
    # Split the computation into blocks and use sharing.
    
    # For now, let's try to find a valid decomposition with rank < 64 using
    # a systematic search with random initialization and alternating minimization.
    
    def check_decomposition(U, V, W, tol=1e-10):
        """Check if the decomposition is exact."""
        # Compute sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for all i,j,k
        # and check if it equals 1
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(U.shape[1]):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    err = max(err, abs(val - 1.0))
        return err < tol
    
    def try_rank(r, num_attempts=5, iters_per_attempt=50):
        """Try to find a decomposition with rank r."""
        nonlocal best_U, best_V, best_W, best_rank
        
        for attempt in range(num_attempts):
            if time.time() > deadline - 15:
                return False
            
            np.random.seed(42 + attempt)
            
            # Initialize with small random values
            U = np.random.randn(n*m, r) * 0.1
            V = np.random.randn(m*p, r) * 0.1
            W = np.random.randn(p*n, r) * 0.1
            
            # Alternating minimization: fix two, solve for third
            # This is a non-convex problem, so we use multiple restarts
            
            for it in range(iters_per_attempt):
                if time.time() > deadline - 15:
                    return False
                
                # Fix U and V, solve for W
                # For each (k,i): we need sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * V[j*p+k, c] * A[i,j] * B[j,k]) = C[i,k]
                # But we're solving for the tensor, not actual matrices.
                # 
                # Let's use a different approach: gradient descent on the objective
                # f = sum_{i,j,k} (sum_c U[i*m+j, c]*V[j*p+k, c]*W[k*n+i, c] - 1)^2
                
                # Compute gradient with respect to W
                # df/dW[k*n+i, c] = 2 * (sum_c' U[i*m+j, c']*V[j*p+k, c']*W[k*n+i, c'] - 1) * U[i*m+j, c] * V[j*p+k, c]
                # But this sums over j for each (i,k,c)
                
                # Actually, let's just do a simple iterative update.
                # This is getting complex. Let's try a different approach.
                
                # Use the fact that for fixed (i,j), the equation is linear in W:
                # sum_c W[k*n+i, c] * (U[i*m+j, c] * V[j*p+k, c]) = 1 for all k
                # This is a system of p equations in r unknowns for each (i,j).
                # If r >= p, we can solve this.
                
                pass  # Skip for now, too complex
        
        return False
    
    # Let's try a more direct approach: construct a specific low-rank decomposition
    # using the structure of the problem.
    
    # Key idea: For the tensor T[i,j,k] = 1, we can write:
    # T = sum_{a=1}^{n} sum_{b=1}^{p} (e_a ⊗ e_b) ⊗ (something)
    # But this doesn't directly help.
    
    # Alternative: Use the identity 1 = 1, and try to share the "1" across different (i,j,k).
    # For example, if we set U[i*m+j, c] = 1 for all i,j and some c,
    # V[j*p+k, c] = 1 for all j,k and the same c,
    # W[k*n+i, c] = 1 for all k,i and the same c,
    # then we get 1 for all (i,j,k) with just rank 1!
    # But wait, that would mean U[i*m+j, c] = 1 for ALL i,j, which means U has all ones in column c.
    # Similarly for V and W. Then sum_c U[i*m+j, c]*V[j*p+k, c]*W[k*n+i, c] = 1*1*1 = 1.
    # This works! Rank 1!
    
    # Let's verify: U[i*m+j, 0] = 1 for all i in [0,n), j in [0,m)
    # V[j*p+k, 0] = 1 for all j in [0,m), k in [0,p)
    # W[k*n+i, 0] = 1 for all k in [0,p), i in [0,n)
    # Then sum_c U[i*m+j, c]*V[j*p+k, c]*W[k*n+i, c] = U[i*m+j, 0]*V[j*p+k, 0]*W[k*n+i, 0] = 1*1*1 = 1.
    # This is valid!
    
    # But wait, the problem says "exact tensor decomposition". Let me re-read the task.
    # "Find a rank-minimal exact tensor decomposition for multiplying a 2 by 4 matrix by a 4 by 8 matrix."
    # The output factor follows the official trace convention W[k*n+i,r].
    # 
    # So the tensor is T[i,j,k] = 1, and we need sum_c U[i*m+j, c]*V[j*p+k, c]*W[k*n+i, c] = 1.
    # The rank-1 solution above works!
    
    U_rank1 = np.zeros((n*m, 1))
    V_rank1 = np.zeros((m*p, 1))
    W_rank1 = np.zeros((p*n, 1))
    U_rank1[:, 0] = 1
    V_rank1[:, 0] = 1
    W_rank1[:, 0] = 1
    
    # Verify
    if check_decomposition(U_rank1, V_rank1, W_rank1):
        best_U, best_V, best_W = U_rank1, V_rank1, W_rank1
        best_rank = 1
    
    # Convert to lists of lists
    def to_list(mat):
        return mat.tolist()
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
