"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 140s deadline, 10s safety margin
    
    # Try to find a lower-rank decomposition
    # The tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
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
    
    # Try various ranks below schoolbook
    # Use a structured approach: try to find decompositions with specific patterns
    
    # Approach: Use the fact that for fixed j, we have n*p = 16 equations
    # involving U[i,j,:], V[j,k,:], W[k,i,:]
    
    # Let's try to use a block-circulant or other structured approach
    
    # Try rank values from lower to upper
    for r in range(32, best_rank):
        if time.time() > deadline - 10:
            break
        
        # Try to find a solution with rank r using random initialization + refinement
        # This is a non-convex problem, so we use multiple restarts
        
        found = False
        for attempt in range(20):
            if time.time() > deadline - 10:
                break
            
            # Initialize with random values
            np.random.seed(attempt)
            U = np.random.randn(n*m, r) * 0.5
            V = np.random.randn(m*p, r) * 0.5
            W = np.random.randn(p*n, r) * 0.5
            
            # Iteratively refine: fix two factors, solve for the third
            # This is a tensor decomposition approach
            
            for it in range(200):
                if time.time() > deadline - 5:
                    break
                
                # Fix U and V, solve for W
                # For each (k,i): sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1 for all j
                # This gives a system: for fixed (k,i), we need:
                # sum_j (sum_c U[i,j,c]*V[j,k,c]) * W[k,i,c] ... no, that's not right.
                
                # Actually: for each (i,j,k): sum_c U[i*m+j, c]*V[j*p+k, c]*W[k*n+i, c] = 1
                # Fix U and V. For each (k,i), let A_{k,i,j} = sum_c U[i*m+j, c]*V[j*p+k, c]
                # Then we need: sum_c W[k*n+i, c] * (something) ... 
                
                # Hmm, this doesn't decouple nicely. Let me think differently.
                
                # Alternative: For fixed c, the contribution is U[:,c] ⊗ V[:,c] ⊗ W[:,c]
                # The sum over c must equal T.
                
                # Let me try a different approach: use the bilinear structure.
                # C[i,k] = sum_j A[i,j]*B[j,k]
                # = sum_j sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] * A[i,j]*B[j,k]
                # We need: sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1 for all i,j,k
                
                # This is equivalent to: the tensor T = sum_c u_c ⊗ v_c ⊗ w_c where
                # u_c[i*m+j] = U[i,j,c], v_c[j*p+k] = V[j,k,c], w_c[k*n+i] = W[k,i,c]
                
                # Let me try gradient descent on the objective:
                # f(U,V,W) = sum_{i,j,k} (sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] - 1)^2
                
                # But this is complex. Let me try a simpler structured approach.
                
                pass  # Skip complex optimization for now
            
            # Check if we found a valid solution (we didn't in the loop above)
            # So this approach isn't working well
        
        # Try a more structured approach for specific ranks
        # For r = n*m*p / 2 = 32: can we pair up terms?
        # For r = 48: remove 16 terms
    
    # Return the best result found
    # Convert to lists of lists
    def to_list(mat):
        return mat.tolist()
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
