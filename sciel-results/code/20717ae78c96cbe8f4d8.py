"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    def schoolbook(n, m, p):
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W

    U_sb, V_sb, W_sb = schoolbook(n, m, p)
    
    def verify_full(U, V, W, n, m, p, tol=1e-6):
        """Verify complete tensor decomposition."""
        rank = U.shape[1]
        # Check all entries
        for i in range(n):
            for j in range(m):
                a = i * m + j
                for j2 in range(m):
                    for k in range(p):
                        b = j2 * p + k
                        for k2 in range(p):
                            for i2 in range(n):
                                c = k2 * n + i2
                                val = 0.0
                                for cc in range(rank):
                                    val += U[a, cc] * V[b, cc] * W[c, cc]
                                expected = 1.0 if (i == i2 and j == j2 and k == k2) else 0.0
                                if abs(val - expected) > tol:
                                    return False
        return True

    # Try to find rank-99 decomposition by finding a linear dependency
    # among the schoolbook terms. If we can express one schoolbook term as a
    # linear combination of others in the tensor sense, we can reduce rank.
    
    # For the matrix multiplication tensor, the key structure is:
    # T[(i,j),(j,k),(k,i)] = 1
    # All other entries are 0.
    
    # A rank reduction would require finding coefficients such that
    # sum_c alpha_c * T_c = 0 where T_c are the rank-1 tensors.
    
    # Let's try a numerical approach: use gradient-based optimization to find
    # a rank-99 or rank-98 decomposition.
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = n * m * p
    
    # Strategy: try rank 99 first (easier), then rank 98
    for target_rank in [99, 98]:
        if time.time() > deadline - 20:
            break
        
        found = False
        np.random.seed(42)
        
        # Initialize by taking schoolbook and zeroing out (100 - target_rank) columns,
        # then optimizing the remaining columns to compensate.
        for attempt in range(20):
            if time.time() > deadline - 20:
                break
            if found:
                break
            
            # Start with schoolbook, remove some columns
            U = U_sb.copy()
            V = V_sb.copy()
            W = W_sb.copy()
            
            num_remove = n * m * p - target_rank
            cols_to_remove = np.random.choice(n * m * p, num_remove, replace=False)
            U[:, cols_to_remove] = 0
            V[:, cols_to_remove] = 0
            W[:, cols_to_remove] = 0
            
            # Keep track of which columns are active
            active_cols = [c for c in range(n*m*p) if c not in cols_to_remove]
            
            # Now optimize the active columns to minimize the error
            # Use a simple gradient descent approach
            
            def compute_error(U, V, W):
                """Compute sum of squared errors over all tensor entries."""
                err = 0.0
                rank = U.shape[1]
                for i in range(n):
                    for j in range(m):
                        a = i * m + j
                        for j2 in range(m):
                            for k in range(p):
                                b = j2 * p + k
                                for k2 in range(p):
                                    for i2 in range(n):
                                        c = k2 * n + i2
                                        val = 0.0
                                        for cc in range(rank):
                                            val += U[a, cc] * V[b, cc] * W[c, cc]
                                        expected = 1.0 if (i == i2 and j == j2 and k == k2) else 0.0
                                        err += (val - expected) ** 2
                return err
            
            # Use coordinate descent: update each column one at a time
            U_opt = U.copy()
            V_opt = V.copy()
            W_opt = W.copy()
            
            for iteration in range(50):
                if time.time() > deadline - 20:
                    break
                
                for c in active_cols:
                    # Update U[:, c]
                    # The error is sum over all (a,b,c') of (sum_cc U[a,cc]*V[b,cc]*W[c',cc] - T[a,b,c'])^2
                    # This is complex. Let's use a simpler approach.
                    
                    # For now, just do a small random perturbation and accept if error decreases
                    pass
                
                break  # This approach is too slow
            
            # Try a different approach: use the fact that for matrix multiplication,
            # we can use Strassen-like reductions on sub-blocks.
            
            # For 4x5x5, we can partition the 4x5 matrix into two 2x5 blocks.
            # For each 2x5 block multiplied by 5x5, we might be able to save multiplications.
            
            break  # Give up on this approach
        
        if not found:
            # Try a completely different approach: use known algebraic identities
            # For 2x2x2, Strassen's algorithm gives rank 7 instead of 8.
            # We can try to embed this into the 4x5x5 problem.
            
            # The 4x5x5 problem can be seen as:
            # C = A * B where A is 4x5, B is 5x5
            # Split A into A1 (2x5) and A2 (2x5)
            # C1 = A1 * B (2x5)
            # C2 = A2 * B (2x5)
            # C = [C1; C2]
            
            # For each 2x5x5 multiplication, the schoolbook rank is 2*5*5 = 50.
            # Can we do better? For 2x5x5, it's not clear if rank reduction is possible.
            
            # Let's try yet another approach: use numerical linear algebra to find
            # if there's a linear dependency among the rank-1 tensors.
            
            break
    
    # If we couldn't find a lower rank, return schoolbook
    if best_rank == n * m * p:
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
