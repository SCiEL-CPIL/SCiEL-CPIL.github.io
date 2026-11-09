"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]
DEADLINE = time.time() + 145  # Safety margin before 150s kill

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # C[i,k] = sum_j A[i,j]*B[j,k]
                # In tensor form: T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
                # We need T[i,j,k] = 1 if j==j (always), so T should be the identity in j for each (i,k)
                # Actually T[i,j,k] should equal 1 when we're computing C[i,k] = sum_j A[i,j]B[j,k]
                # The tensor T[i,j,k] = delta_{j,j} = 1 for all valid (i,j,k)
                # Wait, let me reconsider. The matrix multiplication tensor is:
                # T[i,j,k] = 1 for all i,j,k (it's the full tensor of ones in the "multiplication" sense)
                # No, that's not right either.
                # 
                # Matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
                # As a tensor: T[i,j,k] where T[i,j,k] = 1 for all valid indices
                # The decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
                val = 0
                for r in range(rank):
                    val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if val != 1:
                    return False
    return True

def _try_free_term(U, V, W, n, m, p, term_idx):
    """
    Try to free (eliminate) term term_idx from the decomposition.
    Returns (new_U, new_V, new_W, new_rank) if successful, else None.
    """
    rank = len(U[0])
    
    # Collect all (i,j,k) triples affected by this term
    # A term contributes to T[i,j,k] where U[i*m+j,r] != 0, V[j*p+k,r] != 0, W[k*n+i,r] != 0
    affected = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U[i*m+j][term_idx]
                v_val = V[j*p+k][term_idx]
                w_val = W[k*n+i][term_idx]
                if u_val != 0 and v_val != 0 and w_val != 0:
                    affected.add((i, j, k))
    
    if not affected:
        return None
    
    # For each affected (i,j,k), we need to add new terms to compensate
    # Strategy: for each affected (i,j,k), we can add a new term that only affects that entry
    # But a single new term affects multiple entries. We need to be clever.
    #
    # Simpler approach: Add one new term per affected (i,j,k) that exactly compensates
    # New term for (i,j,k): U'[i*m+j, new] = 1, V'[j*p+k, new] = 1, W'[k*n+i, new] = 1
    # But this adds |affected| new terms, removing 1, so net change is |affected|-1
    # This only helps if |affected| < 1, which is impossible.
    #
    # Better: Try to find a set of new terms that collectively compensate all affected entries
    # while using fewer than |affected| new terms.
    #
    # For now, let's just try the naive approach and see if we can do better with optimization.
    # Actually, the key insight is that we don't need to add a term for EVERY affected entry.
    # If two affected entries share the same i or same k, we might be able to combine.
    
    # Let's try a different approach: use linear algebra
    # We want to find new terms (U_new, V_new, W_new) such that:
    # sum_r' U_new[i*m+j, r'] * V_new[j*p+k, r'] * W_new[k*n+i, r'] = -1 for all (i,j,k) in affected
    # and 0 for all other (i,j,k)
    #
    # This is a bilinear/trilinear system. Let's try small ranks for the new terms.
    
    # Try adding 1 new term that compensates all affected entries
    # This requires: U_new[i*m+j] * V_new[j*p+k] * W_new[k*n+i] = -1 for all (i,j,k) in affected
    # and 0 otherwise. This is generally impossible unless affected has special structure.
    
    # Try adding 2 new terms, etc.
    # For efficiency, let's limit the number of new terms we try.
    
    max_new_terms = min(len(affected), 5)
    
    for num_new in range(1, max_new_terms + 1):
        if time.time() > DEADLINE - 5:
            return None
        
        # Try random initializations for the new terms
        for trial in range(20):
            if time.time() > DEADLINE - 5:
                return None
            
            # Initialize new terms with small random values
            U_new = np.zeros((n*m, num_new))
            V_new = np.zeros((m*p, num_new))
            W_new = np.zeros((p*n, num_new))
            
            # Set up the system: for each affected (i,j,k), we need compensation of -1
            # For non-affected (i,j,k), we need compensation of 0 (but they're already 0 from removed term)
            
            # Use gradient descent or direct solve
            # The equations are: sum_{r'=0}^{num_new-1} U_new[i*m+j, r'] * V_new[j*p+k, r'] * W_new[k*n+i, r'] = target[i,j,k]
            # where target is -1 for affected, 0 otherwise
            
            # This is a trilinear system. Let's try to solve it iteratively.
            # Fix V and W, solve for U, then fix U and W, solve for V, etc.
            
            # Initialize
            for r in range(num_new):
                for i in range(n):
                    for j in range(m):
                        U_new[i*m+j, r] = np.random.uniform(-1, 1)
                for j in range(m):
                    for k in range(p):
                        V_new[j*p+k, r] = np.random.uniform(-1, 1)
                for k in range(p):
                    for i in range(n):
                        W_new[k*n+i, r] = np.random.uniform(-1, 1)
            
            # Iterative refinement
            for it in range(50):
                if time.time() > DEADLINE - 5:
                    break
                
                # Update U: for each (i,j), solve for U_new[i*m+j, :]
                # The equation for (i,j,k) is: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = target
                # Group by (i,j): for fixed (i,j), we have p equations (one for each k)
                # sum_r U[i*m+j,r] * (V[j*p+k,r] * W[k*n+i,r]) = target[i,j,k]
                # This is a linear system in U[i*m+j, :]
                
                # Build the system for all (i,j)
                # For each (i,j), we have a p x num_new matrix M[i,j] where M[i,j][k, r] = V[j*p+k,r] * W[k*n+i,r]
                # And we need M[i,j] @ U[i*m+j, :] = target_vec[i,j]
                
                # Let's do this for all (i,j) simultaneously
                # Actually, let's just do one (i,j) at a time for simplicity
                
                for i in range(n):
                    for j in range(m):
                        # Build p x num_new matrix
                        M = np.zeros((p, num_new))
                        target_vec = np.zeros(p)
                        for k in range(p):
                            idx = (i, j, k)
                            if idx in affected:
                                target_vec[k] = -1
                            else:
                                target_vec[k] = 0
                            for r in range(num_new):
                                M[k, r] = V_new[j*p+k, r] * W_new[k*n+i, r]
                        
                        # Solve M @ u = target_vec
                        # Use least squares
                        try:
                            u_sol, _, _, _ = np.linalg.lstsq(M, target_vec, rcond=None)
                        except:
                            continue
                        U_new[i*m+j, :] = u_sol
                
                # Update V similarly
                for j in range(m):
                    for k in range(p):
                        M = np.zeros((n, num_new))
                        target_vec = np.zeros(n)
                        for i in range(n):
                            idx = (i, j, k)
                            if idx in affected:
                                target_vec[i] = -1
                            else:
                                target_vec[i] = 0
                            for r in range(num_new):
                                M[i, r] = U_new[i*m+j, r] * W_new[k*n+i, r]
                        
                        try:
                            v_sol, _, _, _ = np.linalg.lstsq(M, target_vec, rcond=None)
                        except:
                            continue
                        V_new[j*p+k, :] = v_sol
                
                # Update W similarly
                for k in range(p):
                    for i in range(n):
                        M = np.zeros((m, num_new))
                        target_vec = np.zeros(m)
                        for j in range(m):
                            idx = (i, j, k)
                            if idx in affected:
                                target_vec[j] = -1
                            else:
                                target_vec[j] = 0
                            for r in range(num_new):
                                M[j, r] = U_new[i*m+j, r] * V_new[j*p+k, r]
                        
                        try:
                            w_sol, _, _, _ = np.linalg.lstsq(M, target_vec, rcond=None)
                        except:
                            continue
                        W_new[k*n+i, :] = w_sol
            
            # Check if we achieved the target
            success = True
            for (i, j, k) in affected:
                val = 0
                for r in range(num_new):
                    val += U_new[i*m+j, r] * V_new[j*p+k, r] * W_new[k*n+i, r]
                if abs(val - (-1)) > 1e-8:
                    success = False
                    break
            
            if success:
                # Also check that non-affected entries remain 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            if (i, j, k) not in affected:
                                val = 0
                                for r in range(num_new):
                                    val += U_new[i*m+j, r] * V_new[j*p+k, r] * W_new[k*n+i, r]
                                if abs(val) > 1e-8:
                                    success = False
                                    break
                        if not success:
                            break
                    if not success:
                        break
            
            if success:
                # Construct new matrices
                new_rank = rank - 1 + num_new
                new_U = [[0] * new_rank for _ in range(n*m)]
                new_V = [[0] * new_rank for _ in range(m*p)]
                new_W = [[0] * new_rank for _ in range(p*n)]
                
                # Copy old terms except term_idx
                col = 0
                for r in range(rank):
                    if r == term_idx:
                        continue
                    for i in range(n*m):
                        new_U[i][col] = U[i][r]
                    for j in range(m*p):
                        new_V[j][col] = V[j][r]
                    for k in range(p*n):
                        new_W[k][col] = W[k][r]
                    col += 1
                
                # Add new terms
                for r in range(num_new):
                    for i in range(n*m):
                        new_U[i][col] = float(U_new[i, r])
                    for j in range(m*p):
                        new_V[j][col] = float(V_new[j, r])
                    for k in range(p*n):
                        new_W[k][col] = float(W_new[k, r])
                    col += 1
                
                return (new_U, new_V, new_W, new_rank)
    
    return None

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Start with schoolbook decomposition
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
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank
    
    # Try to free each term
    for term_idx in range(rank):
        if time.time() > DEADLINE - 10:
            break
        
        result = _try_free_term(U, V, W, n, m, p, term_idx)
        if result is not None:
            new_U, new_V, new_W, new_rank = result
            if new_rank < best_rank:
                best_U, best_V, best_W = new_U, new_V, new_W
                best_rank = new_rank
                # Update U, V, W for next iterations
                U, V, W = new_U, new_V, new_W
                rank = new_rank
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
