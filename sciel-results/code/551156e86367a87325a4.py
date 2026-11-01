"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix by a 4x6 matrix.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill
    
    def schoolbook():
        r = n * m * p
        U = np.zeros((n, m, r), dtype=int)
        V = np.zeros((m, p, r), dtype=int)
        W = np.zeros((p, n, r), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i, j, col] = 1
                    V[j, k, col] = 1
                    W[k, i, col] = 1
                    col += 1
        # Reshape to (n*m, r), (m*p, r), (p*n, r) for output
        U_out = U.reshape(n * m, r)
        V_out = V.reshape(m * p, r)
        W_out = W.reshape(p * n, r)
        return U_out, V_out, W_out, r
    
    U_sb, V_sb, W_sb, r_sb = schoolbook()
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = r_sb
    
    # Vectorized verification
    def verify(U, V, W, r):
        """
        U: (n*m, r)
        V: (m*p, r)
        W: (p*n, r)
        """
        # Reshape to tensor form
        U_t = U.reshape(n, m, r)
        V_t = V.reshape(m, p, r)
        W_t = W.reshape(p, n, r)
        
        # Compute tensor product sum
        # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        T = np.einsum('ijr,jkr,kin->ijk', U_t, V_t, W_t)
        
        # Check if all entries are 1
        return np.allclose(T, 1.0, atol=1e-9)

    # Search for lower rank solutions
    # Known lower bound is max(nm, mp, pn) = 24.
    # Let's search from 32 down to 25.
    # Use values -2, -1, 0, 1, 2 to allow for more complex decompositions.
    search_vals = [-2, -1, 0, 1, 2]
    
    for target_rank in range(32, 24, -1):
        if time.time() > deadline:
            break
        
        found = False
        # Number of attempts depends on time remaining
        # Estimate time per attempt: np.einsum is fast, ~0.1ms?
        # Let's try 1000 attempts per rank if time allows.
        attempts = 0
        while time.time() < deadline and not found:
            attempts += 1
            # Random initialization
            U = np.random.choice(search_vals, size=(n * m, target_rank))
            V = np.random.choice(search_vals, size=(m * p, target_rank))
            W = np.random.choice(search_vals, size=(p * n, target_rank))
            
            if verify(U, V, W, target_rank):
                best_result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                best_rank = target_rank
                found = True
                break
        
        if found:
            continue
        else:
            # If we didn't find one at this rank, it might be impossible or just hard.
            # Continue to lower ranks? Or stop?
            # Usually if random search fails at R, it might fail at R-1 too, but not guaranteed.
            # However, if we are stuck, we might want to try a different strategy or stop.
            # For now, we break to save time for the next rank if we were iterating downwards, 
            # but here we are iterating downwards in rank (32, 31...). 
            # If 32 fails, we try 31.
            pass

    return best_result
# EVOLVE-BLOCK-END
