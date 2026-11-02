"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build the target tensor T[a, b, c]
    # T[a, b, c] = 1 if a%m == b//p and b%p == c//n and c%n == a//m
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Try to find a low-rank decomposition
    # Start with schoolbook rank
    best_rank = nm * mp * pn  # This is wrong, schoolbook is n*m*p
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Schoolbook decomposition
    U_school = np.zeros((nm, best_rank))
    V_school = np.zeros((mp, best_rank))
    W_school = np.zeros((pn, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # Try to find lower rank decompositions
    deadline = time.time() + 140  # Leave 10s margin
    
    # Try ranks from best_rank-1 down to some minimum
    # Use a simple iterative method
    for r in range(best_rank - 1, max(n, m, p) - 1, -1):
        if time.time() > deadline:
            break
        
        # Try a few random initializations
        for trial in range(5):
            if time.time() > deadline:
                break
            
            np.random.seed(42 + trial)
            U = np.random.randn(nm, r)
            V = np.random.randn(mp, r)
            W = np.random.randn(pn, r)
            
            # Try to refine using a simple iterative approach
            # This is a toy approach; real optimization would be more complex
            # For now, just check if the random init happens to work (unlikely)
            
            # Check if this decomposition is exact
            # Compute the reconstructed tensor
            # T_recon[a,b,c] = sum_k U[a,k] * V[b,k] * W[c,k]
            # This is expensive for large r, so only do it for small r
            if r < 50:
                T_recon = np.einsum('ak,bk,ck->abc', U, V, W)
                if np.allclose(T_recon, T, atol=1e-10):
                    best_rank = r
                    best_U = U
                    best_V = V
                    best_W = W
                    break
        else:
            continue
        break
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
