"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    
    # Build the 28-term Strassen decomposition
    rank_s = 28
    U = np.zeros((n*m, rank_s))
    V = np.zeros((m*p, rank_s))
    W = np.zeros((p*n, rank_s))
    
    for q in range(4):
        c0 = 2 * q
        c1 = 2 * q + 1
        
        r = 7 * q + 0
        U[0][r] = 1; U[3][r] = 1
        V[c0][r] = 1; V[8+c1][r] = 1
        W[2*c0][r] = 1; W[2*c1+1][r] = 1
        
        r = 7 * q + 1
        U[2][r] = 1; U[3][r] = 1
        V[c0][r] = 1
        W[2*c0+1][r] = 1; W[2*c1+1][r] = -1
        
        r = 7 * q + 2
        U[0][r] = 1
        V[c1][r] = 1; V[8+c1][r] = -1
        W[2*c1][r] = 1; W[2*c1+1][r] = 1
        
        r = 7 * q + 3
        U[3][r] = 1
        V[8+c0][r] = 1; V[c0][r] = -1
        W[2*c0][r] = 1; W[2*c0+1][r] = 1
        
        r = 7 * q + 4
        U[0][r] = 1; U[1][r] = 1
        V[8+c1][r] = 1
        W[2*c0][r] = -1; W[2*c1][r] = 1
        
        r = 7 * q + 5
        U[2][r] = 1; U[0][r] = -1
        V[c0][r] = 1; V[c1][r] = 1
        W[2*c1+1][r] = 1
        
        r = 7 * q + 6
        U[1][r] = 1; U[3][r] = -1
        V[8+c0][r] = 1; V[8+c1][r] = 1
        W[2*c0][r] = 1
    
    # Check for linear dependencies among the 28 terms
    # Each term t_r is a tensor in R^{4x16x16}
    M = np.zeros((4*16*16, 28))
    for r in range(28):
        M[:,r] = np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r]).ravel()
    
    _, s, Vt = np.linalg.svd(M, full_matrices=False)
    tol = 1e-10
    rank_M = int(np.sum(s > tol))
    
    # If rank_M < 28, we could potentially reduce rank, but for now return 28
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
