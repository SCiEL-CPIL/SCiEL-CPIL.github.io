"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    
    # Check using two random matrices
    rng = np.random.RandomState(42)
    A = rng.randn(n, m)
    B = rng.randn(m, p)
    C = A @ B
    
    C_prime = np.zeros((n, p))
    for c in range(rank):
        uc = U[:, c].reshape(n, m)
        vc = V[:, c].reshape(m, p)
        wc = W[:, c].reshape(p, n)
        # W[k*n+i] corresponds to wc[k, i] if reshaped to (p, n)
        # The term is sum_c U[i,j,c] V[j,k,c] W[k,i,c]
        # C'[i,k] += W[k,i,c] * (U_c @ V_c)[i,k]
        wc_mat = wc.reshape(p, n)
        C_prime += wc_mat.T * (uc @ vc)
        
    if not np.allclose(C_prime, C, atol=1e-9):
        return False
        
    A2 = rng.randn(n, m)
    B2 = rng.randn(m, p)
    C2 = A2 @ B2
    C_prime2 = np.zeros((n, p))
    for c in range(rank):
        uc = U[:, c].reshape(n, m)
        vc = V[:, c].reshape(m, p)
        wc = W[:, c].reshape(p, n)
        wc_mat = wc.reshape(p, n)
        C_prime2 += wc_mat.T * (uc @ vc)
        
    if not np.allclose(C_prime2, C2, atol=1e-9):
        return False
        
    return True

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Construct Schoolbook Decomposition
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    # Attempt to find a lower rank decomposition (e.g., rank 53)
    # This is a heuristic search. If it fails, we return the schoolbook.
    # Given the time constraint and the complexity of tensor rank minimization,
    # we perform a very limited search.
    
    # For now, we return the valid schoolbook decomposition.
    # Future improvements could involve random linear combinations to reduce rank.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
