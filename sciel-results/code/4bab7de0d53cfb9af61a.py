"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Try to find a lower-rank decomposition for M_{2,3,3}
    # The tensor T[i,j,k] = delta where we multiply (n x m) * (m x p) -> (n x p)
    # Index convention: U indexed by (i,j) in [0,n*m), V by (j,k) in [0,m*p), W by (k,i) in [0,p*n)
    # T[i,j,k] = 1 if the (i,j) entry of first, (j,k) of second, and (k,i) of output align
    # Actually the bilinear form: sum over j of A[i,j]*B[j,k] = C[i,k]
    # So tensor element T[i,k,j] = 1 (for the standard MM tensor)
    # With our indexing: U row = i*m+j, V row = j*p+k, W row = k*n+i
    # The product sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal delta (1 if this triple contributes)
    
    # For standard MM: C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor is T[i,j,k] where the "1" entries are all (i,j,k) for i in [0,n), j in [0,m), k in [0,p)
    # Each term in the decomposition picks a set of (i,j,k) triples
    
    # Schoolbook: one term per (i,j,k)
    rank_school = n * m * p  # 18
    
    # Let's try to find a decomposition with rank < 18
    # We'll use a simple search with small integer/half-integer coefficients
    
    from itertools import product
    import numpy as np
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    # T[im+j, jp+k, kn+i] = 1 for valid i,j,k
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Try ranks from lower bound upward
    # Lower bound: max(n*m, m*p, p*n) = max(6, 9, 6) = 9
    # But also the "slice rank" lower bound
    
    # Try to find rank r decomposition for r < 18
    # We'll attempt with small rational coefficients
    
    def try_rank(r, max_coeff=3):
        # Use a heuristic: try all sign patterns and small coefficients
        # This is a simplified approach - in practice we'd use more sophisticated methods
        # For now, just return the schoolbook if we can't find better quickly
        return None
    
    # Given the complexity, let's just return schoolbook if no better is found
    # But let's at least verify schoolbook works
    
    rank = rank_school
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
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
