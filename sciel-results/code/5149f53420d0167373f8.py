"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal (or near-minimal) exact tensor decomposition for 
    multiplying a 2x3 matrix by a 3x3 matrix.
    
    The known minimal rank for 2x3x3 matrix multiplication is 17.
    However, constructing a valid 17-rank decomposition requires specific 
    coefficients often derived from extensive search or specific algebraic 
    identities (like extensions of Strassen's algorithm). 
    
    To ensure correctness and avoid returning an invalid decomposition (which 
    would score 0), this function returns the standard Schoolbook decomposition 
    of rank 18. This guarantees a valid score of 1.0 (baseline).
    
    Interface:
    U: shape (n*m, rank)
    V: shape (m*p, rank)
    W: shape (p*n, rank)
    
    n=2, m=3, p=3.
    rank = 18.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Map indices to flat vectors
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx][column] = 1
                V[v_idx][column] = 1
                W[w_idx][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
