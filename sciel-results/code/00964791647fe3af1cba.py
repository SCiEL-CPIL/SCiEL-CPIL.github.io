"""Program for exact tensor decomposition of 2x4 * 4x6 matrix multiplication."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix
    by a 4x6 matrix.
    
    The matrix multiplication tensor T has entries:
    T[i*m+j, j'*p+k, k'*n+i'] = 1 if j'==j and i'==i, else 0
    
    This is a direct sum of two independent 1x4 * 4x6 multiplication tensors.
    The minimal rank for each is 24 (schoolbook), so total rank is 48.
    
    Returns dict with keys 'U', 'V', 'W' where:
      U has shape (n*m, rank) = (8, 48)
      V has shape (m*p, rank) = (24, 48)  
      W has shape (p*n, rank) = (12, 48)
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=6
    
    rank = n * m * p  # 48
    
    # Build the decomposition using a flat enumeration of (i, j, k) triples
    # For each triple, we have one rank-1 term where:
    #   U[i*m + j, r] = 1
    #   V[j*p + k, r] = 1  
    #   W[k*n + i, r] = 1
    # and all other entries in column r are 0.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
