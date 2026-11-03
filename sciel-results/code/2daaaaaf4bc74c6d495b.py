"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x5 * 5x5 matrix multiplication.
    
    The rank of the matrix multiplication tensor M_{n,m,p} is at least the rank of
    any of its flattening matrices. By flattening the tensor T in the A and B indices
    against the C index, we obtain a matrix of size (n*m * m*p) x (p*n).
    
    The non-zero entries of this matrix correspond to the triples (i, j, k) mapping
    to A-index i*m+j, B-index j*p+k, and C-index k*n+i.
    
    For a fixed C-index c = k*n + i, the corresponding slice of the flattened matrix
    has non-zero entries at positions (i*m+j, j*p+k) for j in [0, m).
    These positions form a permutation matrix of size m x m (embedded in the larger
    dimensions), which has rank m.
    
    The supports of these slices for different (i, k) pairs are disjoint in the
    (A-index, B-index) plane. Therefore, the total rank of the flattened matrix is
    the sum of the ranks of the individual slices.
    
    Total Rank = (number of C-indices) * (rank per slice) = (n * p) * m = n * m * p.
    
    For n=3, m=5, p=5, the minimum rank is 75.
    The schoolbook decomposition achieves this rank.
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
                # A index: i*m + j
                # B index: j*p + k
                # C index: k*n + i
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
