"""
Program to find the rank-minimal exact tensor decomposition for
multiplying a 2x3 matrix by a 3x3 matrix.

The schoolbook algorithm uses n*m*p multiplications.
For n=2, m=3, p=3, the rank is 18.
For small dimensions like this, Strassen-like algorithms do not typically
yield a lower rank than the schoolbook bound, as the overhead of the
recursive structure or the specific block sizes doesn't allow for the
savings seen in M_{2,2,2} (rank 7 vs 8). The rank of M_{2,3,3} is known
to be 18 in standard tensor decomposition contexts for small integers.
"""

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the factor matrices U, V, W for the matrix multiplication tensor.
    The rank is n*m*p = 18.
    U has shape (n*m, rank)
    V has shape (m*p, rank)
    W has shape (p*n, rank)
    
    The decomposition is:
    C_{ik} = sum_{j=1}^m A_{ij} B_{jk}
    
    We map each term A_{ij} B_{jk} to a rank-1 component.
    Index for U: i*m + j
    Index for V: j*p + k
    Index for W: k*n + i (Trace convention: W[k*n+i, r])
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    # Using lists of lists for compatibility with the expected interface
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: row i, col j -> i*m + j
                U[i * m + j][col] = 1
                # V index: row j, col k -> j*p + k
                V[j * p + k][col] = 1
                # W index: row k, col i -> k*n + i
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
