"""Program for matrix multiplication 3x5 x 5x6 using Strassen with fallback."""
import math

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def _strassen_2x2(A, B):
    """Multiply 2x2 matrices A and B using Strassen's algorithm."""
    a, b = A[0][0], A[0][1]
    c, d = A[1][0], A[1][1]
    e, f = B[0][0], B[0][1]
    g, h = B[1][0], B[1][1]
    
    m1 = (a + d) * (e + h)
    m2 = (c + d) * e
    m3 = a * (f - h)
    m4 = d * (g - e)
    m5 = (a + b) * h
    m6 = (c - a) * (e + f)
    m7 = (b - d) * (g + h)
    
    return [
        [m1 + m4 - m5 + m7, m3 + m5],
        [m2 + m4, m1 - m2 + m3 + m6]
    ]

def _mat_add(A, B):
    """Add two 2D matrices."""
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

def _mat_sub(A, B):
    """Subtract two 2D matrices."""
    return [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

def _pad_matrix(A, target_rows, target_cols):
    """Pad matrix A with zeros to target size."""
    rows, cols = len(A), len(A[0])
    P = [[0] * target_cols for _ in range(target_rows)]
    for i in range(rows):
        for j in range(cols):
            P[i][j] = A[i][j]
    return P

def _unpad_matrix(A, orig_rows, orig_cols):
    """Remove padding from matrix A."""
    return [row[:orig_cols] for row in A[:orig_rows]]

def _strassen_rank(n, m, p):
    """
    Estimate the rank of Strassen decomposition for n x m x p.
    We pad to the next power of 2 for all dimensions.
    The rank is 7^k where k is the log2 of the padded dimension.
    Since we need a cubic tensor, we pad to N x N x N where N is the max of next_pow2(n,m,p).
    """
    max_dim = max(n, m, p)
    # Find next power of 2
    k = 0
    while (1 << k) < max_dim:
        k += 1
    N = 1 << k
    # Rank is 7^k
    return 7 ** k

def _build_strassen_factors(n, m, p):
    """
    Build U, V, W for Strassen decomposition of n x m x p.
    Pads to N x N x N where N is next power of 2.
    """
    max_dim = max(n, m, p)
    k = 0
    while (1 << k) < max_dim:
        k += 1
    N = 1 << k
    rank = 7 ** k
    
    # We need to generate the factors for the N x N x N tensor.
    # This is complex to generate symbolically. 
    # Instead, we can use the recursive structure to build the factors.
    # However, for a general n,m,p, we just need the sub-tensor.
    
    # For simplicity and correctness, if the estimated rank is not better than schoolbook,
    # we should return None to signal fallback.
    schoolbook_rank = n * m * p
    if rank >= schoolbook_rank:
        return None
        
    # If we get here, we need to actually build the factors.
    # Generating Strassen factors for arbitrary N is non-trivial.
    # We can use a recursive generator.
    
    # Base case: 2x2x2
    if N == 2:
        # Rank 7
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        # Strassen 2x2 factors
        # m1 = (a+d)(e+h) -> U: a, d; V: e, h; W: c11, c22
        # This is getting complicated to map to global indices.
        
        # Given the complexity and the fact that for 3x5x6, Strassen rank (343) > Schoolbook (90),
        # this function will always return None for the target dimensions.
        pass
        
    return None

def construct():
    """
    Construct rank-minimal (or near-minimal) tensor decomposition for 3x5 * 5x6.
    Tries Strassen first, falls back to schoolbook.
    """
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Try Strassen
    strassen_factors = _build_strassen_factors(n, m, p)
    if strassen_factors:
        return strassen_factors
        
    # Fallback to Schoolbook
    rank = schoolbook_rank
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
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
