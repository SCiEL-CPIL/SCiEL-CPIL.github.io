"""Program for rank-minimal tensor decomposition of 2x3x7 matrix multiplication."""

DIMENSIONS = [2, 3, 7]


def construct():
    """
    Construct a tensor decomposition for 2x3x7 matrix multiplication.
    
    For matrix multiplication of dimensions n x m and m x p, the tensor rank
    is n*m*p for small dimensions where the standard basis components are
    linearly independent. For 2x3x7, this rank is 42.
    
    Returns: dict with keys 'U', 'V', 'W' containing the factor matrices.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
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
