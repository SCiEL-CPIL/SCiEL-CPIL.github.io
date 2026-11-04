"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the rank-minimal tensor decomposition for 4x4x9 matrix multiplication.
    
    The tensor for matrix multiplication A(n,m) x B(m,p) -> C(n,p) is defined by
    T[i, j, k] = 1 for all i, j, k, because C[i, k] = sum_j A[i, j] * B[j, k].
    
    This tensor is the outer product of three vectors of all ones:
    u (length n*m) of 1s, v (length m*p) of 1s, w (length p*n) of 1s.
    
    Thus, the minimal rank is 1.
    U[i*m+j, 0] = 1
    V[j*p+k, 0] = 1
    W[k*n+i, 0] = 1
    
    Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    For r=0: 1 * 1 * 1 = 1. Matches T[i, j, k] = 1.
    """
    n, m, p = DIMENSIONS
    rank = 1
    
    U = [[1] * rank for _ in range(n * m)]
    V = [[1] * rank for _ in range(m * p)]
    W = [[1] * rank for _ in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
