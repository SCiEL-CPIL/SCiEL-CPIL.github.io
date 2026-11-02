"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    WFT-based decomposition for (2x4) * (4x7) -> (2x7).
    
    Strategy: Use the cyclic group Z_7 for the output dimension (p=7).
    Embed the problem so that the 7 output columns are treated as a cyclic
    convolution. The middle dimension m=4 is zero-padded to length 7 in the
    Fourier domain.
    
    For each Fourier mode v in Z_7, we compute a 2x4 matrix-vector product
    with the Fourier-transformed B. This gives rank = 7 * 8 = 56 naively,
    but we can exploit that A is constant across Fourier modes.
    
    Actually, let's use a more direct approach: pair up the 4 middle indices
    using the identity (a+b)(c+d) to reduce multiplications.
    
    For each output column k, we need:
      C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k] + A[0,3]*B[3,k]
      C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k] + A[1,3]*B[3,k]
    
    Using (a+b)(c+d) = ac+ad+bc+bd, we can group:
      Let P0 = (A[0,0]+A[0,1])*(B[0,k]+B[1,k]) = A[0,0]B[0,k]+A[0,0]B[1,k]+A[0,1]B[0,k]+A[0,1]B[1,k]
      This introduces cross terms, so it doesn't directly help.
    
    Given the difficulty of finding a rank < 56 decomposition with simple
    integer entries, we return the schoolbook decomposition with rank 56.
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
# EVOLVE-BLOCK-END
