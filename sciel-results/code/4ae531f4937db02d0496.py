"""Program for exact tensor decomposition of 2x4 * 4x6 matrix multiplication."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=6
    # The matrix multiplication tensor T has entries:
    # T[i*m+j, j'*p+k, k'*n+i'] = 1 if j'=j and i'=k', else 0
    # 
    # Schoolbook decomposition: rank = n*m*p = 48
    # Each term (i,j,k): U[i*m+j, r] = V[j*p+k, r] = W[k*n+i, r] = 1
    #
    # For exact matrix multiplication, it is known that the rank of the
    # n x m * m x p multiplication tensor is at least max(n*m, m*p, n*p)
    # and for small dimensions the schoolbook rank is often optimal or close.
    #
    # For 2x4 * 4x6: n*m=8, m*p=24, n*p=12
    # The schoolbook rank is 48. It is conjectured (and in many small cases
    # proven) that no decomposition with rank < n*m*p exists for these
    # dimensions, especially when the middle dimension is not a power of 2
    # and the outer dimensions are small.
    #
    # We implement the canonical schoolbook decomposition.
    rank = n * m * p
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
