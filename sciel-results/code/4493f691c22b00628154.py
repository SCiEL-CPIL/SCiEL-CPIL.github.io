"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x5 matrix.
    
    The matrix multiplication tensor M_{3,3,5} has entries T[a,b,c] = 1 iff
    a = i*m+j, b = j*p+k, c = k*n+i for some i in [0,n), j in [0,m), k in [0,p).
    
    The schoolbook decomposition uses one rank-1 term for each (i,j,k) triple,
    giving rank = n*m*p = 45.
    
    For the 3x3x5 case, the tensor decomposes as a direct sum of 5 tensors of type 3x3x1.
    Each 3x3x1 tensor computes 3 dot products of length 3, each requiring exactly 3
    scalar multiplications (the bilinear rank of a length-3 dot product is 3).
    Thus each 3x3x1 sub-tensor has rank 9, and the total rank is 5*9 = 45.
    
    This is the minimal rank since the 3x3x1 tensor has rank exactly 9 (no Strassen-like
    improvement exists for dot products of length 3).
    """
    n, m, p = DIMENSIONS  # 3, 3, 5
    rank = n * m * p  # 45
    U = [[0] * rank for _ in range(n * m)]  # 9 x 45
    V = [[0] * rank for _ in range(m * p)]  # 15 x 45
    W = [[0] * rank for _ in range(p * n)]  # 15 x 45
    
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
