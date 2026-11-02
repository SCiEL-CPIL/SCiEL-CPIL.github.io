"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the factor matrices for the tensor decomposition of matrix multiplication.
    Dimensions: n x m * m x p = n x p
    n=2, m=4, p=8
    Uses a block-based approach with Strassen for 2x2x2 blocks to achieve rank 56.
    """
    n, m, p = DIMENSIONS
    rank = 56
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for q in range(2):
        for s in range(4):
            # Define the 7 Strassen products for this 2x2x2 block
            # A is 2x2: indices (0, 2q), (0, 2q+1), (1, 2q), (1, 2q+1)
            # B is 2x2: indices (2q, 2s), (2q, 2s+1), (2q+1, 2s), (2q+1, 2s+1)
            # C is 2x2: indices (0, 2s), (0, 2s+1), (1, 2s), (1, 2s+1)
            
            # P1: (a+d) * (e+h)
            # a = A[0, 2q], d = A[1, 2q+1]
            # e = B[2q, 2s], h = B[2q+1, 2s+1]
            U[0 * m + 2 * q][col] += 1
            U[1 * m + 2 * q + 1][col] += 1
            V[2 * q * p + 2 * s][col] += 1
            V[(2 * q + 1) * p + 2 * s + 1][col] += 1
            # C00 += P1, C11 += P1
            W[2 * s * n + 0][col] += 1
            W[(2 * s + 1) * n + 1][col] += 1
            col += 1
            
            # P2: (c+d) * e
            # c = A[1, 2q], d = A[1, 2q+1]
            # e = B[2q, 2s]
            U[1 * m + 2 * q][col] += 1
            U[1 * m + 2 * q + 1][col] += 1
            V[2 * q * p + 2 * s][col] += 1
            # C10 += P2
            W[2 * s * n + 1][col] += 1
            col += 1
            
            # P3: a * (f-h)
            # a = A[0, 2q]
            # f = B[2q, 2s+1], h = B[2q+1, 2s+1]
            U[0 * m + 2 * q][col] += 1
            V[2 * q * p + 2 * s + 1][col] += 1
            V[(2 * q + 1) * p + 2 * s + 1][col] -= 1
            # C01 += P3, C11 += P3
            W[(2 * s + 1) * n + 0][col] += 1
            W[(2 * s + 1) * n + 1][col] += 1
            col += 1
            
            # P4: d * (g-e)
            # d = A[1, 2q+1]
            # g = B[2q+1, 2s], e = B[2q, 2s]
            U[1 * m + 2 * q + 1][col] += 1
            V[(2 * q + 1) * p + 2 * s][col] += 1
            V[2 * q * p + 2 * s][col] -= 1
            # C00 += P4, C10 += P4
            W[2 * s * n + 0][col] += 1
            W[2 * s * n + 1][col] += 1
            col += 1
            
            # P5: (a+b) * h
            # a = A[0, 2q], b = A[0, 2q+1]
            # h = B[2q+1, 2s+1]
            U[0 * m + 2 * q][col] += 1
            U[0 * m + 2 * q + 1][col] += 1
            V[(2 * q + 1) * p + 2 * s + 1][col] += 1
            # C00 -= P5, C01 += P5
            W[2 * s * n + 0][col] -= 1
            W[(2 * s + 1) * n + 0][col] += 1
            col += 1
            
            # P6: (c-a) * (e+f)
            # c = A[1, 2q], a = A[0, 2q]
            # e = B[2q, 2s], f = B[2q, 2s+1]
            U[1 * m + 2 * q][col] += 1
            U[0 * m + 2 * q][col] -= 1
            V[2 * q * p + 2 * s][col] += 1
            V[2 * q * p + 2 * s + 1][col] += 1
            # C11 += P6
            W[(2 * s + 1) * n + 1][col] += 1
            col += 1
            
            # P7: (b-d) * (g+h)
            # b = A[0, 2q+1], d = A[1, 2q+1]
            # g = B[2q+1, 2s], h = B[2q+1, 2s+1]
            U[0 * m + 2 * q + 1][col] += 1
            U[1 * m + 2 * q + 1][col] -= 1
            V[(2 * q + 1) * p + 2 * s][col] += 1
            V[(2 * q + 1) * p + 2 * s + 1][col] += 1
            # C00 += P7
            W[2 * s * n + 0][col] += 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
