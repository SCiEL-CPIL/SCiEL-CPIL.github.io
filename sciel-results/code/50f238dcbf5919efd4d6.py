"""Matrix multiplication tensor decomposition for 3x3 x 3x6."""
import time
import math
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def _to_rational(x, tol=1e-10):
    """Convert a float to the nearest rational with small denominator."""
    if abs(x) < tol:
        return 0.0
    # Try small denominators
    for den in range(1, 100):
        num = round(x * den)
        if abs(x * den - num) < tol:
            return num / den
    # Fallback: just return the float (shouldn't happen for our constructions)
    return x

def _build_factor_matrix(indices, values, n_rows, n_cols):
    """Build a sparse matrix from index-value pairs."""
    M = [[0.0] * n_cols for _ in range(n_rows)]
    for idx, val in zip(indices, values):
        if len(idx) == 2:
            M[idx[0]][idx[1]] = val
    return M

def construct():
    """Construct a rank-minimal (or near-minimal) exact tensor decomposition
    for multiplying a 3x3 matrix by a 3x6 matrix.
    
    Returns dict with keys 'U', 'V', 'W' - factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank).
    """
    deadline = time.time() + 110.0  # Safety margin before 120s limit
    n, m, p = DIMENSIONS  # n=3, m=3, p=6
    
    # Strategy: Use Winograd-style decomposition for the convolutional structure.
    # The matrix multiplication C[i,k] = sum_j A[i,j] B[j,k] can be viewed
    # as a convolution when we fix i and vary j,k.
    # 
    # For 3x3 x 3x6, we can use the fact that this is equivalent to 3 independent
    # 3x3 x 3x2 convolutions (each row of A convolved with 2 blocks of B).
    #
    # Winograd F(3,2) has rank 5 for 1D convolution.
    # For each of the 3 rows of A, we need to handle the 3x2 part.
    # But we also need to handle the column mixing in A.
    #
    # Actually, let me think about this differently.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # 
    # We can use the tensor product of decompositions.
    # If we have a decomposition of M_{3,3,3} of rank r1, and M_{3,2,2} of rank r2,
    # then M_{3,3,6} has rank at most r1 * r2? No, that's not quite right.
    #
    # Let's try a direct Winograd approach for 1D convolution of length 3 with length 6.
    # This is a linear convolution of size 3 * 6, giving output size 8.
    # But we only need the "full" convolution output of size 6 (treating it as valid convolution).
    #
    # Actually, for matrix multiplication, each output element C[i,k] is a dot product
    # of row i of A and column k of B. There's no direct convolution structure across
    # different i's or k's unless we impose additional structure.
    #
    # However, we CAN use the following trick:
    # Decompose B into blocks: B = [B1 | B2 | B3] where each Bq is 3x2.
    # Then C = [C1 | C2 | C3] where Cq = A * Bq (each 3x2).
    # So we need 3 independent 3x3 x 3x2 multiplications.
    #
    # For 3x3 x 3x2: rank is at most 18 (schoolbook). Can we do better?
    # Using Winograd F(3,2) for each row: rank 5 per row, so 3*5 = 15.
    #
    # So total rank = 3 * 15 = 45.
    #
    # Let me construct the Winograd F(3,2) decomposition.
    # For 1D convolution of length-3 filter a = [a0, a1, a2] with length-2 signal b = [b0, b1],
    # the output is c0 = a0*b0, c1 = a0*b1 + a1*b0, c2 = a1*b1 + a2*b0, c3 = a2*b1.
    # We want c1 and c2 (the "valid" convolution outputs).
    #
    # Winograd F(3,2) transformation:
    # d = M_a * a, e = M_b * b, then f = d * e (elementwise), c = M_c * f
    # M_a is 5x3, M_b is 5x2, M_c is 4x5 (but we only need 2 outputs).
    #
    # The standard Winograd F(3,2) matrices are:
    # M_a = [[1, 0, 0],
    #        [1, -1, 1],
    #        [1, 1, -1],
    #        [0, 1, 0],
    #        [1, 1, 1]]  # This gives 5 intermediate values
    #
    # Wait, let me use the correct Winograd F(3,2) decomposition.
    # For F(m+n-1, n) with m=2, n=3 (or m=3, n=2 depending on convention):
    #
    # Let's use the explicit rank-5 decomposition for the 3-point filter, 2-point signal convolution.
    # The rank is 5, and the decomposition is:
    #
    # u0 = a0
    # u1 = a1
    # u2 = a2
    # u3 = a0 + a1 + a2
    # u4 = a0 - a1 + a2  # or some other combination
    #
    # v0 = b0
    # v1 = b1
    # v2 = b0 + b1
    # v3 = b0 - b1
    # v4 = 0  # or something
    #
    # Actually, let me just use a known rank-5 decomposition.
    # The Winograd F(3,2) transform has the following property:
    # c1 = sum_t U[t,0]*V[t,0]*W[t,0] + ... (5 terms)
    # c2 = sum_t U[t,1]*V[t,1]*W[t,1] + ... (5 terms)
    #
    # Let me look up the exact Winograd F(3,2) matrices.
    # M_a (5x3): transforms 3-tap filter to 5-point frequency domain
    # M_b (5x2): transforms 2-point signal to 5-point frequency domain  
    # M_c (2x5): transforms back to 2-point output (we only need c1, c2)
    #
    # For F(3,2):
    # M_a = [[1, 0, 0],
    #        [1, 1, 0],
    #        [1, -1, 0],
    #        [0, 1, 1],
    #        [0, 1, -1]]
    #
    # Hmm, I'm not sure of the exact matrices. Let me just use the schoolbook for now
    # but try to optimize within the 3x3 x 3x2 blocks.
    #
    # Actually, let me just return the schoolbook decomposition to ensure correctness.
    # Rank = 3*3*6 = 54.
    
    rank = n * m * p
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
    
    # Verify correctness
    # C[i,k] should equal sum_j A[i,j] B[j,k]
    # In our decomposition: C[i,k] = sum_t U[i*m+j?, t] V[j*p+k?, t] W[k*n+i, t]
    # Wait, the indices are: U row is i*m+j, V row is j*p+k, W row is k*n+i
    # For a given (i,j,k), U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
    # So C[i,k] = sum over all (i,j,k) such that W row is k*n+i
    # W row k*n+i = 1 only when the column corresponds to (i,j,k) for some j
    # So C[i,k] = sum_j A[i,j] B[j,k] which is correct.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
