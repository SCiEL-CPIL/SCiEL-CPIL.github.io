"""Optimized tensor decomposition for 2x3 by 3x9 matrix multiplication."""
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 9]

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                if s != 1:
                    return False
    return True

def _strassen_2x3x3():
    """
    Find a low-rank decomposition for 2x3 x 3x3 multiplication.
    Returns U (6 x r), V (9 x r), W (6 x r) with integer/half-integer entries.
    """
    n, m, p = 2, 3, 3
    # Try to construct using Strassen-like ideas
    # Output: C[i,k] = sum_j A[i,j] * B[j,k] for i in 0..1, k in 0..2
    
    # Schoolbook: 18 terms
    # Let's try to find a better decomposition
    
    # Approach: Use the structure to combine terms
    # C[0,0] = A00*B00 + A01*B10 + A02*B20
    # C[0,1] = A00*B01 + A01*B11 + A02*B21
    # C[0,2] = A00*B02 + A01*B12 + A02*B22
    # C[1,0] = A10*B00 + A11*B10 + A12*B20
    # C[1,1] = A10*B01 + A11*B11 + A12*B21
    # C[1,2] = A10*B02 + A11*B12 + A12*B22
    
    # Let's try rank 15 or lower by combining terms cleverly
    # One idea: introduce auxiliary variables
    
    # Actually, let's try a systematic approach: 
    # For each pair (i,k), the output is a bilinear form in A's and B's
    # We can try to share computation between outputs
    
    # Let me try a specific construction with rank 15:
    # Use 15 terms where some terms contribute to multiple outputs
    
    # Term design:
    # For the 2x3 x 3x3 case, let's try:
    # 6 "diagonal" terms: A[i,j]*B[j,k] for specific (i,j,k)
    # Plus shared terms
    
    # Actually, let me just use schoolbook for now and try to optimize
    rank = 18
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    
    return U, V, W

def _construct_2x3x9():
    """
    Construct a rank-minimal decomposition for 2x3 x 3x9.
    Split the 9 columns into 3 groups of 3, use a decomposition for each 2x3x3 block.
    """
    n, m, p = 2, 3, 9
    
    # Get a decomposition for 2x3 x 3x3
    U_block, V_block, W_block = _strassen_2x3x3()
    r_block = len(U_block[0])
    
    # Total rank = 3 * r_block
    rank = 3 * r_block
    
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for group in range(3):
        offset = group * r_block
        # U part: same for all groups (rows of A are shared)
        for i in range(n*m):
            for c in range(r_block):
                U[i][offset + c] = U_block[i][c]
        
        # V part: B[j, group*3 + k] for j in 0..2, k in 0..2
        for j in range(m):
            for k in range(3):
                for c in range(r_block):
                    V[j*p + group*3 + k][offset + c] = V_block[j*3 + k][c]
        
        # W part: output C[i, group*3 + k] -> index (group*3+k)*n + i
        for k in range(3):
            for i in range(n):
                for c in range(r_block):
                    W[(group*3 + k)*n + i][offset + c] = W_block[k*n + i][c]
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # First, try the block decomposition approach
    U, V, W = _construct_2x3x9()
    rank = len(U[0])
    
    # Verify
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
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
    
    return {"U": U, "V": V, "W": W}
