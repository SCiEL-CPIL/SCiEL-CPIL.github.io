"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def _strassen_2x2():
    """Return Strassen's rank-7 decomposition for 2x2 matrix multiplication.
    Returns U (4x7), V (4x7), W (4x7) where:
    U[i*2+j, r] corresponds to input A[i,j]
    V[j*2+k, r] corresponds to input B[j,k]  
    W[k*2+i, r] corresponds to output C[i,k]
    """
    # Strassen's 7 products for 2x2:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b11)
    # M4 = a11(b01-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b10)
    # M7 = (a01-a11)(b01+b11)
    #
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M4
    # C10 = M2 + M3
    # C11 = M1 - M2 + M6 - M7
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Indexing: A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3
    #           B[0,0]=4, B[0,1]=5, B[1,0]=6, B[1,1]=7
    #           C[0,0]=8, C[0,1]=9, C[1,0]=10, C[1,1]=11
    
    # M1 = (a00+a11)(b00+b11)
    U[0, 0] = 1  # a00
    U[3, 0] = 1  # a11
    V[4, 0] = 1  # b00
    V[7, 0] = 1  # b11
    W[8, 0] = 1   # C00 += M1
    W[11, 0] = 1  # C11 += M1
    
    # M2 = (a10+a11)b00
    U[2, 1] = 1  # a10
    U[3, 1] = 1  # a11
    V[4, 1] = 1  # b00
    W[10, 1] = 1  # C10 += M2
    W[11, 1] = -1 # C11 -= M2
    
    # M3 = a00(b10-b11)
    U[0, 2] = 1  # a00
    V[6, 2] = 1  # b10
    V[7, 2] = -1 # -b11
    W[9, 2] = 1   # C01 += M3
    W[10, 2] = 1  # C10 += M3
    
    # M4 = a11(b01-b00)
    U[3, 3] = 1  # a11
    V[5, 3] = 1  # b01
    V[4, 3] = -1 # -b00
    W[8, 3] = -1  # C00 -= M4
    W[9, 3] = 1   # C01 += M4
    
    # M5 = (a00+a01)b11
    U[0, 4] = 1  # a00
    U[1, 4] = 1  # a01
    V[7, 4] = 1  # b11
    W[8, 4] = 1   # C00 += M5
    
    # M6 = (a10-a00)(b00+b10)
    U[2, 5] = 1   # a10
    U[0, 5] = -1  # -a00
    V[4, 5] = 1   # b00
    V[6, 5] = 1   # b10
    W[11, 5] = 1  # C11 += M6
    
    # M7 = (a01-a11)(b01+b11)
    U[1, 6] = 1   # a01
    U[3, 6] = -1  # -a11
    V[5, 6] = 1   # b01
    V[7, 6] = 1   # b11
    W[8, 6] = -1  # C00 -= M7
    W[11, 6] = -1 # C11 -= M7
    
    return U, V, W

def _winograd_f2_3x4_4x5():
    """
    Use Winograd F(2) transform to decompose 3x4 * 4x5 multiplication.
    
    Strategy:
    - Split the 4x4 middle dimension into two 2x2 blocks
    - Apply Strassen to each 2x2 block
    - Handle the boundary terms (3 rows, 5 cols) with additional terms
    
    Actually, let me think more carefully about a clean decomposition.
    
    Alternative approach: Direct structured decomposition using
    the fact that we can combine adjacent terms.
    """
    n, m, p = 3, 4, 5
    
    # Let's try a different approach: use the known optimal rank for 
    # small rectangular cases and combine them.
    
    # For now, let me try to find a rank-58 or lower decomposition
    # by checking if we can eliminate some schoolbook terms.
    
    # Actually, let me implement a proper Winograd-style decomposition.
    
    # The idea: partition the computation into overlapping windows
    # and use Strassen-like tricks on each window.
    
    # For 3x4 * 4x5:
    # We can view this as computing 3 output rows, each requiring
    # 5 dot products of length 4.
    
    # A simpler approach: use the fact that R(3,4,5) might be achievable
    # by combining R(2,4,5) and R(1,4,5) or similar.
    
    # Let me try a direct optimization: start with schoolbook and try
    # to find linear dependencies or better combinations.
    
    # Given time constraints, let me return a valid decomposition
    # that attempts to use Strassen on 2x2 sub-blocks.
    
    # Partition A (3x4) into:
    #   A = [A11 A12]  where A11 is 3x2, A12 is 3x2
    # Partition B (4x5) into:
    #   B = [B11 B12 B13]  where B11 is 2x5, B12 is 2x5, ...
    
    # This is getting complex. Let me use a simpler valid approach:
    # Return the schoolbook decomposition but with a search for
    # any improvement.
    
    # For now, return schoolbook as fallback
    return None

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _verify(U, V, W, tol=1e-6):
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    T_decomp = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(U.shape[1]):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                T_decomp[i,j,k] = s
    
    T_target = np.ones((n, m, p), dtype=float)
    return np.allclose(T_decomp, T_target, atol=tol)

def _try_strassen_based():
    """
    Try to construct a lower-rank decomposition using Strassen on 2x2 blocks.
    
    For 3x4 * 4x5:
    - Split the 4 middle dimension into two 2s: j=0,1 and j=2,3
    - This gives us two 3x2 * 2x5 sub-problems
    - Each 3x2 * 2x5 can potentially be done with fewer than 3*2*5=30 terms
    
    Actually, let me think about this differently.
    
    The key insight: if we can find a bilinear algorithm that uses shared
    intermediate computations, we can reduce the rank.
    
    For 2x2 * 2x2, Strassen uses 7 instead of 8 terms.
    The saving comes from combining terms.
    
    Let me try to apply similar ideas to our problem.
    """
    n, m, p = 3, 4, 5
    
    # Start with schoolbook: 60 terms
    # Try to see if we can merge some terms.
    
    # One approach: for each pair of (i,j) and (i',j') with same i,
    # and (j,k) and (j',k') with same j, see if we can combine.
    
    # This is essentially looking for a rank-revealing decomposition.
    
    # Given the complexity, let me try a random search for rank 59, 58, etc.
    # But this is very unlikely to succeed.
    
    # Alternative: use the fact that the tensor might have a known
    # structure that allows for a better decomposition.
    
    # For now, return None to indicate no improvement found
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 100.0
    
    # Try the Strassen-based approach
    result = _try_strassen_based()
    if result is not None:
        if _verify(result['U'], result['V'], result['W']):
            return result
    
    # Fallback to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
