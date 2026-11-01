"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # Try rank-18 structured decomposition first
    result = _rank18_decomposition(n, m, p)
    if result is not None:
        U, V, W = result
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}

def _rank18_decomposition(n, m, p):
    """
    Construct a rank-18 decomposition for 2x3 * 3x8 matrix multiplication.
    
    Strategy: Split the 8 columns of B into two blocks of 4 columns each.
    For each 2x3 * 3x4 sub-problem, use a rank-9 decomposition.
    
    For 2x3 * 3x4 (rank 9):
    A = [[a,b,c],[d,e,f]], B is 3x4, C is 2x4.
    
    The 9 bilinear forms:
    - 3 forms for C[0,0]: a*B00, b*B10, c*B20
    - 3 forms for C[1,0]: d*B00, e*B10, f*B20  
    - 3 forms for columns 1-3 using cross terms
    
    Actually, let me use a cleaner construction. For 2x3*3x4:
    We can think of it as two 2x3*3x2 problems (cols 0-1 and cols 2-3).
    Each 2x3*3x2 has rank at most 12 (schoolbook) but we can share.
    
    Let me just directly construct rank-18 for the full 2x3*3x8:
    Use 9 terms where each term handles one (i,j) pair of A with all 8 k values,
    but that's 6*8=48. Instead, group:
    
    For each of the 3 j-values (columns of A / rows of B), we have:
    - A[0,j] * B[j,0:8] contributes to C[0,0:8]
    - A[1,j] * B[j,0:8] contributes to C[1,0:8]
    
    That's 6 * 8 = 48 schoolbook terms. To get rank 18, we need to combine.
    
    Key insight: For fixed j, A[0,j]*B[j,k] and A[1,j]*B[j,k] can be combined
    if we use the same B projection. But they go to different C rows.
    
    Let me try: for each j (0,1,2) and each pair of k-values, use one term.
    3 j-values * 4 k-pairs = 12 terms. Still not 18.
    
    Alternative: use Strassen-like on the 3x3 middle. 
    For 2x3 * 3x8, think of B as 3 blocks of 8 columns... no.
    
    Let me just return None and use a different approach: solve for rank-18 directly.
    """
    if n != 2 or m != 3 or p != 8:
        return None
    
    # Direct construction: use the fact that we can write
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # Rank-18 scheme:
    # For j=0: 3 terms handling (A[0,0]*B[0,k] for k=0,1,2) + (A[1,0]*B[0,k] for k=0,1,2)
    #           = 6 terms for j=0, cols 0-2
    # For j=0: 3 terms handling (A[0,0]*B[0,k] for k=3,4,5) + (A[1,0]*B[0,k] for k=3,4,5)  
    #           = 6 terms for j=0, cols 3-5
    # ... this is getting to 24+
    #
    # Let me try a completely different approach: use the known result that
    # 2x3 * 3x8 can be done in rank 18 by splitting into 2x2 * 2x2 blocks.
    #
    # Actually, let me just implement a valid rank-18 by construction:
    # 
    # C[0,k] = a*B[0,k] + b*B[1,k] + c*B[2,k]
    # C[1,k] = d*B[0,k] + e*B[1,k] + f*B[2,k]
    #
    # For each k, we need 6 products. With 8 k-values, that's 48.
    # To reduce to 18, we need to share across k-values.
    #
    # Idea: For k in {0,1}, use one set of 9 terms. For k in {2,3}, another 9.
    # But 9+9=18, and each set handles 2 k-values.
    #
    # For 2x3 * 3x2 (k=0,1): rank 9?
    # C[0,0] = a*B00 + b*B10 + c*B20
    # C[0,1] = a*B01 + b*B11 + c*B21
    # C[1,0] = d*B00 + e*B10 + f*B20
    # C[1,1] = d*B01 + e*B11 + f*B21
    #
    # 9 terms:
    # T0: a*B00 -> C[0,0]
    # T1: b*B10 -> C[0,0]
    # T2: c*B20 -> C[0,0]
    # T3: d*B00 -> C[1,0]
    # T4: e*B10 -> C[1,0]
    # T5: f*B20 -> C[1,0]
    # T6: a*B01 -> C[0,1]
    # T7: d*B01 -> C[1,1]
    # T8: (b+c)*B11 -> C[0,1] AND (e+f)*B21 -> C[1,1]? No, that doesn't work.
    
    # This is hard. Let me just return None for now.
    return None

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Test with random matrices
    np.random.seed(42)
    for trial in range(5):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for c in range(rank):
            s_A = np.dot(U[:, c], A.flatten())
            s_B = np.dot(V[:, c], B.flatten())
            W_col = W[:, c].reshape(p, n)
            C_computed += W_col.T * (s_A * s_B)
        
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            return False
    
    return True
# EVOLVE-BLOCK-END
